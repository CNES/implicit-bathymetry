#!/usr/bin/env python3
# =========================================================================
#   Program:   iota2
#
#   Copyright (c) CESBIO. All rights reserved.
#
#   See LICENSE for details.
#
#   This software is distributed WITHOUT ANY WARRANTY; without even
#   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#   PURPOSE.  See the above copyright notices for more information.
#
# =========================================================================
"""Implicit Bathymetry sequence builder"""

import logging
import os
import pickle
import shutil
import time
import zipfile
from abc import ABC, abstractmethod
from collections import OrderedDict
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING
from itertools import chain

from dask.base import collections_to_dsk
from dask.dot import graphviz_to_file, to_graphviz

from implicit_bathy.orchestration.common.file_utils import (
    ensure_dir,
    file_search_and,
)
from implicit_bathy.orchestration.configuration_files import (
    read_config_file as rcf,
)
from implicit_bathy.orchestration.steps.il_job_helpers import JobHelper
from implicit_bathy.orchestration.steps.implicit_bathy_step import (
    Step,
    StepContainer,
)

try:
    COMPRESSION = zipfile.ZIP_DEFLATED
except Exception:
    COMPRESSION = zipfile.ZIP_STORED

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger("distributed.worker")


class waiting_il_graph:
    """
    data class containing dask's graph ready to be build.
    """

    def __init__(
        self,
        builder,
        step_container,
        starting_step,
        ending_step,
        output_graph,
        prelaunch_function=None,
        figure_suffix: str = "",
    ):
        self.container = step_container
        self.starting_step = starting_step
        self.ending_step = ending_step
        self.prelaunch_function = prelaunch_function
        self.builder = builder
        self.output_graph = output_graph
        self.figure_graph = None
        if figure_suffix:
            self.figure_suffix = figure_suffix
        else:
            self.figure_suffix = builder.__class__.__name__

    def build_graph(self):
        """Build graph"""
        if self.prelaunch_function:
            self.prelaunch_function()
        # call every needed steps constructor
        _ = [
            step() for step in self.container[self.starting_step : self.ending_step + 1]
        ]

        figure_graph = Step.get_figure_graph()
        self.builder.figure_graph = figure_graph
        self.figure_graph = figure_graph
        self.builder.dask_figure_graphics.append((figure_graph, self.figure_suffix))

        if self.output_graph:
            self.builder.figure_graph.visualize(
                filename=self.output_graph,
                optimize_graph=True,
                collapse_outputs=True,
            )
            LOGGER.info("graph generated at : %s", self.output_graph)
        return Step.get_exec_graph()


def zip_implicit_bathy_logging_dir(il_root_dir: str, out_zip: str) -> None:
    """prepare il archive containing logs"""
    il_log_dir = Path(il_root_dir) / "logs"

    # get logs
    log_err_files = filter(
        lambda x: "html" not in x, file_search_and(il_log_dir, True, ".err")
    )
    log_out_files = filter(
        lambda x: "html" not in x, file_search_and(il_log_dir, True, ".out")
    )
    try:
        status_figures = file_search_and(il_log_dir, True, "tasks_status")
    except FileExistsError:
        status_figure = None

    with zipfile.ZipFile(out_zip, "w") as my_zip:
        for file_n in log_err_files:
            my_zip.write(
                file_n,
                file_n.replace(il_root_dir, ""),
                compress_type=COMPRESSION,
            )
        for file_n in log_out_files:
            my_zip.write(
                file_n,
                file_n.replace(il_root_dir, ""),
                compress_type=COMPRESSION,
            )
        if status_figures:
            for status_figure in status_figures:
                my_zip.write(
                    status_figure,
                    status_figure.replace(il_root_dir, ""),
                    compress_type=COMPRESSION,
                )


class il_builder(ABC):
    """
    class use to describe steps sequence and variable to use at each
    step (config)
    """

    _instances = {}

    def __init__(
        self,
        cfg: str,
        config_resources: str,
        schduler_type: str,
        csv_file: str = None,
        restart: bool = False,
        tasks_states_file: str | None = None,
        hpc_working_directory: str = "TMPDIR",
    ):
        self.config_resources = config_resources
        self.restart = restart
        self.csv_file = csv_file
        self.cfg = cfg
        self.workingDirectory = None
        self.il_cfg_params = rcf.ReadConfigFile(self.cfg)
        if hpc_working_directory:
            self.workingDirectory = os.getenv(hpc_working_directory)
        self.output_il_directory = self.il_cfg_params.config.dataprocessing.output_path
        self.log_dir = Path(self.output_il_directory) / "logs"

        max_ram_by_tasks = self.il_cfg_params.config.task_retry_limits.maximum_ram
        max_cpu_by_tasks = self.il_cfg_params.config.task_retry_limits.maximum_cpu
        max_task_retry = self.il_cfg_params.config.task_retry_limits.allowed_retry

        self.tasks_status_directory = self.output_il_directory
        self.slurm_section = self.il_cfg_params.config.slurm
        # steps definitions
        self.steps_group = OrderedDict()

        # build steps
        # self.steps = self.build_steps(self.cfg, config_ressources)
        # self.sort_step()

        self.figure_graph = None
        Step.set_tasks_status_directory(self.tasks_status_directory)
        if restart and not tasks_states_file:
            Step.restart_from_il_state(Step.tasks_status_file)
        if tasks_states_file:
            Step.restart_from_il_state(tasks_states_file)
        Step.set_allowed_task_failure(max_task_retry)
        Step.set_hardware_limits(max_cpu_by_tasks, max_ram_by_tasks)

        self.done_node_color = "#648FFF"
        self.unlaunchable_node_color = "#FFB000"
        self.fail_node_color = "#DC267F"

        Step.execution_mode = schduler_type

        self.dask_figure_graphics = []
        self.background_informations_file = Path(self.log_dir) / "run_informations.txt"

        self._instances[self.__class__.__name__] = self

    @property
    def builders_instances(self):
        """builders instances"""
        return self._instances

    @property
    def steps(self):
        """steps"""
        il_running_steps_container = self.build_steps(self.cfg, self.config_resources)
        self.check_build_steps_returns_type(il_running_steps_container)
        return il_running_steps_container

    def check_build_steps_returns_type(self, steps_containers) -> None:
        """raise an exception if the returned value of self.build_steps()
        is incorrect
        """
        if isinstance(steps_containers, list):
            for steps_container in steps_containers:
                if not isinstance(steps_container, StepContainer):
                    raise TypeError(
                        "build_steps() must return a list of StepContainer instances."
                    )
        elif not isinstance(steps_containers, StepContainer):
            raise TypeError(
                "the returned value from build_steps() method must be StepContainer"
            )

    def update_graph_status(self, dask_graph):
        """call once an error occurs on a previous dask graph,
        this function set every tasks in 'dask_graph' to
        the 'unlaunchable' status
        """
        tasks_key = dask_graph.dask.keys()
        tasks_name = [task_key.split("-")[0] for task_key in tasks_key]

        if Path(Step.tasks_status_file).exists():
            with open(Step.tasks_status_file, "rb") as tasks_status_file:
                tasks_status = pickle.load(tasks_status_file)
            tasks_status_updated = tasks_status.copy()
            for task_name in tasks_name:
                tasks_status_updated[task_name] = "unlaunchable"
            with open(Step.tasks_status_file, "wb") as tasks_status_file:
                pickle.dump(tasks_status_updated, tasks_status_file)

    def sort_step(self):
        """
        use to establish which step is going to which step group
        """
        for step_place, step in enumerate(chain.from_iterable(self.steps)):
            self.steps_group[step.step_group][step_place + 1] = {
                "description": step.func.step_description(),
                "resources_block": step.func.resources_block_name,
                "log_id": step.func.__name__,
            }

    def gen_figure_tasks_attributes(
        self,
        tasks_keys: list[str],
        il_tasks_status_dico: dict[str, str],
        rst_dir: str,
    ) -> dict:
        """generate attributes in order to colorized a dask graph, one color by task's state

        Parameters
        ----------
        tasks_keys
            dask task's key
        il_tasks_status_dico:
            dictionary build as : {'task_name': 'tasks_status'}
        rst_dir:
            directory containing all rst files
        """
        # waiting every logs are comming from scheduler
        time.sleep(10)
        failed_tasks_attributes = {
            "color": "black",
            "fontcolor": "black",
            "fillcolor": self.fail_node_color,
            "style": "filled",
        }
        succeeded_tasks_attributes = {
            "color": "black",
            "fontcolor": "black",
            "fillcolor": self.done_node_color,
            "style": "filled",
        }
        unlaunchable_tasks_attributes = {
            "color": "black",
            "fontcolor": "black",
            "fillcolor": self.unlaunchable_node_color,
            "style": "filled",
        }
        attributes = {}
        for task_name, task_status in il_tasks_status_dico.items():
            # get the dask key correspondingn to the the task_name
            for key in tasks_keys:
                if task_name == key.split("-")[0]:
                    if task_status == "done":
                        attributes[key] = succeeded_tasks_attributes.copy()
                        url = self.get_task_log(
                            task_name, std_flux="out", rst_dir=rst_dir
                        )
                    elif task_status == "failed":
                        attributes[key] = failed_tasks_attributes.copy()
                        url = self.get_task_log(
                            task_name, std_flux="err", rst_dir=rst_dir
                        )
                    else:
                        attributes[key] = unlaunchable_tasks_attributes.copy()
                        url = self.get_task_log(
                            task_name, std_flux=None, rst_dir=rst_dir
                        )
                    if url:
                        attributes[key]["URL"] = url
                    break
        # get last key
        for key in tasks_keys:
            if "ending" in key:
                attributes[key] = unlaunchable_tasks_attributes.copy()
        return attributes

    def get_task_log(self, task_name: str, std_flux: str, rst_dir: str) -> str:
        """from task's name, return it's associated log file

        Parameters
        ----------
        task_name:
            task's name
        std_flux :
            define which log file is returned : 'err' or 'out'
        rst_dir:
            directory containing all rst files
        """
        log_file = ""
        if std_flux:
            log_file = list(
                filter(
                    lambda x: std_flux in os.path.splitext(x)[-1],
                    file_search_and(
                        Path(self.output_il_directory) / "logs",
                        True,
                        task_name + ".",
                    ),
                )
            )
            if log_file:
                log_file = log_file[0]
                _, file_n = os.path.split(log_file)
                if not (Path(rst_dir) / file_n).exists():
                    shutil.copy(log_file, rst_dir)
                log_file = str(Path("..") / "html" / "source" / file_n)
            else:
                LOGGER.info(f"WARNING : cannot find log for task {task_name}")
        return log_file

    def tasks_summary(self):
        """generate a colorized a dask graph according to task's state"""
        html_dir = Path(self.output_il_directory) / "logs" / "html"
        rst_dir = Path(self.output_il_directory) / "logs" / "html" / "source"
        if Path(html_dir).exists():
            shutil.rmtree(html_dir)
        ensure_dir(html_dir)
        if Path(rst_dir).exists():
            shutil.rmtree(rst_dir)
        ensure_dir(rst_dir)
        relative_tasks_fig_files = []
        for graph_num, (dask_figure_graphic, suffix_name) in enumerate(
            self.dask_figure_graphics
        ):
            tasks_figure_file = (
                Path(self.output_il_directory)
                / "logs"
                / f"tasks_status_{suffix_name}_{graph_num + 1}.svg"
            )
            if Path(tasks_figure_file).exists():
                Path(tasks_figure_file).unlink()

            il_tasks_status_file = Step.tasks_status_file
            with open(il_tasks_status_file, "rb") as tasks_status_file:
                il_tasks_status_dico = pickle.load(tasks_status_file)

            tasks_keys = dask_figure_graphic.dask.keys()
            tasks_attributes = self.gen_figure_tasks_attributes(
                tasks_keys, il_tasks_status_dico, rst_dir
            )

            dsk = dict(collections_to_dsk([dask_figure_graphic], optimize_graph=True))
            g = to_graphviz(
                dsk, collapse_outputs=True, data_attributes=tasks_attributes
            )

            with g.subgraph() as s:
                s.attr(rank="same")
                s.node(
                    "DONE",
                    fillcolor=self.done_node_color,
                    style="filled",
                    shape="circle",
                )
                s.node(
                    "UNLAUNCH",
                    fillcolor=self.unlaunchable_node_color,
                    style="filled",
                    shape="circle",
                )
                s.node(
                    "FAIL",
                    fillcolor=self.fail_node_color,
                    style="filled",
                    shape="circle",
                )
            graphviz_to_file(g, str(tasks_figure_file), None)
            relative_tasks_fig_files.append(
                Path("..") / ".." / os.path.split(str(tasks_figure_file))[-1]
            )

    def print_step_summarize(
        self,
        start: int,
        end: int,
        show_resources: bool = False,
        checked: str = "x",
        running_step: bool = False,
        running_sym: str = "r",
        step_num_offset: int = 0,
    ):
        """
        print implicit bathy steps that will be run
        """
        summarize = (
            "Full processing include the following steps (checked steps will be run):\n"
        )
        step_position = 0
        summarize = ""
        for group in list(self.steps_group.keys()):
            if len(self.steps_group[group]) > 0:
                summarize += f"Group {group}:\n"

            for key in self.steps_group[group]:
                highlight = "[ ]"
                if key >= start and key <= end:
                    highlight = f"[{checked}]"
                if key == end and running_step:
                    highlight = f"[{running_sym}]"
                summarize += "\t {} Step {}: {}".format(
                    highlight,
                    key + step_num_offset,
                    self.steps_group[group][key]["description"],
                )
                if show_resources:
                    block_name = self.steps_group[group][key]["resources_block"]
                    resources = JobHelper.parse_resource_file(
                        block_name, self.config_resources
                    )
                    cpu = resources["cpu"]
                    gpu = resources["gpu"]
                    ram = resources["ram"]
                    walltime = resources["walltime"]
                    resource_block_name = resources["resource_block_name"]
                    resource_block_found = resources["resource_block_found"]
                    log_identifier = self.steps_group[group][key]["log_id"]
                    resource_miss = "" if resource_block_found else " -> MISSING"
                    summarize += "\n\t\t\tresources block name : {}{}\n\t\t\tcpu : {}\n\t\t\tgpu : {}\n\t\t\tram :  {}\n\t\t\twalltime : {}\n\t\t\tlog identifier : {}".format(
                        resource_block_name,
                        resource_miss,
                        cpu,
                        gpu,
                        ram,
                        walltime,
                        log_identifier,
                    )
                summarize += "\n"
                step_position += 1
        return summarize

    def get_steps_number(self):
        """
        Get steps number
        """
        start = "download"  # self.i2_cfg_params.config.chain.first_step 1
        end = "analyze"  # self.i2_cfg_params.config.chain.last_step
        start_ind = list(self.steps_group.keys()).index(start)
        end_ind = list(self.steps_group.keys()).index(end)
        steps = [
            self.steps_group[key]
            for key in list(self.steps_group.keys())[start_ind : end_ind + 1]
        ]
        step_to_compute = list(chain.from_iterable(steps))
        return step_to_compute

    def get_steps_sequence_size(self) -> int:
        """Get steps sequence size"""
        number_of_steps = 0
        for step_sequence_description in self.steps_group.values():
            if step_sequence_description:
                number_of_steps += len(step_sequence_description)
        return number_of_steps

    def get_indexes_by_container(self, first_step_index, last_step_index):
        """the purpose of this function is to get for each container of steps,
        indexes of steps to process according to user demand.
        """

        dico = {}
        len_container_buff = 0
        max_step_index = 0
        for container_num, container in enumerate(self.steps):
            dico[container_num] = OrderedDict()
            for cpt in range(len(container)):
                indice = cpt + len_container_buff
                dico[container_num][indice] = cpt
                max_step_index += 1
            len_container_buff += len(container)
        if last_step_index > max_step_index - 1:
            last_step_index = max_step_index - 1

        list_ind = []
        for container_num, container in enumerate(self.steps):
            if (
                first_step_index in dico[container_num]
                and last_step_index not in dico[container_num]
            ):
                list_ind.append(
                    (
                        dico[container_num][first_step_index],
                        dico[container_num][next(reversed(dico[container_num]))],
                    )
                )
            elif (
                last_step_index in dico[container_num]
                and first_step_index not in dico[container_num]
            ):
                list_ind.append((0, dico[container_num][last_step_index]))
            elif (
                last_step_index in dico[container_num]
                and first_step_index in dico[container_num]
            ):
                list_ind.append(
                    (
                        dico[container_num][first_step_index],
                        dico[container_num][last_step_index],
                    )
                )
            elif (
                last_step_index not in dico[container_num]
                and first_step_index not in dico[container_num]
            ):
                first_user_index = next(iter(dico[container_num]))
                last_user_index = next(reversed(dico[container_num]))

                if (
                    first_user_index > first_step_index
                    and last_user_index < last_step_index
                ):
                    first_container_index = dico[container_num][first_user_index]
                    last_container_index = dico[container_num][last_user_index]
                    list_ind.append((first_container_index, last_container_index))
                else:
                    list_ind.append([])
        return list_ind

    def get_final_il_exec_graph(
        self,
        first_step_index: int,
        last_step_index: int,
        output_figures: list[str] | None = [],
    ) -> list[waiting_il_graph]:
        """get every processing graph ready to be build.

        first_step_index:
            index of the first step to launch
        last_step_index:
            index of the last step to launch
        output_figures:
            list of file to draw processing graph
        """

        # instanciate steps which must me launched
        il_graphs = []
        indexes = self.get_indexes_by_container(first_step_index, last_step_index)
        # if user asked too much figures, limit by the number of steps container
        if output_figures:
            output_figures = output_figures[0 : len(self.steps)]

        output_figures_files = []
        if output_figures:
            for cpt, (figure_file, _) in enumerate(
                zip_longest(output_figures, range(len(self.steps)))
            ):
                if figure_file is None:
                    fig_dir, fig_name = os.path.split(output_figures[0])
                    base_name, ext = os.path.splitext(fig_name)
                    file_name = f"{base_name}_{cpt + 1}{ext}"
                    output_figures_files.append(Path(fig_dir) / file_name)
                else:
                    output_figures_files.append(figure_file)
        for container, tuple_index, out_figure in zip_longest(
            self.steps, indexes, output_figures_files
        ):
            if tuple_index:
                container_start_ind, container_end_ind = tuple_index
                il_graphs.append(
                    waiting_il_graph(
                        self,
                        container,
                        container_start_ind,
                        container_end_ind,
                        out_figure,
                        container.prelaunch_function,
                    )
                )
        return il_graphs

    def preliminary_informations(self, file_content: str) -> None:
        """save informations in a txt file"""
        ensure_dir(self.log_dir)
        with self.background_informations_file.open("w") as info_file:
            info_file.write(file_content)

    @abstractmethod
    def generate_output_directories(self, first_step_index: int, restart: bool):
        """generate needed output directories for a dedicated builder"""
        pass

    @abstractmethod
    def build_steps(
        self, cfg: str, config_ressources: str | None = None
    ) -> list[StepContainer]:
        """."""
        pass
