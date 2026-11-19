#!/usr/bin/env python3

# ==================================================================
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
# ==================================================================
"""module containing base class for implicit bathy's steps"""

import logging
import os
import pickle
import random
import subprocess
import threading
import time
import traceback
import uuid
import warnings
from functools import wraps
from subprocess import PIPE, STDOUT, Popen
from weakref import WeakValueDictionary

from pathlib import Path
import dask
import dill
from dask.distributed import Client, LocalCluster
from dask_jobqueue import PBSCluster

# from dask_jobqueue import PBSCluster
from distributed.scheduler import KilledWorker

import implicit_bathy.orchestration.common.implicit_bathy_constants as il_const
from implicit_bathy.orchestration.common.file_utils import ensure_dir
from implicit_bathy.orchestration.configuration_files import (
    read_config_file as rcf,
)
from implicit_bathy.orchestration.steps.il_job_helpers import JobHelper

warnings.simplefilter(action="ignore", category=FutureWarning)

lock = threading.Lock()
dask.config.set({"distributed.comm.timeouts.connect": "300000s"})
dask.config.set({"distributed.scheduler.allowed-failures": 0})
# dask.config.set({"worker.memory.pause": False})
dask.config.set({"distributed.worker.memory.pause": False})

LOGGER = logging.getLogger("distributed.worker")

IL_CONST = il_const.ImplicitBathyConstants()


def run_as_process(cmd: str, stdout: str, stderr: str, env=os.environ):
    """
    Launch a system command and raise an execption if fail

    Parameters
    ----------
    cmd:
        the system command to be launched
    stdout
        file to log stdout
    stderr
        file to log stderr
    env:
        the environ variable if None, os.environ is used
    logger:
        by default module LOGGER value is used
    """
    start = time.time()
    with open(stdout, "w") as out, open(stderr, "w") as err:
        proc = subprocess.Popen(cmd, env=env, shell=True, stdout=out, stderr=err)

    # Get output as strings
    out, err = proc.communicate()

    # Get return code
    rtc = proc.returncode
    end = time.time()
    with open(stdout, "a", encoding="UTF-8") as out:
        out.write("\n------- TASK SUMMARY -------\n")
        out.write(f"EXECUTION TIME : {end - start} seconds\n")
        out.write("----------------------------\n")
    # Log error code
    if rtc != 0:
        exception_msg = f"Launch command fail : {cmd} \n {out} \n {err}"
        raise Exception(exception_msg)
    return rtc


def parse_ram(ram_str: str):
    """
    Parse RAM
    """
    ram = ram_str.lower().replace(" ", "")
    if "gb" in ram:
        spliter = "gb"
    elif "mb" in ram:
        spliter = "mb"
    else:
        raise ValueError("can't parse ram parameter")
    ram_float, ram_unit = ram.split(spliter)

    return (ram_float, ram_unit, spliter)


def ram_str_to_gb_float(ram_str: str) -> int:
    """convert ram str to a float representation in gb

    ram_str_to_gb_float('10gb')
    > 10
    """
    ram_value, _, spliter = parse_ram(ram_str)

    if spliter == "mb":
        ram_float = int(ram_value) * 1024
    elif spliter == "gb":
        ram_float = int(ram_value)
    else:
        raise RuntimeError("RAM resources must have mb or gb units.")
    return ram_float


def increase_resources(cpu: int, ram: str, walltime: str):
    """definition of the strategy for increasing resources in case
       of killed task due to resources consumption. The strategy is
       : double resources cpu/ram and raise by 25% the walltime

    Parameters
    ----------
    cpu :
        initial cpu
    ram :
        initial ram expected format : '10gb' or '10mb'
    walltime :
        initial walltime, expected format : 'HH:MM:SS'

    Notes
    -----
    the returned tuple is composed as the following
    (new_cpu, new_ram, new_walltime)

    """
    # cpu
    new_cpu = cpu * 2

    # ram
    (ram_value, _, spliter) = parse_ram(ram)
    new_ram = f"{int(ram_value) * 2}{spliter}"
    if spliter == "mb":
        ram_float = int(ram_value) * 2 * 1024
    elif spliter == "gb":
        ram_float = int(ram_value) * 2
    else:
        raise RuntimeError("RAM resources must have mb or gb units.")

    # walltime
    hh, mm, ss = walltime.split(":")
    walltime_sec = 3600 * float(hh) + 60 * float(mm) + float(ss)
    increase_walltime = 1.25 * walltime_sec
    new_walltime = time.strftime("%H:%M:%S", time.gmtime(increase_walltime))

    return new_cpu, new_ram, ram_float, new_walltime


class Step:
    """
    Implicit Bathy step base class
    """

    class il_task:
        """data class to modelize a implicit bathy task"""

        il_tasks_container = {}

        def __init__(
            self,
            task_name: str,
            log_dir: str,
            execution_mode: str,
            task_parameters: dict,
            task_resources={
                "cpu": 1,
                "ram": "5Gb",
                "walltime": "01:00:00",
                "gpu": None,
                "queue": None,
            },
        ):
            # care about __closure__
            inner_resources = task_resources.copy()

            self.task_name = task_name
            if not isinstance(task_name, str):
                raise ValueError("parameter 'task_name' must be a string")
            if " " in self.task_name:
                self.task_name = self.task_name.replace(" ", "")
                LOGGER.info(
                    f"WARNING: task_name = {task_name} contains whitespaces, "
                    f"casted as {self.task_name}"
                )
            if "-" in self.task_name:
                self.task_name = self.task_name.replace("-", "")
                LOGGER.info(
                    "WARNING: task_name contains '-' character, automatically removed"
                )
            self.log_err = os.path.join(log_dir, f"{task_name}.err")
            self.log_out = os.path.join(log_dir, f"{task_name}.out")
            self.execution_mode = execution_mode
            self.parameters = task_parameters
            self.resources = inner_resources

        def is_addable_to_graph(self, graph_name: str) -> bool:
            """
            check if the task is not already present in the list of tasks.
            Indeed, a task must be unique in all graphs
            """
            if graph_name not in self.il_tasks_container:
                self.il_tasks_container[graph_name] = [self.task_name]
                is_addable = True
            else:
                buff = []
                for (
                    containers_name,
                    container_tasks,
                ) in self.il_tasks_container.items():
                    container_name_no_uuid = "_".join(containers_name.split("_")[0:-1])
                    graph_name_no_uuid = "_".join(graph_name.split("_")[0:-1])
                    if container_name_no_uuid == graph_name_no_uuid:
                        continue
                    buff.append(self.task_name not in container_tasks)
                is_addable = all(buff)
                if is_addable:
                    self.il_tasks_container[graph_name].append(self.task_name)
            return is_addable

    tiles = []
    spatial_models_distribution = {}
    spatial_models_distribution_no_sub_splits = {}

    # status file
    tasks_status_file = None

    # hardware computing limits (ram in gb)
    ram_hardware_limits: float = 0
    cpu_hardware_limits: int = 0

    # number of allowed task failure
    allowed_task_failure: int = 0

    # other run state
    older_il_state = {}

    execution_mode = None

    tasks_graph = {}
    tasks_graph_figure = {}

    def __init__(
        self,
        cfg: str,
        cfg_resources_file: str,
        csv_file: str | None = None,
        resources_block_name: str | None = None,
    ):
        """ """

        # attributes
        self.cfg = cfg
        self.step_name = self.build_step_name()
        self.step_group = ""
        parameters = rcf.ReadConfigFile(self.cfg)
        slurm_static_params = parameters.config.slurm
        # get_section_as_dataclass(
        # "slurm"
        # )
        self.job_helper = JobHelper(
            cfg_resources_file,
            resources_block_name,
            csv_file,
            slurm_static_params,
        )

        # get resources needed
        self.resources = self.job_helper.resources
        self.resource_block_name = self.resources["resource_block_name"]
        self.resource_block_found = self.resources["resource_block_found"]

        # define log path
        outputPath = parameters.config.dataprocessing.output_path
        self.logger_lvl = "DEBUG"

        log_dir = os.path.join(outputPath, "logs")
        self.log_dir = log_dir
        self.log_step_dir = os.path.join(self.log_dir, self.step_name)

        # step_tasks is a list containing
        # delayed task_launcher of this step's dependencies
        self.step_tasks = []
        self.step_tasks_figure = []
        self.il_const = il_const.ImplicitBathyConstants()
        step_container.append(self)

    def get_resources(self) -> dict:
        """get resources for a tasks (parsed from a resources
        configuration file)"""
        return {
            "cpu": self.resources["cpu"],
            "ram": self.resources["ram"],
            "walltime": self.resources["walltime"],
            "gpu": self.resources["gpu"],
            "queue": self.resources["queue"],
        }

    def set_container_name(self, name):
        """define the name of the container, useful to populate the
        right task graph"""
        self.container_name = name

    @classmethod
    def set_hardware_limits(cls, cpu: int, ram: float):
        """set maximum cpu and ram (Gb) allowed for a task"""
        cls.ram_hardware_limits = ram
        cls.cpu_hardware_limits = cpu

    @classmethod
    def set_allowed_task_failure(cls, allowed_task_failure: int):
        """."""
        cls.allowed_task_failure = allowed_task_failure

    @classmethod
    def restart_from_il_state(cls, older_il_state_file: str):
        """relaunch failed or not launched tasks"""

        if Path(older_il_state_file).exists():
            with open(older_il_state_file, "rb") as tasks_status_file:
                try:
                    cls.older_il_state = pickle.load(tasks_status_file)
                except EOFError:
                    raise EOFError(f"{older_il_state_file} is empty")

    @classmethod
    def set_tasks_status_directory(cls, directory_path: str):
        """set directory which will contains the implicit bathy state status"""
        cls.tasks_status_file = os.path.join(
            directory_path, IL_CONST.il_tasks_status_filename
        )

    @classmethod
    def set_models_spatial_information(cls, tiles, spatial_models_distribution):
        """define models distribution across tiles

        TODO : move to il_classification builder.
        """
        cls.tiles = tiles
        cls.spatial_models_distribution = spatial_models_distribution.copy()
        cls.spatial_models_distribution_no_sub_splits = {}
        for model_name, model_meta in spatial_models_distribution.items():
            model = model_name
            if "f" in model_name:
                model = model_name.split("f")[0]

            if model not in cls.spatial_models_distribution_no_sub_splits:
                cls.spatial_models_distribution_no_sub_splits[model] = model_meta
                cls.spatial_models_distribution_no_sub_splits[model]["nb_sub_model"] = 0
            cls.spatial_models_distribution_no_sub_splits[model]["nb_sub_model"] += 1

    @classmethod
    def get_exec_graph(cls):
        """Return the execution graph of the last step append."""

        def ending(
            dependencies: list,
        ) -> str:  # pylint: disable=unused-argument
            """Inconsequential function.

            Used to add "ending" node to graph
            it takes the list of dependencies in input
            and returns a string that is ignored.
            """
            return "ending"

        return dask.delayed(ending)(step_container[-1].step_tasks)

    @classmethod
    def get_figure_graph(cls):
        """get the dask graph where each node are labelized by task's name"""
        return dask.delayed(change_name("ending")(cls.task_launcher))(
            step_container[-1].step_tasks_figure
        )

    @classmethod
    def get_dependencies_keys(cls):
        """expose the state of the tasks graph by showing last
        task'name associated to tasks graph keys
        """
        tasks_dict = {}
        for key_dep, key_sub_dep in cls.tasks_graph.items():
            if key_dep not in tasks_dict:
                tasks_dict[key_dep] = {}
            if isinstance(key_sub_dep, dict):
                for key_sub_dep_name, delayed_obj in key_sub_dep.items():
                    tasks_dict[key_dep][key_sub_dep_name] = delayed_obj.key.split("-")[
                        0
                    ]
            else:
                task_name = key_sub_dep.key.split("-")[0]
                tasks_dict[key_dep] = task_name
        return tasks_dict

    def init_log(self):
        """function use to init logging in every workers"""
        log_formatter = logging.Formatter(
            "%(asctime)s [%(name)s] [%(levelname)s] - %(message)s"
        )
        logging.basicConfig(format="%(asctime)s %(message)s")
        rootLogger = logging.getLogger("distributed.worker")

        rootLogger.setLevel(self.logger_lvl)
        for handler in rootLogger.handlers:
            handler.setFormatter(log_formatter)

        # consoleHandler = logging.StreamHandler()
        # consoleHandler.setFormatter(log_formatter)
        # consoleHandler.setLevel(self.logger_lvl)
        # rootLogger.addHandler(consoleHandler)

    def init_log_local(self):
        """function use to init logging in every workers"""
        log_formatter_local = logging.Formatter(
            "%(asctime)s [%(name)s] [%(levelname)s] - %(message)s"
        )
        rootLogger = logging.getLogger("distributed.worker")
        rootLogger.setLevel(self.logger_lvl)
        # consoleHandler = logging.StreamHandler()
        # consoleHandler.setFormatter(log_formatter_local)
        # consoleHandler.setLevel(self.logger_lvl)
        # rootLogger.addHandler(consoleHandler)

        if Path(self.log_out).exists():
            Path(self.log_out).unlink()
        file_handler = logging.FileHandler(self.log_out)
        file_handler.setFormatter(log_formatter_local)
        file_handler.setLevel(self.logger_lvl)
        rootLogger.addHandler(file_handler)

        if Path(self.log_err).exists():
            Path(self.log_err).unlink()
        file_handler = logging.FileHandler(self.log_err)
        file_handler.setFormatter(log_formatter_local)
        file_handler.setLevel(self.logger_lvl)
        rootLogger.addHandler(file_handler)

    def save_task_status(self, task_name: str, task_status: str):
        """save task state in a shared dictionary"""
        if not self.tasks_status_file:
            raise RuntimeError(f"No tasks_status_file for task {task_name}")
        else:
            lock.acquire()
            success = False
            max_nb_fail = 10
            for nb_try in range(max_nb_fail):
                try:
                    time.sleep(random.randint(1, 10))
                    if not Path(self.tasks_status_file).exists():
                        with open(self.tasks_status_file, "wb") as tasks_status_file:
                            pickle.dump({task_name: task_status}, tasks_status_file)
                    else:
                        with open(self.tasks_status_file, "rb") as tasks_status_file:
                            task_completed_dic = pickle.load(tasks_status_file)

                        task_completed_dic[task_name] = task_status

                        with open(self.tasks_status_file, "wb") as tasks_status_file:
                            pickle.dump(task_completed_dic, tasks_status_file)
                    success = True
                    lock.release()
                    break
                except EOFError:
                    LOGGER.info(
                        f"EOFError DETECTECTED, task name {task_name}, "
                        f"nb try : {nb_try}"
                    )
                    time.sleep(random.randint(1, 10))
            if not success:
                raise EOFError(f"EOFError DETECTECT, task name {task_name}")

    def log_traceback(self, client, logger_name, task_tb):
        """send traceback to workers

        Parameters
        ----------
        client : dask.client
            scheduler client
        task_tb : traceback
            traceback
        """
        logger = logging.getLogger(logger_name)
        client.run(logger.log, logging.ERROR, task_tb)

    def parse_ram(self, ram_str: str) -> int:
        ram = ram_str.lower().replace(" ", "")
        if "gb" in ram:
            spliter = "gb"
        elif "mb" in ram:
            spliter = "mb"
        else:
            raise ValueError("can't parse ram parameter")
        ram_float, ram_unit = ram.split(spliter)

        return (ram_float, ram_unit, spliter)

    def ram_str_to_gb_float(self, ram_str: str) -> int:
        """convert ram str to a float representation in gb

        ram_str_to_gb_float('10gb')
        > 10
        """
        ram_float, _, spliter = self.parse_ram(ram_str)

        if spliter == "mb":
            ram_float = ram_float * 1024
        elif spliter == "gb":
            pass
        return ram_float

    def increase_resources(self, cpu: int, ram: str, walltime: str):
        """definition of the strategy for increasing resources in case
           of killed task due to resources consumption. The strategy is
           : double resources cpu/ram and raise by 25% the walltime

        Parameters
        ----------
        cpu :
            initial cpu
        ram :
            initial ram expected format : '10gb' or '10mb'
        walltime :
            initial walltime, expected format : 'HH:MM:SS'

        Notes
        -----
        the returned tuple is composed as the following
        (new_cpu, new_ram, new_walltime)

        """
        # cpu
        new_cpu = cpu * 2
        (ram_float, _, spliter) = self.parse_ram(ram)

        # ram
        new_ram = f"{(ram_float) * 2}{spliter}"
        if spliter == "mb":
            ram_float = ram_float * 2 * 1024
        elif spliter == "gb":
            ram_float = ram_float * 2

        # walltime
        hh, mm, ss = walltime.split(":")
        walltime_sec = 3600 * float(hh) + 60 * float(mm) + float(ss)
        increase_walltime = 1.25 * walltime_sec
        new_walltime = time.strftime("%H:%M:%S", time.gmtime(increase_walltime))

        return new_cpu, new_ram, ram_float, new_walltime

    def send_task(
        self,
        scheduler_type: str,
        task_kwarguments: dict,
        log_err: str,
        log_out: str,
        pbs_worker_name: str,
        resources,
    ):
        """send task to the appropriate dask scheduler

        Parameters
        ----------
        scheduler_type:
            define which kind of scheduling will be operate
            choices are 'PBS', 'cluster', 'localCluster' and 'debug'
        task_kwarguments:
            dictionary containing task function and it's arguments
        log_err:
            file to store error flux
        log_out:
            file to store std flux
        pbs_worker_name:
            task name
        resources:
            dictionnary of resources
            resources = {"cpu": 1, "ram": "1gb", "walltime":"01:00:00"}

        Notes
        -----
        the returned type is formated as
        (task_status, killed_worker)

        * task_status : indicate one of available task's status, 'done',
                        'failed' or 'unlaunchable'

        * killed_worker : inform if the task has been killed by the scheduler
                          due to resources consumption
        """
        task_kwargs = task_kwarguments.copy()

        logging.captureWarnings(True)
        log_dir, _ = os.path.split(log_err)
        if not Path(log_dir).exists():
            ensure_dir(log_dir)

        func = task_kwargs["f"]
        task_kwargs.pop("f", None)
        f_kwargs = task_kwargs

        # here we launch the fonction with all the arguments of kwargs
        killed_worker = False
        if scheduler_type in ("PBS", "Slurm"):
            self.init_log()
            job_file = log_err.replace(
                ".err",
                f".{self.job_helper.inventory_jobfile_helper[scheduler_type.lower()].extension}",
            )
            self.job_helper.write(
                scheduler_type.lower(),
                pbs_worker_name,
                job_file,
                log_out,
                log_err,
                func,
                f_kwargs,
                self.logger_lvl,
            )
            cmd = (
                f"{self.job_helper.inventory_jobfile_helper[scheduler_type.lower()].submit_cmd}"
                f" {job_file}"
            ).split(" ")
            cmd = [val for val in cmd if val != ""]
            process = Popen(cmd, shell=False, stdout=PIPE, stderr=STDOUT)
            process.wait()
            stdout, _ = process.communicate()
            task_status, killed_worker = self.job_helper.get_task_status(
                scheduler_type.lower(), log_err, log_out, str(stdout)
            )
        if scheduler_type == "cluster":
            env_vars = [
                f"export PYTHONPATH={os.environ.get('PYTHONPATH')}",
                f"export PATH={os.environ.get('PATH')}",
                f"export LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}",
                f"export OTB_APPLICATION_PATH={os.environ.get('OTB_APPLICATION_PATH')}",
                f"export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt",
                f"export GDAL_DATA={os.environ.get('GDAL_DATA')}",
                f"export GEOTIFF_CSV={os.environ.get('GEOTIFF_CSV')}",
                f"export GRASSDIR={os.environ.get('GRASSDIR')}",
                f"export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={resources['cpu']}",
            ]
            extras = [f"-N {pbs_worker_name[0:30]}"]
            if log_err is not None:
                extras.append(f"-e {log_err}")
            if log_out is not None:
                extras.append(f"-o {log_out}")
            if resources["queue"] is not None:
                extras.append(f"-q {resources['queue']}")
            if resources["gpu"] is not None:
                extras.append(f"-l select=1:ngpus={resources['gpu']}")
            if resources["qos"] is not None:
                extras.append(f"-l qos={resources['qos']}")
            cluster = PBSCluster(
                n_workers=1,
                cores=resources["cpu"],
                memory=resources["ram"],
                walltime=resources["walltime"],
                env_extra=env_vars,
                interface="ib0",
                silence_logs="error",
                processes=1,
                nanny=False,
                job_extra=extras,
                local_directory="$TMPDIR",
            )
            client = Client(cluster)
            client.wait_for_workers(1)
            client.register_worker_callbacks(self.init_log)

            for _, worker_meta in client.scheduler_info()["workers"].items():
                working_dir = os.path.split(worker_meta["local_directory"])[0]
            f_kwargs = self.set_working_dir_parameter(f_kwargs, working_dir)

            def set_allowed_failures(dask_scheduler):
                dask_scheduler.allowed_failures = 0

            client.run_on_scheduler(set_allowed_failures)
            sub_results = client.submit(func, **f_kwargs)
            try:
                sub_results.result()
                task_status = "done"
            except KilledWorker:
                task_status = "failed"
                killed_worker = True
            except Exception as task_exception:
                task_status = "failed"
                message = f"Implicit Bathymetry found python exception :{type(task_exception).__name__} "
                self.log_traceback(client, pbs_worker_name, message)
            finally:
                time.sleep(20)
                client.close()
                client.shutdown()
        if scheduler_type == "localCluster":
            cluster = LocalCluster(
                n_workers=1,
                threads_per_worker=resources["cpu"],
                memory_limit=resources["ram"],
                processes=True,
                silence_logs="ERROR",
            )
            client = Client(cluster)
            client.wait_for_workers(1)
            serialized_file = log_err.replace(".err", ".dill")
            with open(serialized_file, "wb") as file_to_dump:
                dill.dump((func, f_kwargs, self.logger_lvl), file_to_dump)
            cmd = (
                f"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={resources['cpu']} "
                f"task_launcher.py -dill_file {serialized_file}"
            )
            sub_results = client.submit(
                run_as_process, cmd, stdout=log_out, stderr=log_err
            )
            try:
                sub_results.result()
                task_status = "done"
            except KilledWorker:
                task_status = "failed"
                killed_worker = True
            except Exception:
                task_status = "failed"
                LOGGER.error(
                    f"the task '{pbs_worker_name}' has failed. "
                    f"Please check logs '{log_err}' and '{log_out}' "
                    "for more details"
                )
            finally:
                client.close()
                client.shutdown()
        if scheduler_type == "debug":
            self.log_err = log_err
            self.log_out = log_out
            self.init_log_local()
            try:
                task_status = "done"
                func(**f_kwargs)
            except Exception:
                traceback.print_exc()
                task_status = "failed"
        return task_status, killed_worker

    def task_launcher(
        self,
        *dependencies,
        task_name: str | None = None,
        log_err: str | None = None,
        log_out: str | None = None,
        scheduler_type: str = "debug",
        pbs_worker_name: str = "il-worker",
        resources={
            "cpu": 1,
            "ram": "5Gb",
            "walltime": "01:00:00",
        },
        **kwargs,
    ) -> str:
        """this function encapsulate tasks to be delayed

        *dependencies: dask.delayed dependencies
           captures all arguments (after self) into a list of tasks dependencies
           each dependency is a delayed task_launcher or its string output value

        il_log_dir :
            path to a directory which will contains tasks files
        scheduler_type :
            define which kind of scheduling will be operate
            choices are 'PBS', 'cluster', 'localCluster' and 'debug'
        resources :
            dictionnary of resources
            resources = {"cpu": 1, "ram": "1gb", "walltime":"01:00:00"}
        Notes
        -----
        the return value represent the task's status and is in
          ('done','failed','unlaunchable')
        """
        if task_name and self.older_il_state:
            if (
                task_name in self.older_il_state
                and self.older_il_state[task_name] == "done"
            ):
                return "done"
        dependencies_complete = all(dep == "done" for dep in dependencies)
        if task_name and dependencies and not dependencies_complete:
            task_status = "unlaunchable"
            self.save_task_status(task_name, task_status)
            return task_status

        task_kwargs = kwargs.copy().get("task_kwargs", None)

        if task_kwargs:
            max_retry = self.allowed_task_failure
            max_ram = self.ram_hardware_limits
            max_cpu = self.cpu_hardware_limits

            cpu = resources["cpu"]
            ram = resources["ram"]
            walltime = resources["walltime"]

            task_status = ""
            nb_try = 0
            ram_gb = ram_str_to_gb_float(ram)

            if max_retry == 0:
                nb_try = -1

            not_consistent_msg = ""
            resources_consistancy = True
            if cpu > max_cpu:
                not_consistent_msg = (
                    f"task : {task_name}, allowed cpu = {cpu} "
                    f"is not consistent with cpu hardware limit : {max_cpu} "
                )
            if ram_gb > max_ram:
                not_consistent_msg += (
                    f"task : {task_name}, allowed ram = {ram_gb} "
                    f"is not consistent with ram hardware limit : {max_ram}"
                )
            if not_consistent_msg:
                not_consistent_msg = (
                    f"{not_consistent_msg} please refer to "
                    "configuration resources file and "
                    "'task_retry_limits' configuration file parameters"
                )
                LOGGER.error(not_consistent_msg)
                task_status = "failed"
                resources_consistancy = False

            while (
                resources_consistancy
                and nb_try < max_retry
                and ram_gb <= max_ram
                and cpu <= max_cpu
            ):
                task_status, killed_worker = self.send_task(
                    scheduler_type,
                    task_kwargs,
                    log_err,
                    log_out,
                    pbs_worker_name,
                    resources,
                )
                if task_status == "done":
                    break
                if killed_worker:
                    cpu, ram, ram_gb, walltime = increase_resources(
                        resources["cpu"],
                        resources["ram"],
                        resources["walltime"],
                    )
                    resources["cpu"] = cpu
                    resources["ram"] = ram
                    resources["walltime"] = walltime

                nb_try += 1

        if task_kwargs and task_name:
            self.save_task_status(task_name, task_status)
            return task_status

    def trace_graph(self, *args, **kwargs) -> None:
        """purposely empty"""

    def set_working_dir_parameter(
        self, t_kwargs: dict, worker_working_dir: str
    ) -> dict:
        """set working directory in tasks arguments"""
        new_t_kwargs = t_kwargs.copy()
        working_dir_names = [
            "working_directory",
            "pathWd",
            "workingDirectory",
            "working_dir",
            "path_wd",
            "WORKING_DIR",
        ]
        for working_dir_name in working_dir_names:
            if working_dir_name in new_t_kwargs:
                new_t_kwargs[working_dir_name] = worker_working_dir
        return new_t_kwargs

    def add_task_to_il_processing_graph(
        self,
        task: il_task,
        task_group: str,
        task_sub_group: str | None = None,
        task_dep_dico=None,
    ) -> None:
        """function used to add a step's task to a processing graph

        task :
            task to launch
        task_group:
            group of tasks the task belong to
        task_sub_group:
            sub group of tasks the task belong to
        task_dep_dico:
            task's dependencies
        """
        if task_dep_dico and not isinstance(task_dep_dico, dict):
            raise ValueError("task_dep_dico parameter must be a dictionary")
        is_addable = task.is_addable_to_graph(self.container_name)
        if not is_addable:
            raise ValueError(
                f"task called : {task.task_name} can't be add to the execution graph"
                ", already in execution graph"
            )
        new_task = None
        if self.container_name not in self.tasks_graph:
            self.tasks_graph[self.container_name] = {}
            self.tasks_graph_figure[self.container_name] = {}

        if task_group not in self.tasks_graph[self.container_name]:
            self.tasks_graph[self.container_name][task_group] = {}
            self.tasks_graph_figure[self.container_name][task_group] = {}

        if task_group == "first_task":
            if self.tasks_graph[self.container_name]["first_task"]:
                raise ValueError("first task already exists")
            new_task = dask.delayed(self.task_launcher)(
                task_name=task.task_name,
                log_err=task.log_err,
                log_out=task.log_out,
                scheduler_type=task.execution_mode,
                pbs_worker_name=task.task_name,
                resources=task.resources,
                task_kwargs=task.parameters,
            )
            new_task_figure = dask.delayed(
                change_name(task.task_name)(self.trace_graph)
            )()
            self.tasks_graph[self.container_name]["first_task"] = new_task
            self.tasks_graph_figure[self.container_name]["first_task"] = new_task_figure
        else:
            # second step case, then the dependency is the "first_task"
            dep_granularity_names = list(task_dep_dico.keys())
            if (
                len(dep_granularity_names) == 1
                and task_dep_dico[dep_granularity_names[0]] == []
            ):
                if dep_granularity_names[0] in self.tasks_graph[self.container_name]:
                    new_task = dask.delayed(self.task_launcher)(
                        self.tasks_graph[self.container_name][dep_granularity_names[0]],
                        task_name=task.task_name,
                        log_err=task.log_err,
                        log_out=task.log_out,
                        scheduler_type=task.execution_mode,
                        pbs_worker_name=task.task_name,
                        resources=task.resources,
                        task_kwargs=task.parameters,
                    )
                    new_task_figure = dask.delayed(
                        change_name(task.task_name)(self.trace_graph)
                    )(
                        self.tasks_graph_figure[self.container_name][
                            dep_granularity_names[0]
                        ]
                    )
                else:
                    new_task = dask.delayed(self.task_launcher)(
                        task_name=task.task_name,
                        log_err=task.log_err,
                        log_out=task.log_out,
                        scheduler_type=task.execution_mode,
                        pbs_worker_name=task.task_name,
                        resources=task.resources,
                        task_kwargs=task.parameters,
                    )
                    new_task_figure = dask.delayed(
                        change_name(task.task_name)(self.trace_graph)
                    )()

            elif len(step_container) != 1:
                dep_list = []
                dep_list_figure = []
                for (
                    dep_granularity_name,
                    dep_tasks_names,
                ) in task_dep_dico.items():
                    for dep_task in dep_tasks_names:
                        if (
                            dep_task
                            not in self.tasks_graph[self.container_name][
                                dep_granularity_name
                            ]
                        ):
                            raise ValueError(
                                f"[{task.task_name}] dependency {dep_task} "
                                f"does not exists in dependency graph"
                            )
                        dep_list.append(
                            self.tasks_graph[self.container_name][dep_granularity_name][
                                dep_task
                            ]
                        )
                        dep_list_figure.append(
                            self.tasks_graph_figure[self.container_name][
                                dep_granularity_name
                            ][dep_task]
                        )
                new_task = dask.delayed(self.task_launcher)(
                    *dep_list,
                    task_name=task.task_name,
                    log_err=task.log_err,
                    log_out=task.log_out,
                    scheduler_type=task.execution_mode,
                    pbs_worker_name=task.task_name,
                    resources=task.resources,
                    task_kwargs=task.parameters,
                )
                new_task_figure = dask.delayed(
                    change_name(task.task_name)(self.trace_graph)
                )(*dep_list_figure)
            else:
                new_task = dask.delayed(self.task_launcher)(
                    task_name=task.task_name,
                    log_err=task.log_err,
                    log_out=task.log_out,
                    scheduler_type=task.execution_mode,
                    pbs_worker_name=task.task_name,
                    resources=task.resources,
                    task_kwargs=task.parameters,
                )
                new_task_figure = dask.delayed(
                    change_name(task.task_name)(self.trace_graph)
                )()
            self.tasks_graph[self.container_name][task_group][task_sub_group] = new_task
            self.tasks_graph_figure[self.container_name][task_group][task_sub_group] = (
                new_task_figure
            )

        self.step_tasks.append(new_task)
        self.step_tasks_figure.append(new_task_figure)

    def build_step_name(self) -> str:
        """
        strategy to build step name
        the name define logging ouput files and resources access
        """
        return self.__class__.__name__

    def __str__(self):
        return self.step_name

    @classmethod
    def step_description(cls) -> str:
        """short step description, will be printed at launch"""
        return "short step description"


class Singleton(type):
    """metaclass dedicated to build a class as a singleton

    from https://stackoverflow.com/questions/43619748/destroying-a-singleton-object-in-python
    """

    _instances = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class singleton_list(list, metaclass=Singleton):
    """singleton list"""

    pass


# step_container is a list where **Step instances**
# are to be appended after instanciation
# it is only used twice:
# 1. in `get_exec_graph`
#   when building execution graph to get the dependencies of last step
# 2. in `add_task_to_il_processing_graph`
#   to check is the step is the first one or not
step_container = singleton_list()


class StepContainer:
    """
    this class is dedicated to contains Step
    """

    def __init__(self, name: str = "implicit_bathy_tasks"):
        # container attribute is a list where the builder will append
        # partials of Step constructors
        # i.e. `container.append(partial(Step.__init__))`
        self.container = []
        self.name = f"{name}_{uuid.uuid1()}"
        self.prelaunch_function = None

    def append(self, step, step_group=""):
        """."""
        if step not in self.container:
            self.container.append(step)
            step.step_group = step_group
            step.func.set_container_name(step.func, self.name)

        else:
            raise Exception(f"step '{step.step_name}' already present in container")
        # link steps
        if len(self.container) > 1:
            self.container[len(self.container) - 2].next_step = step
            step.previous_step = self.container[len(self.container) - 2]

    def __contains__(self, step_ask):
        """
        The __contains__ method is based on step's name
        """
        return any(step.step_name == step_ask.step_name for step in self.container)

    def __setitem__(self, index, val):
        self.container[index] = val

    def __getitem__(self, index):
        return self.container[index]

    def __str__(self):
        return f"[{', '.join(step.step_name for step in self.container)}]"

    def __len__(self):
        return len(self.container)


class change_name:
    """
    decorator to temporary change fonction's name
    useful to plot dask graph
    """

    def __init__(self, new_name: str):
        self.new_name = new_name

    def __call__(self, f):
        if "__name__" in dir(f):
            f.__name__ = self.new_name
        else:
            f.__func__.__name__ = self.new_name

        @wraps(f)
        def wrapped_f(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapped_f


if __name__ == "__main__":
    pass
