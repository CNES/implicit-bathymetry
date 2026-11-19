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
"""ImplicitBathyOrchestrator.py allow user to deploy implicit bathymetry on HPC architecture

check 'ImplicitBathyOrchestrator.py -h' for options
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from subprocess import PIPE, Popen
from dask.distributed import Client, LocalCluster

from implicit_bathy.orchestration.common.file_utils import (
    ensure_dir,
    get_implicit_bathy_project_dir,
)
from implicit_bathy.orchestration.configuration_files import (
    read_config_file as rcf,
)
from implicit_bathy.orchestration.steps import il_job_helpers

LOGGER = logging.getLogger("distributed.worker")


def run(
    config_path: str,
    config_ressources: str | None,
    csv_file: str,
    scheduler_type: str,
    start: int,
    end: int,
    restart: bool,
    restart_from: str,
    graph_figures: list[str],
    only_summary: bool,
    nb_parallel_tasks: int,
    tmp_directory: str | None,
) -> int:
    """run implicit bathy
    Parameters
    ----------

    config_path:
        path to the configuration file which define the run_informations
    config_ressources:
        path to a implicit bathy configuration file dedicated to resources by steps
    csv_file:
        path to csv_file with differents parameters
    scheduler_type:
        define on which architecture launch tasks
    start:
        first step index
    end:
        last step index
    restart:
        restart the chain from the last state
    restart_from:
        restart the chain from the the state file
    graph_figures:
        paths to draw executions graphs
    only_summary:
        only print steps summarize, no treatement are triggered
    nb_parallel_tasks:
        number of tasks in parallel (nb dask workers)
    tmp_directory:
        name of the environment variable containing a path to a directory
         where tmp files will be written
    """
    # #########################################################################
    if restart and restart_from:
        raise ValueError("parameters 'restart' and 'restart_from' are not compatible")
    cfg = rcf.ReadConfigFile(config_path)

    chain_to_process = cfg.get_builders()[0](
        config_path,
        config_ressources,
        scheduler_type,
        restart,
        restart_from,
        tmp_directory,
    )
    if start == end == 0:
        all_steps = chain_to_process.get_steps_number()
        start = all_steps[0]
        end = all_steps[-1]

    first_step_index = start - 1
    last_step_index = end - 1

    final_graphs = chain_to_process.get_final_il_exec_graph(
        first_step_index, last_step_index, graph_figures
    )

    step_summarize = chain_to_process.print_step_summarize(
        start, end, config_ressources is not None
    )

    if not only_summary:
        chain_to_process.generate_output_directories(first_step_index, restart)

    print(step_summarize)

    if not only_summary:
        nb_workers = nb_parallel_tasks
        if nb_workers < 1:
            nb_workers = 1
        if scheduler_type == "debug" and nb_parallel_tasks != 1:
            LOGGER.warning(
                "'debug' mode activated with a number "
                "of parallel tasks different from 1."
            )
        cluster = LocalCluster(
            n_workers=nb_workers,
            threads_per_worker=1,
            memory_limit="10GB",
            processes=False,
            silence_logs="error",
        )
        client = Client(cluster)
        dashboard_information = f"dashboard available at : {client.dashboard_link}\n"
        print(dashboard_information)

        chain_to_process.preliminary_informations(
            f"{step_summarize}\n{dashboard_information}"
        )
        # final_graph.compute()

        failed_flag = False
        for graph in final_graphs:
            if failed_flag:
                chain_to_process.update_graph_status(graph.figure_graph)
                continue
            delayed_graph = graph.build_graph()

            res = client.submit(delayed_graph.compute)

            il_status = res.status
            while il_status == "pending":
                time.sleep(2)
                il_status = res.status
                if il_status == "error":
                    print("Some task failed, please check logs")
                    failed_flag = True
                    break
                if il_status == "finished":
                    break

        chain_to_process.tasks_summary()
        client.close()
    return 0


def get_ram(ram: str) -> float:
    """
    usage return ram in gb
    ram [param] [str]

    out [ram] [str] : ram in gb
    """

    ram = ram.lower().replace(" ", "")
    if "gb" in ram:
        ram_f = float(ram.split("gb")[0])
    elif "mb" in ram:
        ram_f = float(ram.split("mb")[0]) / 1024
    return ram_f


def submit_job_cmd(
    cfg_file: str,
    config_ressources: str | None,
    csv_file: str,
    starting_step: int,
    ending_step: int,
    restart: bool,
    restart_from: str | None,
    nb_parallel_tasks: int,
    scheduler_type: str,
    execution_graph_files: list[str],
    tmp_directory: str,
    scheduler_name: str,
):
    """Build command to launch implicit bathy on HPC."""
    cfg = rcf.ReadConfigFile(cfg_file)
    slurm_section = cfg.config.slurm
    log_dir = Path(cfg.config.dataprocessing.output_path) / "logs"
    job_dir = log_dir
    ensure_dir(job_dir)

    job_helper = il_job_helpers.JobHelper(
        config_ressources, "implicit_bathy_chain", csv_file, slurm_section
    )

    implicit_bathy_main = (
        Path(job_dir)
        / f"implicit_bathy.{job_helper.inventory_jobfile_helper[scheduler_name.lower()].extension}"
    )
    log_err = Path(log_dir) / "implicit_bathy_err.log"
    log_out = Path(log_dir) / "implicit_bathy_out.log"

    if Path(implicit_bathy_main).exists():
        Path(implicit_bathy_main).unlink()

    cpu_from_cfg = job_helper.resources["cpu"]
    nb_cpu = max(nb_parallel_tasks, cpu_from_cfg)
    job_helper.force_resource("cpu", nb_cpu)
    scripts = Path(get_implicit_bathy_project_dir(), "orchestration")
    exe = (
        f"python {scripts}/ImplicitBathy.py "
        f"-config {cfg_file} "
        f"-starting_step {starting_step} "
        f"-ending_step {ending_step} "
        f"-nb_parallel_tasks {nb_parallel_tasks} "
        f"-scheduler_type {scheduler_type} "
        f"-tmp_directory_env {tmp_directory} "
        f"-csv_file {csv_file}"
    )

    if restart_from is not None:
        exe = f"{exe} -restart_from {restart_from}"
    if restart:
        exe = f"{exe} -restart"
    if config_ressources:
        exe = f"{exe} -config_ressources {config_ressources}"
    if execution_graph_files:
        exe = f"{exe} -execution_graph_files {' '.join(execution_graph_files)}"
    job_helper.write(
        scheduler_name.lower(),
        "ImplicitBathy",
        implicit_bathy_main,
        log_out,
        log_err,
        force_executable=exe,
    )
    submit_cmd = (
        f"{job_helper.inventory_jobfile_helper[scheduler_name.lower()].submit_cmd} "
        f"{implicit_bathy_main}"
    )
    return submit_cmd


def run_cluster(parsed_args: argparse.ArgumentParser):
    """launch implicit bathy on HPC"""
    if parsed_args.restart and parsed_args.restart_from:
        raise ValueError("parameters 'restart' and 'restart_from' are not compatible")
    cfg = rcf.ReadConfigFile(parsed_args.config_path)
    # if len(chains) > 1:
    #     chain_to_process = workflow_merger(
    #         chains,
    #         parsed_args.config_path,
    #         parsed_args.config_ressources,
    #         parsed_args.scheduler_type,
    #         parsed_args.restart,
    #         parsed_args.restart_from,
    #     )
    # else:
    chain_to_process = cfg.get_builders()[0](
        parsed_args.config_path,
        parsed_args.config_ressources,
        parsed_args.scheduler_type,
        parsed_args.csv_file,
        parsed_args.restart,
        parsed_args.restart_from,
    )
    if parsed_args.start == parsed_args.end == 0:
        all_steps = chain_to_process.get_steps_number()
        parsed_args.start = all_steps[0]
        parsed_args.end = all_steps[-1]

    first_step_index = parsed_args.start - 1
    last_step_index = parsed_args.end - 1

    chain_to_process.get_final_il_exec_graph(
        first_step_index, last_step_index, parsed_args.graph_figures
    )

    print(
        chain_to_process.print_step_summarize(
            parsed_args.start,
            parsed_args.end,
            parsed_args.config_ressources is not None,
        )
    )

    if not parsed_args.only_summary:
        submit_cmd = submit_job_cmd(
            parsed_args.config_path,
            parsed_args.config_ressources,
            parsed_args.csv_file,
            parsed_args.start,
            parsed_args.end,
            parsed_args.restart,
            parsed_args.restart_from,
            parsed_args.nb_parallel_tasks,
            parsed_args.scheduler_type,
            parsed_args.graph_figures,
            parsed_args.tmp_directory,
            parsed_args.scheduler_type.lower(),
        )
        Popen(submit_cmd, shell=True, stdout=PIPE, stderr=PIPE)
    return 0


def implicit_bathy_arguments() -> argparse.ArgumentParser:
    """Definition of ImplicitBathy.py and ImplicitBathyOrchestrator.py launching options."""
    parser = argparse.ArgumentParser(
        description="This function allow you to launch implicit bathymetry processing chain"
    )

    parser.add_argument(
        "-config",
        dest="config_path",
        help="path to the configurationfile which rule le run",
        required=True,
    )
    parser.add_argument(
        "-starting_step",
        dest="start",
        help="start chain from 'starting_step'",
        default=0,
        type=int,
        required=False,
    )
    parser.add_argument(
        "-ending_step",
        dest="end",
        help="run chain until 'ending_step'-1 mean 'to the end'",
        default=0,
        type=int,
        required=False,
    )
    parser.add_argument(
        "-config_ressources",
        dest="config_ressources",
        help="path to Implicit Bathymetry ressources configuration file",
        required=False,
    )
    parser.add_argument(
        "-csv_file",
        dest="csv_file",
        help="path to csv file configuration file",
        required=False,
        default="None",
    )
    parser.add_argument(
        "-execution_graph_files",
        dest="graph_figures",
        help="output figure for execution graphs",
        default="",
        nargs="+",
        required=False,
    )
    parser.add_argument(
        "-restart_from",
        dest="restart_from",
        help="file containing task's states generated by implicit bathymetry",
        default=None,
        required=False,
    )
    parser.add_argument(
        "-restart",
        dest="restart",
        help="flag to restart implicit bathymetry from a previous run",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-only_summary",
        dest="only_summary",
        help=(
            "if set, only the summary will be printed. The chain will not be launched"
        ),
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-nb_parallel_tasks",
        dest="nb_parallel_tasks",
        help=(
            "define the number of worker launching tasks. "
            "One worker consumes one thread"
        ),
        default=1,
        type=int,
        required=False,
    )
    parser.add_argument(
        "-scheduler_type",
        default="localCluster",
        dest="scheduler_type",
        const="local",
        nargs="?",
        choices=["debug", "cluster", "PBS", "Slurm", "localCluster"],
        help="list of available scheduler in implicit bathymetry",
    )
    parser.add_argument(
        "-tmp_directory_env",
        default=None,
        dest="tmp_directory",
        help=(
            "name of the environment variable "
            "containing a path to a directory "
            "where tmp files will be written"
        ),
    )
    return parser


def main():
    """Function to create conda entry point"""
    parser = implicit_bathy_arguments()

    # if we use a cluster, we must have a working directory
    parser.set_defaults(tmp_directory="TMPDIR")

    args = parser.parse_args()
    cluster_schedulers = "Slurm"
    if args.scheduler_type in cluster_schedulers:
        run_cluster(args)
    else:
        print(
            f"'ImplicitBathyOrchestrator.py -scheduler_type' must be in {cluster_schedulers} "
        )


if __name__ == "__main__":
    sys.exit(main())
