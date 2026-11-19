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
"""Implicit Bathymetry main"""

import argparse
import logging
import sys
import time

import dask
from dask.distributed import Client, LocalCluster

from implicit_bathy.orchestration.configuration_files import (
    read_config_file as rcf,
)

dask.config.set({"distributed.comm.timeouts.connect": "300000s"})
# dask.config.set({"worker-memory-pause": False})
dask.config.set({"distributed.worker.memory.pause": False})
LOGGER = logging.getLogger("distributed.worker")


def run(
    config_path: str,
    config_ressources: str | None,
    scheduler_type: str,
    start: int,
    end: int,
    csv_file: str,
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
        csv_file,
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
        LOGGER.info(dashboard_information)

        chain_to_process.preliminary_informations(
            f"{step_summarize}\n{dashboard_information}"
        )
        # final_graphs[0].compute()

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
                    LOGGER.info("Some task failed, please check logs")
                    failed_flag = True
                    break
                if il_status == "finished":
                    break
        chain_to_process.tasks_summary()
        client.close()
    return 0


def implicit_bathy_arguments() -> argparse.ArgumentParser:
    """Definition of ImplicitBathy.py and ImplicitBathyOrchestrator.py launching options."""
    parser = argparse.ArgumentParser(
        description="This function allow you to launch implicit_bathy processing chain"
    )

    parser.add_argument(
        "-config",
        dest="config_path",
        help="path to the configurationfile which rule le run",
        required=True,
    )
    parser.add_argument(
        "-csv_file",
        dest="csv_file",
        help="path to csv file configuration file",
        required=False,
        default="None",
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
        help="path to Implicit Bathy ressources configuration file",
        required=False,
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
    """Use to create conda entry points."""
    args = implicit_bathy_arguments().parse_args()
    run(
        args.config_path,
        args.config_ressources,
        args.scheduler_type,
        args.start,
        args.end,
        args.csv_file,
        args.restart,
        args.restart_from,
        args.graph_figures,
        args.only_summary,
        args.nb_parallel_tasks,
        args.tmp_directory,
    )


if __name__ == "__main__":
    sys.exit(main())
