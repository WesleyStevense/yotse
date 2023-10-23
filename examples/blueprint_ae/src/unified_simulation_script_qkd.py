import numpy as np
from argparse import ArgumentParser
import pandas
import time
import warnings
from netsquid_netconf.netconf import get_included_params
import collections.abc as col
import pandas as pd
from qlink_interface import MeasurementBasis, ReqMeasureDirectly, ResMeasureDirectly
from nlblueprint.control_layer.egp_datacollector import EGPDataCollector
from nlblueprint.entanglement_tracker.entanglement_tracker_service import LinkStatus, EntanglementTrackerService
from nlblueprint.control_layer.EGP import EGPService
from simulations.unified_simulation_script_state import setup_networks, find_varied_param, save_data,\
    magic_and_protocols, run_simulation
import netsquid as ns
from netsquid_simulationtools.repchain_dataframe_holder import RepchainDataFrameHolder
from netsquid_simulationtools.repchain_data_process import process_repchain_dataframe_holder, process_data_duration, \
    process_data_bb84
from netsquid_simulationtools.repchain_data_plot import plot_qkd_data
from simulations.plot_ae_qkd import plot_ae_qkd
import logging

distance_delft_eindhoven = 226.5

"""
This script takes as input a configuration file and a parameter file. You can find examples of these for each of the
platforms we simulate in the same folder. It parses these files using netconf and runs a BB84 experiment using a
link layer protocol. It then stores this data in a RepChainDataFrameHolder object and saves it in a pickle file.
The script is capable of detecting if a parameter is being varied in a configuration file and, using the netconf
snippet, run the simulation for each of the values of this parameter.

Through optional input arguments one can define the path where the results should be saved, how many runs per data
point to simulate, the name of the saved file and whether the simulation results should be plotted.
"""


def collect_qkd_data(generator, sim_params, n_runs, ae, varied_param, varied_object, number_nodes,
                     suppress_output=False):
    """
    Runs simulation and collects data for each network configuration. Stores data in RepchainDataFrameHolder in format
    that allows for plotting at later point.

    Parameters
    ----------
    generator : generator
        Generator of network configurations.
    sim_params : dict or None
        Dictionary holding simulation parameters to be passed to final
        :obj:`netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`
    n_runs : int
        Number of runs per data point.
    ae : bool
        True if simulating atomic ensemble hardware. Will result in usage of
        `netsquid.qubits.qformalism.QFormalism.SPARSEDM` for the simulation.
    varied_param : str or None
        Name of parameter being varied.
    varied_object : str or None
        Name of object whose parameter is being varied.
    number_nodes : int
        Number of nodes in chain being simulated.
    suppress_output : bool
        If true, status print statements are suppressed.

    Returns
    -------
    meas_holder : :class:`netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`
        RepchainDataFrameHolder with collected simulation data.

    """
    # Ignoring annoying future warning from pd.concat #TODO: at some point this should probably be properly fixed
    warnings.filterwarnings('ignore', category=FutureWarning)
    if ae:
        ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.SPARSEDM)
    else:
        ns.set_qstate_formalism(ns.qubits.qformalism.QFormalism.KET)

    if not suppress_output:
        if varied_param is None:
            print("Simulating", n_runs, "runs")
        else:
            print("Simulating", n_runs, "runs for each value of", varied_param)

    data = []
    measurement_bases = [MeasurementBasis.X, MeasurementBasis.Z]
    for objects, config in generator:
        data_one_configuration = []
        network = objects["network"]
        egp_services = magic_and_protocols(network, config)
        start_time_simulation = time.time()
        for basis in measurement_bases:
            if basis == MeasurementBasis.X:
                exp_spec_args = {"y_rotation_angle_local": np.pi / 2, "y_rotation_angle_remote": np.pi / 2}
            else:
                exp_spec_args = {"y_rotation_angle_local": 0, "y_rotation_angle_remote": 0}
            full_data_one_m_basis = pandas.DataFrame()
            #while len(full_data_one_m_basis.index) < n_runs:
                #print(f'submitting again for {n_runs - len(full_data_one_m_basis.index)} at time {ns.sim_time()}')
                # atomic ensemble measurements don't always succeed, therefore keep sending requests until initial
                # request is satisfied
            for node in network.nodes.values():
                assert len(node.driver[EntanglementTrackerService].get_links_by_status(LinkStatus.AVAILABLE)) == 0
            data_one_meas_basis = run_simulation(egp_services=egp_services,request_type=ReqMeasureDirectly,
                                                 response_type=ResMeasureDirectly,data_collector=EGPDataCollector,
                                                 n_runs=n_runs - len(full_data_one_m_basis.index),
                                                 experiment_specific_arguments=exp_spec_args)
            full_data_one_m_basis = pd.concat([full_data_one_m_basis, pd.DataFrame(data_one_meas_basis)],
                                              ignore_index=True)
                #reset all nodes, the entanglement tracker and the connections to prevent a rare error where
                #swap asap gets a request when there already is active entanglement
                # node_list = [node for node in network.nodes.values()]
                # node_list[0].driver[EGPService].await_timer(2*1250000)
                # for node in network.nodes.values():
                #     node.reset()
                #     ent_tracker = node.driver[EntanglementTrackerService]
                #     link_ids = ent_tracker.get_links_by_status(LinkStatus.AVAILABLE)
                #     for link_id in link_ids:
                #         ent_info = ent_tracker.get_entanglement_info_of_link(link_id)
                #         ent_tracker.untrack_entanglement_info(ent_info)
                # for connection in network.connections.values():
                #     connection.reset()

            data_one_meas_basis = full_data_one_m_basis
            if varied_param is not None:
                param_value = config["components"][varied_object]["properties"][varied_param]
                data_one_meas_basis[varied_param] = param_value
            data_one_configuration.append(data_one_meas_basis)
        simulation_time = time.time() - start_time_simulation
        if not suppress_output:
            if varied_param is None:
                print(f"Performed {n_runs} runs in X and Z {simulation_time:.2e} s")
            else:
                print(
                    f"Performed {n_runs} runs in X and Z for {varied_param} = {param_value} in {simulation_time:.2e} s")
        data.append(data_one_configuration)

    flat_data = [item for sublist in data for item in sublist]
    data = pd.concat(flat_data, ignore_index=True)
    new_data = data.drop(labels=["time_stamp", "entity_name"], axis="columns")

    if sim_params is not None:
        baseline_parameters = sim_params
        baseline_parameters["number_nodes"] = number_nodes
        # make all lists (required by safe netconf loader) into tuples (required by RDFHolder)
        for key in baseline_parameters.keys():
            if isinstance(baseline_parameters[key], list):
                baseline_parameters[key] = tuple(baseline_parameters[key])
            # double check all params are hashable now
            assert isinstance(baseline_parameters[key], col.Hashable)
    else:
        baseline_parameters = {"length": distance_delft_eindhoven,
                               "number_nodes": number_nodes}
    meas_holder = RepchainDataFrameHolder(baseline_parameters=baseline_parameters, data=new_data, number_of_nodes=2)

    return meas_holder


def run_unified_simulation_qkd(config_file_name, n_runs=10, suppress_output=False):
    """Run unified simulation script.

    Parameters
    ----------
    config_file_name : str
        Name of configuration file.
    paramfile : str
        Name of file holding simulation parameters to be passed to final
        :obj:`netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`
    n_runs : int
        Number of runs per data point.
    suppress_output : bool
        If true, status print statements are suppressed.

    Returns
    -------
    meas_holder : :class:`netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`
        RepchainDataFrameHolder with collected QKD data.
    varied_param : str
        Name of parameter that is being varied in the simulation.

    """
    #ns.logger.setLevel(logging.DEBUG)
    # Create a file handler
    handler = logging.FileHandler('my_log_file.log')  # Log output goes to this file

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    ns.logger.addHandler(handler)
    generator, ae, number_nodes = setup_networks(config_file_name)
    varied_object, varied_param = find_varied_param(generator)
    generator, ae, _ = setup_networks(config_file_name)

    sim_params = get_included_params(config_file=config_file_name)
    repchain_df_holder = collect_qkd_data(generator, sim_params, n_runs, ae, varied_param, varied_object,
                                          number_nodes, suppress_output=suppress_output)

    return repchain_df_holder, varied_param


def plot_data(meas_holder, skr_minmax=True):
    """
    Plots QKD data using the simulation tools snippet.

    Parameters
    ----------
    meas_holder : :class:`netsquid_simulationtools.repchain_dataframe_holder.RepchainDataFrameHolder`
        RepchainDataFrameHolder with collected QKD data.
    skr_minmax:
        Whether the SKR plot should show regular error bars or min/max.
    """
    if len(meas_holder.varied_parameters) != 1:
        raise ValueError("Can only plot for data with exactly one varied parameter. "
                         f"This data has the following varied parameters: {meas_holder.varied_parameters}")
    [varied_param] = meas_holder.varied_parameters
    processed_data = process_repchain_dataframe_holder(repchain_dataframe_holder=meas_holder,
                                                       processing_functions=[process_data_duration,
                                                                             process_data_bb84])
    processed_data.to_csv("output.csv", index=False)
    plot_qkd_data(filename="output.csv", scan_param_name=varied_param,
                  scan_param_label=varied_param, shaded=False, skr_minmax=skr_minmax)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('configfile', type=str, help="Name of the config file.")
    parser.add_argument('-n', '--n_runs', required=False, type=int, default=10,
                        help="Number of runs per configuration. If none is provided, defaults to 10.")
    parser.add_argument('--output_path', required=False, type=str, default="raw_data",
                        help="Path relative to local directory where simulation results should be saved.")
    parser.add_argument('--filebasename', required=False, type=str, help="Name of the file to store results in.")
    parser.add_argument('--plot', dest="plot", action="store_true", help="Plot the simulation results.")

    args, unknown = parser.parse_known_args()

    repchain_df_holder, varied_param = run_unified_simulation_qkd(config_file_name=args.configfile,
                                                                  n_runs=args.n_runs,
                                                                  suppress_output=False)
    # Note: for single run the parameters that are saved with the simulation results are automatically retrieved from
    # the (first) parameter file included in the 'configfile'
    save_data(repchain_df_holder, args)
    if args.plot:
        plot_ae_qkd(repchain_df_holder, args.filebasename)