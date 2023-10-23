import os
import yaml
import shutil
from datetime import datetime
from ruamel.yaml import YAML
from ruamel.yaml.nodes import ScalarNode

from yotse.pre import Experiment


def setup_optimization_dir(experiment: Experiment, step_number: int, job_number: int) -> None:
    """Create the directory structure for an optimization step.

    Parameters:
    ----------
    experiment : Experiment
        The Experiment object for which the directory structure should be set up/
    step_number : int
        The number of the current optimization step.
    job_number: int
        The number of the job within the optimization step.

    Note:
    -----
    This function creates the directory structure for an optimization step within an experiment. The structure
    includes a `src` directory containing several files related to the optimization, and an `output` directory
    containing directories for each step and job. The function does not return anything, but modifies the file system
    to create the necessary directories.

    The directory structure for the optimization step is as follows (for m optimization steps and n jobs):
    > src
        - unified_script.py
        - processing_function.py
        - config.yaml
        - baseline_params.yaml
        - qiapt_runscript.py
    > output
        > experiment_name_timestamp_str
            > step0
                > job0
                    - stdout0.txt
                    - dataframe_holder.pickle (?)
                    - baseline_params_job0.yaml
                    - config_job0.yaml
                ...
                > jobn
            ...
            > stepm
    """
    output_directory = os.path.join(experiment.system_setup.source_directory, '..',
                                    experiment.system_setup.output_dir_name)
    output_directory = os.path.realpath(output_directory)                                       # clean path of '..'
    if step_number == 0 and job_number == 0:
        # for first step create timestamped project directory
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        name_project_dir = os.path.join(output_directory, experiment.name + "_" + timestamp_str)
        new_working_dir = os.path.join(name_project_dir, f'step{step_number}', f'job{job_number}')
    else:
        if not os.path.basename(os.path.normpath(experiment.system_setup.working_directory)).startswith("job"):
            raise RuntimeError("The current working directory does not start with 'job'. "
                               "New working directory can't be set up properly.")
        new_working_dir = os.path.join(experiment.system_setup.working_directory, '..', '..',
                                       f'step{step_number}', f'job{job_number}')
        new_working_dir = os.path.realpath(new_working_dir)                                     # clean path of '..'

    if not os.path.exists(new_working_dir):
        os.makedirs(new_working_dir)
    experiment.system_setup.working_directory = new_working_dir


def update_yaml_params(param_list: list, paramfile_name: str) -> None:
    """Update parameter values in a YAML file and save the updated file.

    Parameters:
    ----------
    param_list : List[Tuple[str, Any]]
        A list of tuples containing parameter names and their updated values.
    paramfile_name : str
        The name of the YAML file containing the parameters to update.
    """

    # Load the YAML file
    with open(paramfile_name, 'r') as f:
        params = yaml.safe_load(f)

    # Update each parameter value
    for param_name, param_value in param_list:
        if param_name in params:
            params[param_name] = param_value
        else:
            raise ValueError(f"Parameter name '{param_name}' not found in YAML file")

    # Save the updated YAML file
    with open(paramfile_name, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

def update_yaml_mpn(configfile_name: str, p_surv: float, protocol: str='double_click', setup: str='nieuwegein'):
    """Function that updates the config yaml file of an AE simulation with a single repeater chain for a non pefect PPS 
    where the heuristic that the probabilities for a photon to arrive at the heralding station are equal.
    Parameters:
    ----------
    configfile_name : str
        The name of the YAML configuration file to modify.
    p_surv : floeat
        Value for the probability for a photon to arrive at heralding station (this is a parameter being sweeped over)
    protocol: str
        String indicating whether using single or double click protocol
    setup: str
        String indicating whether the repaater is placed in Utrecht or Nieuwegien
    """

    def calculate_mpns(p, L_1_l, L_1_r, L_2_l, L_2_r, alpha):
        """
        Function that computes the desired mean photon numbers in a single repeater chain AE protocol based on the heuristic
        that the probabilities for a single photon to arrive at the heralding station are all the same.

        Parameters
        ----------
        p: desired probability for a photon to reach heralding station
        L_1_l: distance from the left end node to the left heralding station
        L_1_r: distance from the left heralding station to the repeater node
        L_2_l: distance from the repeater node to the right heralding station
        L_2_r: distance from the right heralding station to the right end node
        alpha: attenuation of the channels

        Returns
        -------
        mpn_end_node_1: mean photon number of the left end node
        mpn_1: mean photon number of the left source of the repeater node
        mpn_2: mean photon number of the right source of the repeater node
        mpn_end_node_1: mean photon number of the right end node
        """
        transmittance = lambda L: np.power(10, -(alpha*L)/10)
        mpn_end_node_1 = p/transmittance(L_1_l)
        mpn_1 = p/transmittance(L_1_r)
        mpn_2 = p/transmittance(L_2_l)
        mpn_end_node_2 = p / transmittance(L_2_r)
        return mpn_end_node_1, mpn_1, mpn_2, mpn_end_node_2

    # set values of network based on where repeater is
    if setup == 'nieuwegein':
        L_1_l, L_1_r, L_2_l, L_2_r = 83, 19, 75, 50
    elif setup == 'utrecht':
        L_1_l, L_1_r, L_2_l, L_2_r = 15, 68, 19, 125
    else:
        raise ValueError(f"setup should be 'utrecht' or 'nieuwegein', not '{setup}'.")
    mpn_end_node_1, mpn_1, mpn_2, mpn_end_node_2 = calculate_mpns(p=p_surv, L_1_l=L_1_l, L_1_r=L_1_r, L_2_l=L_2_l,
                                                                      L_2_r=L_2_r, alpha=0.2)
    if protocol == 'double_click':
        with open(configfile_name, 'r') as yaml_file:
            lines = yaml_file.readlines()
        #use the line numbers of the config file to update mpn's. Change this if using other config file.
        lines[16] = f'      mean_photon_number: {mpn_end_node_1:.6f}\n'
        lines[28] = f'      mean_photon_number_1: {mpn_1:.6f}\n'
        lines[29] = f'      mean_photon_number_2: {mpn_2:.6f}\n'
        lines[39] = f'      mean_photon_number: {mpn_end_node_2:.6f}\n'
        with open(configfile_name, 'w') as output_yaml_file:
            output_yaml_file.writelines(lines)
    elif protocol == 'single_click': 
        raise NotImplementedError()
    else: 
        raise ValueError("Protocol should be 'single_click' or 'double_click' not {protocol}.")


def represent_scalar_node(dumper: yaml.Dumper, data: yaml.ScalarNode) -> str:
    """Represent a ScalarNode object as a scalar value in a YAML file.

    Parameters:
    ----------
    dumper : yaml.Dumper
        The YAML dumper object being used to write the file.
    data : yaml.ScalarNode
        The ScalarNode object being represented.

    Returns:
    -------
    scalar : str
        The scalar value of the ScalarNode object.
    """
    return dumper.represent_scalar(data.tag, data.value)


def replace_include_param_file(configfile_name: str, paramfile_name: str) -> None:
    """Replace the INCLUDE keyword in a YAML config file with a reference to a parameter file.

    Parameters:
    ----------
    configfile_name : str
        The name of the YAML configuration file to modify.
    paramfile_name : str
        The name of the parameter file to include in the configuration file.

    Note:
    -----
    This function replaces an INCLUDE keyword in a YAML configuration file with a reference to a parameter file.
    It loads the YAML config file, searches recursively for an INCLUDE keyword, and replaces it with a reference
    to the specified parameter file. If the INCLUDE keyword is not found, an error is raised.
    """
    yaml = YAML(typ='rt')
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.representer.add_representer(ScalarNode, represent_scalar_node)

    # Load the YAML config file
    with open(configfile_name, 'r') as f:
        old_config = yaml.load(f)

    # Find the line with the INCLUDE keyword recursively
    def replace_include(config: dict, replace_str: str, found: bool = False) -> bool:
        """Recursively search a dictionary or list for an INCLUDE keyword and replace it with a reference to a parameter
         file.

        Parameters:
        ----------
        config : dict or list
            The dictionary or list to search for an INCLUDE keyword.
        replace_str : str
            The name of the parameter file to include in place of the INCLUDE keyword.
        found : bool, optional
            A boolean flag indicating whether an INCLUDE keyword has already been found and replaced. Defaults to False.

        Returns:
        -------
        found : bool
            True if an INCLUDE keyword was found and replaced in the dictionary or list, False otherwise.
        """
        if isinstance(config, dict):
            for key, value in list(config.items()):
                if key != 'INCLUDE':
                    found = replace_include(value, replace_str, found)
                elif key == 'INCLUDE' and not found:
                    config[key] = ScalarNode(tag='!include', value=replace_str, style=None)
                    found = True
                elif key == 'INCLUDE' and found:
                    del config[key]
        elif isinstance(config, list):
            for i in range(len(config)):
                found = replace_include(config[i], replace_str, found)

        return found

    found_include = replace_include(old_config, paramfile_name)

    # Check if the INCLUDE keyword was found
    if not found_include:
        raise ValueError(f"INCLUDE statement not found in '{configfile_name}'")

    # Save the updated YAML config file
    with open(configfile_name, 'w') as f:
        yaml.dump(old_config, f)


def create_separate_files_for_job(experiment: Experiment, datapoint_item: list, step_number: int,
                                  job_number: int, setup: str=None, protocol: str=None) -> list:
    """Create separate parameter and configuration files for a job and prepare for execution.

    Parameters:
    ----------
    experiment : Experiment
        The experiment object containing information about the experiment.
    datapoint_item : list
        A single item of data points for the job, represented as a list.
    step_number : int
        The number of the step in the experiment.
    job_number: int
        The number of the job within the step.
    setup: str
        In case of AE with one repeater on SURF  Delft-Eindhoven network: indicate location of repeaeter
    protocol: str
        In case of AE: indicate whether using single_click or double_click

    Returns:
    -------
    job_cmdline : list
        The command line arguments for running the job.

    Note:
    -----
    This function creates separate parameter and configuration files for a job based on the provided experiment,
    datapoint item, step number, and job number. It prepares the job for execution by setting up the necessary files
    and returning the command line arguments for running the job. The function returns the command line arguments
     as a list for use with QCG-Pilotjob.

    The created files will be saved in the experiment's directory, under a subdirectory for the step and job.
    The parameter file will have a name like "params_stepY_jobX.yaml" and the configuration file will have a name like
     "config_stepY_jobX.yaml", where "X" is the job number and "Y" the step number.
    """
    # this should execute after the directory for the specific job is set up by setup_optimization_dir
    # 1 - copy the original param and config file to the created dir
    source_directory = experiment.system_setup.source_directory
    working_directory = experiment.system_setup.working_directory
    old_cmdline_args = experiment.system_setup.cmdline_arguments.copy()
    paramfile_name = os.path.basename(old_cmdline_args['paramfile'])
    configfile_name = os.path.basename(old_cmdline_args['configfile'])
    # delete unnecessary args from dict copy
    del old_cmdline_args['paramfile']
    del old_cmdline_args['configfile']
    job_name = f'job{job_number}'
    step_name = f'step{step_number}'

    # Copy paramfile_name to working_directory
    paramfile_base_name, paramfile_ext = os.path.splitext(paramfile_name)
    paramfile_new_name = paramfile_base_name + '_' + step_name + '_' + job_name + paramfile_ext
    paramfile_source_path = os.path.join(source_directory, paramfile_name)
    paramfile_dest_path = os.path.join(working_directory, paramfile_new_name)
    shutil.copy(paramfile_source_path, paramfile_dest_path)

    # Copy configfile_name to working_directory
    configfile_base_name, configfile_ext = os.path.splitext(configfile_name)
    configfile_new_name = configfile_base_name + '_' + step_name + '_' + job_name + configfile_ext
    configfile_source_path = os.path.join(source_directory, configfile_name)
    configfile_dest_path = os.path.join(working_directory, configfile_new_name)
    shutil.copy(configfile_source_path, configfile_dest_path)

    # 2 - take the data_point for the current step + the active parameters/cmdlineargs and then overwrite those
    # in the respective param file
    param_list = []
    vary_mpns = False
    for p, param in enumerate(experiment.parameters):
        if param.is_active and param.name != 'p_surv':
            if len(experiment.parameters) == 1:
                # single parameter
                param_list.append((param.name, datapoint_item))
            else:
                param_list.append((param.name, datapoint_item[p]))
        elif param.is_active and param.name == 'p_surv': 
            #handle changing mean photon numbers in AE simulation
            vary_mpns = True
            if len(experiment.parameters) == 1:
                # single parameter
                p_surv = datapoint_item
            else:
                p_surv = datapoint_item[p]

    if len(param_list) != len(datapoint_item) and not vary_mpns:
        raise RuntimeError("Datapoint has different length then list of parameters to be changes in paramfile.")
    if len(param_list) != len(datapoint_item)-1 and vary_mpns:
        raise RuntimeError("Datapoint has different length then list of parameters to be changes in paramfile.")
    if vary_mpns and not setup:
        raise ValueError("For AE simulations where p_surv is varied a setup needs to be specified")
    if vary_mpns and not protocol:
        raise ValueError("For AE simulations where p_surv is varied a protocol needs to be specified")
        
    update_yaml_params(param_list=param_list, paramfile_name=paramfile_dest_path)

    # 3 - overwrite the name of the paramfile inside the configfile with the new paramfile name
    replace_include_param_file(configfile_name=configfile_dest_path, paramfile_name=paramfile_dest_path)
    #4 - overwrite the configfile's mpn's if applicable 
    if vary_mpns: 
        update_yaml_mpn(configfile_name=configfile_dest_path, p_surv=p_surv, setup=setup, protocol=protocol)
    # 5 - construct new cmdline such that it no longer contains the varied params, but instead the correct paths to
    # the new param and config files
    cmdline = [os.path.join(experiment.system_setup.source_directory, experiment.system_setup.program_name)]
    cmdline.append(configfile_dest_path)
    cmdline.append("--paramfile")
    cmdline.append(paramfile_dest_path)
    # add fixed cmdline arguments
    for key, value in old_cmdline_args.items():
        cmdline.append(key)
        cmdline.append(str(value))

    return cmdline
