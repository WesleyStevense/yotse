"""Example script for execution of a wobbly_function.py experiment."""
import os
import shutil
from qiaopt.pre import Experiment, SystemSetup, Parameter, OptimizationInfo
from qiaopt.run import Executor
from qiaopt.optimization import Optimizer, GAOpt


def wobbly_pre():
    wobbly_experiment = Experiment(
        experiment_name="wobbly_example",
        system_setup=SystemSetup(source_directory=os.getcwd(),
                                 program_name='wobbly_function.py',
                                 command_line_arguments={"--filebasename": 'wobbly_example'},
                                 analysis_script="analyse_function_output.py",
                                 executor="python",
                                 files_needed=["*.py"]),
        parameters=[
            Parameter(
                name="x",
                parameter_range=[-4, 4],
                number_points=2,
                distribution="uniform",
                constraints=[],
                weights=None,
                parameter_active=True,
                data_type="continuous"
            ),
            Parameter(
                name="y",
                parameter_range=[-3, 3],
                number_points=10,
                distribution="uniform",
                constraints=[],
                weights=None,
                parameter_active=True,
                data_type="continuous"
            )
        ],
        opt_info_list=[
            OptimizationInfo(
                name="GA",
                opt_parameters={
                    "num_generations": 100,     # number of iterations of the algorithm
                    "num_points": 5,            # number of points to re-create
                    "refinement_x": 0.5,        # in %
                    "refinement_y": 0.5,        # in %
                    "logging_level": 1,
                },
                is_active=True)]
    )
    return wobbly_experiment


def cost_function(x, y):
    return x**2 + y**2


def main():
    num_opt_steps = 10
    #wobbly_example = WobblyExecutor(experiment=wobbly_pre())
    wobbly_example = Executor(experiment=wobbly_pre())
    wobbly_example.cost_function = cost_function
    for i in range(num_opt_steps):
        wobbly_example.run(step=i)

    # remove files and directories
    shutil.rmtree('output')
    dirs = [f for f in os.listdir(os.getcwd()) if (f.startswith(".qcg"))]
    for d in dirs:
        shutil.rmtree(os.path.join(os.getcwd(), d))


if __name__ == "__main__":
    main()
    # add clean up function that removes qcg folders (see test-run.py)
