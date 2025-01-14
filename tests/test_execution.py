import json
import math
import os
import shutil
import unittest
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas
import pytest
from pygad.pygad import GA

from yotse.execution import Executor
from yotse.optimization.algorithms import GAOpt
from yotse.optimization.optimizer import Optimizer
from yotse.pre import ConstraintDict
from yotse.pre import Experiment
from yotse.pre import OptimizationInfo
from yotse.pre import Parameter
from yotse.pre import SystemSetup


if os.getcwd().endswith("tests"):
    DUMMY_FILE = "myfunction.py"
else:
    DUMMY_FILE = "tests/myfunction.py"


def create_default_param(
    name: str = "bright_state_parameter",
    parameter_range: List[Union[float, int]] = [0.1, 0.9],
    number_points: int = 9,
    distribution: str = "linear",
    constraints: Union[ConstraintDict, np.ndarray, None] = None,
    custom_distribution: Optional[Callable[[float, float, int], np.ndarray]] = None,
) -> Parameter:
    return Parameter(
        name=name,
        param_range=parameter_range,
        number_points=number_points,
        distribution=distribution,
        constraints=constraints,
        custom_distribution=custom_distribution,
    )


def create_default_experiment(
    parameters: Optional[List[Parameter]] = None,
    opt_info_list: Optional[List[OptimizationInfo]] = None,
) -> Experiment:
    return Experiment(
        experiment_name="default_exp",
        system_setup=SystemSetup(
            source_directory=os.getcwd(),
            program_name=DUMMY_FILE,
            command_line_arguments={"arg1": 1.0},
        ),
        parameters=parameters,
        opt_info_list=opt_info_list,
    )


def create_default_executor(experiment: Experiment) -> Executor:
    return Executor(experiment=experiment)


class TestExecutor(unittest.TestCase):
    """Test the executor class."""

    def setUp(self) -> None:
        self.path: Optional[str] = None  # path for tearDown
        self.test_points = np.array([[1], [2], [3], [4]])
        # self.tearDown()

    def tearDown(self) -> None:
        for i in range(4):
            if os.path.exists(f"stdout{i}.txt"):
                os.remove(f"stdout{i}.txt")
        if self.path is not None:
            [os.remove(f) for f in os.listdir(self.path) if f.endswith(".csv")]  # type: ignore[func-returns-value]
            shutil.rmtree(os.path.join(self.path, "output"))
            dirs = [f for f in os.listdir(self.path) if (f.startswith(".qcg"))]
            for d in dirs:
                shutil.rmtree(os.path.join(self.path, d))
            self.path = None

    def test_executor_experiment_input(self) -> None:
        test_exp = create_default_experiment()
        test_exec = create_default_executor(experiment=test_exp)
        self.assertTrue(isinstance(test_exec.experiment, Experiment))
        self.assertEqual(test_exec.experiment, test_exp)

    def test_executor_submit(self) -> None:
        test_exp = create_default_experiment()
        test_exec = create_default_executor(experiment=test_exp)
        test_points = self.test_points
        test_exec.experiment.data_points = test_points
        job_ids = test_exec.submit()

        self.assertEqual(len(test_points), len(job_ids))

        # count no of jobs
        self.path = test_exec.experiment.system_setup.source_directory
        output_path = os.path.join(
            test_exec.experiment.system_setup.working_directory, ".."
        )
        job_dirs = [d for d in os.listdir(output_path)]
        self.assertEqual(len(job_ids), len(job_dirs))
        # check if jobs were finishes successfully
        service_dirs = [
            f for f in os.listdir(self.path) if (f.startswith(".qcgpjm-service"))
        ]
        with open(
            self.path + "/" + service_dirs[0] + "/" + "final_status.json", "r"
        ) as f:
            data = json.load(f)
        jobs_finished = data["JobStats"]["FinishedJobs"]
        jobs_failed = data["JobStats"]["FailedJobs"]
        self.assertEqual(jobs_finished, len(test_points))
        self.assertEqual(jobs_failed, 0)

    def test_executor_submit_with_analysis(self) -> None:
        """Check that when using an analysis script the right number of jobs are created as well."""
        analysis_exp = Experiment(
            experiment_name="default_exp",
            system_setup=SystemSetup(
                source_directory=os.getcwd(),
                program_name=DUMMY_FILE,
                command_line_arguments={"arg1": 1.0},
                analysis_script=DUMMY_FILE,  # now with analysis script
            ),
            parameters=[create_default_param()],
            opt_info_list=[],
        )

        test_exec = create_default_executor(analysis_exp)
        test_points = self.test_points
        test_exec.experiment.data_points = test_points
        job_ids = test_exec.submit()

        self.assertEqual(
            len(test_points) + 1, len(job_ids)
        )  # now one extra analysis job

        # count no of jobs
        self.path = test_exec.experiment.system_setup.source_directory
        output_path = os.path.join(
            test_exec.experiment.system_setup.working_directory, ".."
        )
        job_dirs = [
            d
            for d in os.listdir(output_path)
            if os.path.isdir(os.path.join(output_path, d))
        ]
        self.assertEqual(
            len(job_ids) - 1, len(job_dirs)
        )  # for the analysis job no dir is created
        self.assertEqual(
            len(job_dirs) + 2, len([d for d in os.listdir(output_path)])
        )  # but 2 additional files are created
        # check if jobs were finishes successfully
        service_dirs = [
            f for f in os.listdir(self.path) if (f.startswith(".qcgpjm-service"))
        ]
        with open(
            self.path + "/" + service_dirs[0] + "/" + "final_status.json", "r"
        ) as f:
            data = json.load(f)
        jobs_finished = data["JobStats"]["FinishedJobs"]
        jobs_failed = data["JobStats"]["FailedJobs"]
        self.assertEqual(
            jobs_finished, len(test_points) + 1
        )  # again one extra analysis job
        self.assertEqual(jobs_failed, 0)

    def test_executor_run(self) -> None:
        test_exec = create_default_executor(experiment=create_default_experiment())
        test_points = self.test_points
        test_exec.experiment.data_points = test_points
        test_exec.run()
        # todo: this tests nothing! add test
        self.path = (
            test_exec.experiment.system_setup.source_directory
        )  # path for tearDown

    def test_executor_collect_data(self) -> None:
        def tear_down_dirs(testpath: str, outputfile: str) -> None:
            """Helper function to tear down the temporary test dir."""
            try:
                os.remove(os.path.join(testpath, "step0", outputfile))
                os.remove(os.path.join(testpath, "step1", outputfile))
                os.remove(os.path.join(testpath, "step2", outputfile))
            except FileNotFoundError:
                pass
            os.removedirs(os.path.join(testpath, "step0"))
            os.removedirs(os.path.join(testpath, "step1"))
            os.removedirs(os.path.join(testpath, "step2"))

        for output_extension in ["csv", "json", "pickle"]:
            output_file = f"output.{output_extension}"

            test_exec = create_default_executor(experiment=create_default_experiment())
            test_exec.experiment.system_setup.output_extension = output_extension
            test_path = os.path.join(os.getcwd(), "temp_test_dir")
            if os.path.exists(test_path):
                tear_down_dirs(test_path, output_file)
            os.makedirs(test_path)

            # test with analysis script
            test_exec.experiment.system_setup.analysis_script = DUMMY_FILE
            test_exec.experiment.system_setup.working_directory = test_path

            test_df = pandas.DataFrame({"f": [1, 2, 3], "x": [4, 5, 6], "y": [7, 8, 9]})
            # save dataframe as output.csv with whitespace as delimiter
            test_df.to_csv("output.csv", index=False, sep=" ")

            data = test_exec.collect_data()

            self.assertEqual(type(data), pandas.DataFrame)
            self.assertTrue(data.equals(test_df))

            os.remove("output.csv")

            # test without analysis script
            test_exec.experiment.system_setup.analysis_script = None
            os.makedirs(os.path.join(test_path, "step0"))
            os.makedirs(os.path.join(test_path, "step1"))
            os.makedirs(os.path.join(test_path, "step2"))
            test_exec.experiment.system_setup.working_directory = os.path.join(
                test_path, "step2"
            )

            test_df_1 = pandas.DataFrame({"f": [1], "x": [4], "y": [7]})
            test_df_2 = pandas.DataFrame({"f": [2], "x": [5], "y": [8]})
            test_df_3 = pandas.DataFrame({"f": [3], "x": [6], "y": [9]})
            if output_extension == "csv":
                # save dataframe as output.csv with whitespace as delimiter
                test_df_1.to_csv(
                    os.path.join(test_path, "step0", output_file), index=False, sep=" "
                )
                test_df_2.to_csv(
                    os.path.join(test_path, "step1", output_file), index=False, sep=" "
                )
                test_df_3.to_csv(
                    os.path.join(test_path, "step2", output_file), index=False, sep=" "
                )
            elif output_extension == "json":
                # save dataframe as output.json
                test_df_1.to_json(os.path.join(test_path, "step0", output_file))
                test_df_2.to_json(os.path.join(test_path, "step1", output_file))
                test_df_3.to_json(os.path.join(test_path, "step2", output_file))
            elif output_extension == "pickle":
                # save dataframe as output.pickle
                test_df_1.to_pickle(os.path.join(test_path, "step0", output_file))
                test_df_2.to_pickle(os.path.join(test_path, "step1", output_file))
                test_df_3.to_pickle(os.path.join(test_path, "step2", output_file))
            else:
                raise NotImplementedError

            data = test_exec.collect_data()

            # the columns here are not sorted so rearrange to match test_df
            data = data.sort_values(by="f")
            data = data.reset_index(drop=True)

            self.assertEqual(type(data), pandas.DataFrame)
            self.assertTrue(data.equals(test_df))

            tear_down_dirs(test_path, output_file)

    def test_executor_create_points(self) -> None:
        """Test create_points_based_on_optimization."""

        def mock_function(
            ga_instance: GA, solution: List[float], solution_idx: int
        ) -> float:
            return solution[0] ** 2 + solution[1] ** 2

        for evolutionary in [None, True, False]:
            test_executor = create_default_executor(
                create_default_experiment(
                    parameters=[
                        create_default_param(name="param1"),
                        create_default_param(name="param2"),
                    ]
                )
            )
            initial_num_points = len(test_executor.experiment.data_points)
            ga_opt = GAOpt(
                initial_data_points=test_executor.experiment.data_points,
                num_generations=100,
                num_parents_mating=9,
                fitness_func=mock_function,
                gene_type=float,
                gene_space={"low": 0.1, "high": 0.9},
                mutation_probability=0.02,  # todo: this line breaks it
                crossover_probability=0.7,
                refinement_factors=[0.1, 0.5],
                allow_duplicate_genes=False,
            )
            # self.assertTrue(ga_opt.ga_instance.allow_duplicate_genes is False)
            opt = Optimizer(ga_opt)
            test_executor.optimizer = opt
            test_executor.optimizer.optimization_algorithm = ga_opt
            test_executor.optimizer.optimization_algorithm.can_create_points_evolutionary = (
                True
            )

            x_list = [x for x, y in ga_opt.optimization_instance.population]
            y_list = [y for x, y in ga_opt.optimization_instance.population]
            c_list = [
                mock_function(
                    ga_instance=test_executor.optimizer.optimization_algorithm.optimization_instance,
                    solution=sol,
                    solution_idx=0,
                )
                for sol in ga_opt.optimization_instance.population
            ]
            data = pandas.DataFrame({"f": c_list, "x": x_list, "y": y_list})

            test_executor.create_points_based_on_optimization(
                data=data, evolutionary=evolutionary
            )
            self.assertEqual(
                test_executor.optimizer.optimization_algorithm.optimization_instance.generations_completed,
                1,
            )
            new_points = test_executor.experiment.data_points
            print("types are:", type(new_points), type(new_points[0]))
            self.assertIsInstance(new_points, np.ndarray)  # correct type
            self.assertEqual(len(new_points), initial_num_points)  # correct num points
            [
                self.assertEqual(len(point), 2) for point in new_points  # type: ignore[func-returns-value]
            ]  # each point has two param values
            # s = set([tuple(x) for x in new_points])
            # self.assertEqual(len(s), initial_num_points)                    # all points are unique
            if evolutionary is not False:
                for point in new_points:
                    self.assertTrue(
                        all(0.1 <= x <= 0.9 for x in point)
                    )  # each point is within constraint
            (
                best_solution,
                _,
                _,
            ) = test_executor.optimizer.optimization_algorithm.get_best_solution()
            assert all(math.isclose(0.1, x) for x in best_solution)

    @pytest.mark.xfail(
        reason="pygad can not guarantee uniqueness of genes even with allow_duplicate_genes=False."
    )
    def test_executor_create_points_uniqueness(self) -> None:
        """Test create_points_based_on_optimization."""
        # todo: merge this test with the above once uniqueness is fixed

        def mock_function(
            ga_instance: GA, solution: List[float], solution_idx: int
        ) -> Any:
            return solution[0] ** 2 + solution[1] ** 2

        for evolutionary in [None, True, False]:
            test_executor = create_default_executor(
                create_default_experiment(
                    parameters=[
                        create_default_param(name="param1"),
                        create_default_param(name="param2"),
                    ]
                )
            )
            initial_num_points = len(test_executor.experiment.data_points)
            ga_opt = GAOpt(
                initial_data_points=test_executor.experiment.data_points,
                num_generations=100,
                num_parents_mating=9,
                fitness_func=mock_function,
                gene_type=float,
                gene_space={"low": 0.1, "high": 0.9},
                mutation_probability=0.02,  # todo: this line break it
                crossover_probability=0.7,
                refinement_factors=[0.1, 0.5],
                allow_duplicate_genes=False,
            )
            # self.assertTrue(ga_opt.ga_instance.allow_duplicate_genes is False)
            opt = Optimizer(ga_opt)
            test_executor.optimizer = opt
            test_executor.optimizer.optimization_algorithm = ga_opt
            test_executor.optimizer.optimization_algorithm.can_create_points_evolutionary = (
                True
            )

            x_list = [x for x, y in ga_opt.optimization_instance.population]
            y_list = [y for x, y in ga_opt.optimization_instance.population]
            c_list = [
                mock_function(
                    ga_instance=test_executor.optimizer.optimization_algorithm.optimization_instance,
                    solution=sol,
                    solution_idx=0,
                )
                for sol in ga_opt.optimization_instance.population
            ]
            data = pandas.DataFrame({"f": c_list, "x": x_list, "y": y_list})

            test_executor.create_points_based_on_optimization(
                data=data, evolutionary=evolutionary
            )
            self.assertEqual(
                test_executor.optimizer.optimization_algorithm.optimization_instance.generations_completed,
                1,
            )
            new_points = test_executor.experiment.data_points
            s = set([tuple(x) for x in new_points])
            self.assertEqual(len(s), initial_num_points)  # all points are unique


if __name__ == "__main__":
    unittest.main()
