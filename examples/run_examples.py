import importlib
import inspect
import os
import sys
from typing import Any
from typing import Callable

# This file is used to run all the examples in this folder and subfolders
# Each file should be of the form example*.py and contain a `main`-method.
# If the `main`-method takes an argument `no_output` this will be passed as `True`
# to avoid creating plots etc.


def main() -> None:
    path_to_here = os.path.dirname(os.path.abspath(__file__))
    for root, folders, files in os.walk(path_to_here):
        for filename in files:
            if filename.startswith("example") and filename.endswith(".py"):
                # exclude blueprint example on pipeline as it requires a lot of other code
                if not filename.startswith("example_blueprint_main"):
                    filepath = os.path.join(root, filename)
                    _run_example(filepath)


def _run_example(filepath: str) -> None:
    cwd = os.getcwd()
    sys.path.append(os.path.dirname(filepath))
    example_module_name = os.path.basename(filepath)[: -len(".py")]
    example_module = importlib.import_module(example_module_name)
    print(hasattr(example_module, "main"))
    if hasattr(example_module, "main"):
        main = getattr(example_module, "main")
    else:
        return
    os.chdir(os.path.dirname(filepath))
    if _has_no_output_arg(main):
        main(no_output=True)
    else:
        main()
    os.chdir(cwd)
    sys.path.pop()


def _has_no_output_arg(func: Callable[..., Any]) -> bool:
    return "no_output" in inspect.getfullargspec(func).args


if __name__ == "__main__":
    main()
