from pathlib import Path
import importlib
import pkgutil


def discover_modules_with_submodules(root_folder):
    root_path = Path(root_folder)
    submodules = list(pkgutil.walk_packages([str(root_path)]))
    subpackages = filter(lambda module_info: module_info.ispkg, submodules)
    for module_info in list(pkgutil.walk_packages([str(root_path)])):
        module_name = module_info.name
        module_path = root_path / module_name.replace(".", "/")

        try:
            module = importlib.import_module(module_name)
            docstring = module.__doc__

            submodules = list(pkgutil.walk_packages([str(module_path)]))
            if submodules:
                print(f"Module: {module_name}")
                print(f"Docstring: {docstring}")
                print("Submodules:")
                for submodule_info in submodules:
                    print(f"  - {submodule_info.name}")
                print("-" * 50)

        except ImportError as e:
            print(f"Error importing module {module_name}: {e}")


if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent  # folder of file
    discover_modules_with_submodules(root_folder)


"""
def discover_modules(root_folder):
    root_path = pathlib.Path(root_folder)

    for module_info in pkgutil.walk_packages([str(root_path)]):
        module_name = module_info.name
        module_path = root_path / module_name.replace(".", "/")

        try:
            module = importlib.import_module(module_name)
            docstring = module.__doc__

            print(f"Module: {module_name}")
            print(f"Docstring: {docstring}")
            print("-" * 50)

        except ImportError as e:
            print(f"Error importing module {module_name}: {e}")


if __name__ == "__main__":
    root_folder = "."  # Change this to the root folder of your module
    discover_modules(root_folder)
"""
