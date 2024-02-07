from pathlib import Path
import importlib
import pkgutil
from typing import Dict, List
from erllm.utils import make_markdown_table2


def discover_packages(root_folder) -> List[pkgutil.ModuleInfo]:
    root_path = Path(root_folder)
    modules = list(pkgutil.walk_packages([str(root_path)]))
    return list(filter(lambda module_info: module_info.ispkg, modules))


def get_immediate_submodules(pkg: pkgutil.ModuleInfo):
    loader = pkgutil.get_loader(pkg.name)
    pkg_path = Path(loader.get_filename()).parent
    submodules = []
    # Iterate through the submodules of the specified package
    for _, submodule_name, is_pkg in pkgutil.iter_modules([pkg_path]):
        submodule_path = f"{pkg.name}.{submodule_name}"
        # Check if the submodule is not a package itself
        if not is_pkg:
            submodules.append(submodule_path)
    return submodules


def get_docstring(module_name: str) -> str:
    ds = importlib.import_module(module_name).__doc__
    return "" if not ds else ds


if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent  # folder of file
    pkgs = discover_packages(root_folder)
    pkg_docstring = {}
    for pkg in pkgs:
        pkg_docstring[pkg] = get_docstring(pkg.name).strip().replace("\n", " ")
    pkg_name_docstrings = map(lambda pkg: (pkg.name, pkg_docstring[pkg]), pkgs)
    all_pkgs_array = [("Module", "Purpose")] + list(pkg_name_docstrings)
    all_pkgs_table = make_markdown_table2(all_pkgs_array)
    with open("package_docstrings.md", "w") as f:
        f.write(all_pkgs_table)
    pkg_submods: Dict[pkgutil.ModuleInfo, List[str]] = {}
    submods: List[str] = []
    pkg_tables = []
    for pkg in pkgs:
        submods = get_immediate_submodules(pkg)
        submod_docstrings = list(
            map(
                lambda submod: (
                    submod,
                    get_docstring(submod).strip().replace("\n", " "),
                ),
                submods,
            )
        )
        pkg_array = [("Module", "Purpose")] + submod_docstrings
        pkg_table = make_markdown_table2(pkg_array)
        pkg_tables.append(pkg_table)
        print(submod_docstrings)
    with open("package_docstrings.md", "w") as f:
        f.write(all_pkgs_table)
        for pkg_table in pkg_tables:
            f.write(pkg_table)

"""

def discover_modules_with_submodules_old(root_folder):
    root_module_name = "erllm"
    root_path = Path(root_folder)
    modules = list(pkgutil.walk_packages([str(root_path)]))
    root_module: pkgutil.ModuleInfo = next(
        filter(lambda module_info: module_info.name == root_module_name, modules)
    )
    root_module_path = (
        root_module.module_finder.path + "/" + root_module.name.replace(".", "/")
    )
    module_docstrings = {}
    module_docstrings[root_module.name] = importlib.import_module(
        root_module.name
    ).__doc__
    submodules = list(pkgutil.walk_packages([root_module.module_finder.path]))
    packages = list(filter(lambda module_info: module_info.ispkg, submodules))
    packages.insert(0, root_module)
    for subpkg in packages:
        print("Subpackage:", subpkg.name)
        print("Docstring:", importlib.import_module(subpkg.name).__doc__)

        
        
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
