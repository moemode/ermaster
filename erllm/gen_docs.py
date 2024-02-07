from pathlib import Path
import importlib
import pkgutil
from typing import Dict, List, OrderedDict
from erllm.utils import make_markdown_table


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


def package_table(pkg_name_docstrings):
    all_pkgs_array = [("Module", "Purpose")]
    for pkg_name, docstring in pkg_name_docstrings:
        pkg_cell = f"[{pkg_name}](#package-{pkg_name.replace('.', '')})"
        all_pkgs_array.append((pkg_cell, docstring))
    all_pkgs_table = make_markdown_table(all_pkgs_array)
    return all_pkgs_table


def submodule_table(pkg, submod_docstrings):
    pkg_array = [("Module", "Purpose")]
    for submod, docstring in submod_docstrings:
        submod_name = submod.split(".")[-1] + ".py"
        submod_path = submod.replace(".", "/") + ".py"
        pkg_array.append((f"[{submod_name}]({submod_path})", docstring))
    return make_markdown_table(pkg_array)


if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent  # folder of file
    pkgs = discover_packages(root_folder)
    pkg_docstring = {}
    for pkg in pkgs:
        pkg_docstring[pkg] = get_docstring(pkg.name).strip().replace("\n", " ")
    pkg_name_docstrings = map(lambda pkg: (pkg.name, pkg_docstring[pkg]), pkgs)
    all_pkgs_table = package_table(pkg_name_docstrings)
    pkg_submods: Dict[pkgutil.ModuleInfo, List[str]] = {}
    submods: List[str] = []
    pkg_tables = OrderedDict()
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
        pkg_table = submodule_table(pkg, submod_docstrings)
        pkg_tables[pkg] = pkg_table
        print(submod_docstrings)
    with open("package_docstrings.md", "w") as f:
        f.write("# Package Overview\n")
        f.write(all_pkgs_table)
        for pkg, pkg_table in pkg_tables.items():
            f.write("\n## Package: " + pkg.name + "\n")
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
