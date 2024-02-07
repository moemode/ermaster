from pathlib import Path
import importlib
import pkgutil
from typing import Dict, Iterable, List, OrderedDict, Tuple
from typing import List
import pkgutil
from pathlib import Path
from erllm.utils import make_markdown_table


def discover_packages(root_folder: Path) -> List[pkgutil.ModuleInfo]:
    """
    Discover packages within a given root folder.

    Args:
        root_folder (Path): The root folder to search for packages.

    Returns:
        List[pkgutil.ModuleInfo]: A list of package information objects.

    """
    modules = list(pkgutil.walk_packages([str(root_folder)]))
    return list(filter(lambda module_info: module_info.ispkg, modules))


def get_direct_subfiles(pkg: pkgutil.ModuleInfo) -> Iterable[str]:
    """
    Get the immediate subfiles of a package.

    Args:
        pkg (pkgutil.ModuleInfo): The package to get the submodules from.

    Returns:
        list: A list of fully qualified submodule paths.
    """
    loader = pkgutil.get_loader(pkg.name)
    pkg_path = Path(loader.get_filename()).parent
    submodules = []
    # Iterate through the submodules of the specified package
    for _, submodule_name, is_pkg in pkgutil.iter_modules([pkg_path]):
        full_name = f"{pkg.name}.{submodule_name}"
        # Check if the submodule is not a package itself
        if not is_pkg:
            submodules.append(full_name)
    return submodules


def get_docstring(module_name: str) -> str:
    """
    Retrieve the docstring of a module.

    Args:
        module_name (str): The name of the module.

    Returns:
        str: The docstring of the module, or an empty string if no docstring is found.
    """
    ds = importlib.import_module(module_name).__doc__
    return "" if not ds else ds


def package_table(pkgs: List[pkgutil.ModuleInfo]) -> str:
    pkg_name_docstrings = map(
        lambda pkg: (pkg.name, get_docstring(pkg.name).strip().replace("\n", " ")), pkgs
    )
    return layout_package_table(pkg_name_docstrings)


def layout_package_table(pkg_name_docstrings: Iterable[Tuple[str, str]]) -> str:
    """
    Generate a markdown table for package names and their corresponding docstrings.

    Args:
        pkg_name_docstrings (Iterable[Tuple[str, str]]): An iterable of tuples containing package names and their docstrings.

    Returns:
        str: The generated markdown table.
    """
    all_pkgs_array = [("Module", "Purpose")]
    for pkg_name, docstring in pkg_name_docstrings:
        # add a link to the section where the package is described
        pkg_cell = f"[{pkg_name}](#package-{pkg_name.replace('.', '')})"
        all_pkgs_array.append((pkg_cell, docstring))
    return make_markdown_table(all_pkgs_array)


def subfile_table(pkg: pkgutil.ModuleInfo) -> str:
    submods = get_direct_subfiles(pkg)
    submod_docstrings = map(
        lambda submod: (
            submod,
            get_docstring(submod).strip().replace("\n", " "),
        ),
        submods,
    )
    return layout_subfile_table(submod_docstrings)


def layout_subfile_table(submod_docstrings: Iterable[Tuple[str, str]]) -> str:
    """
    Generate a markdown table containing the files in a package.

    Args:
        submod_docstrings: An iterable of tuples containing submodule names and their corresponding docstrings.

    Returns:
        A string representing the generated markdown table.
    """
    pkg_array = [("Module", "Purpose")]
    for submod, docstring in submod_docstrings:
        submod_name = submod.split(".")[-1] + ".py"
        submod_path = submod.replace(".", "/") + ".py"
        pkg_array.append((f"[{submod_name}]({submod_path})", docstring))
    return make_markdown_table(pkg_array)


def get_doc(all_pkgs_table: str, file_tables: Dict[pkgutil.ModuleInfo, str]):
    """
    Generate documentation for packages and files.

    Args:
        all_pkgs_table (str): Table containing information about all packages.
        file_tables (Dict[pkgutil.ModuleInfo, str]): Dictionary containing file table for each package.

    Returns:
        str: Generated documentation.
    """
    d = "# Package Overview\n"
    d += all_pkgs_table
    for pkg, pkg_table in file_tables.items():
        d += "\n## Package: " + pkg.name + "\n"
        d += pkg_table
    return d


if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent  # folder of file
    pkgs = discover_packages(root_folder)
    all_pkgs_table = package_table(pkgs)
    file_tables = OrderedDict()
    for pkg in pkgs:
        file_tables[pkg] = subfile_table(pkg)
    doc = get_doc(all_pkgs_table, file_tables)
    with open("package_docstrings.md", "w") as f:
        f.write(doc)
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
