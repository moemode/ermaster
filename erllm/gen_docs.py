from pathlib import Path
import importlib
import pkgutil

from erllm.utils import make_markdown_table2


def discover_modules_with_submodules(root_folder) -> dict[pkgutil.ModuleInfo, str]:
    root_path = Path(root_folder)
    modules = list(pkgutil.walk_packages([str(root_path)]))
    packages = list(filter(lambda module_info: module_info.ispkg, modules))
    pkg_docstring = {}
    for pkg in packages:
        docstring = importlib.import_module(pkg.name).__doc__
        if docstring:
            # remove all newlines and replace with a single space
            docstring = docstring.strip().replace("\n", " ")
        pkg_docstring[pkg] = docstring
    return pkg_docstring


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


if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent  # folder of file
    pkg_docstrings = discover_modules_with_submodules(root_folder)
    pkg_name_docstrings = map(
        lambda pkg: (pkg.name, pkg_docstrings[pkg]), pkg_docstrings
    )
    pkg_docstring_tbl_array = [("Module", "Purpose")] + list(pkg_name_docstrings)
    mdt = make_markdown_table2(pkg_docstring_tbl_array)
    with open("package_docstrings.md", "w") as f:
        f.write(mdt)
    for pkg, docstring in pkg_docstrings.items():
        submods = get_immediate_submodules(pkg)
        print(f"Submodules of {pkg.name}: {submods}")


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
