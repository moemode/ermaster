"""
Generates a .pth file with the absolute path to the parent folder of erllm and adds it to the site-packages directory of the current Python interpreter. 
This makes erllm behave as if it were a installed third-party packages. 
Yet it also supports code changes without reinstall.
"""
from pathlib import Path
import site

# Get the absolute path of the erllm folder
script_folder = Path(__file__).resolve().parent.parent
# Specify the content for the .pth file as the absolute path of the script's folder
pth_content = str(script_folder)
# Get the site-packages directory of the current interpreter
site_packages_dir = Path(site.getsitepackages()[0])
pth_file_path = site_packages_dir / "erllm.pth"
pth_file_path.write_text(pth_content)
print(f".pth file added to site-packages: {pth_file_path}")
