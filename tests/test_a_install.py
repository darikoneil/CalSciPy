import subprocess
import sys
from PPVD.style import TerminalStyle
import pytest
from tests.helper_scripts import collect_project
import os

# get project information and work from correct directory
project_dir, project_file, package_name, package_version, package_dependencies = collect_project()
os.chdir(project_dir)


print(f"\n{TerminalStyle.YELLOW}Package: {TerminalStyle.BLUE}{package_name}{TerminalStyle.RESET}")
print(f"{TerminalStyle.YELLOW}Version: {TerminalStyle.BLUE}{package_version}{TerminalStyle.RESET}")
print(f"{TerminalStyle.YELLOW}Dependencies: {TerminalStyle.BLUE}{package_dependencies}{TerminalStyle.RESET}")


@pytest.mark.parametrize("path", [project_file])
def test_install(path):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e ."])
