import subprocess
import sys
import os
from PPVD.style import TerminalStyle
import pytest
import toml
import importlib


def retrieve_details(path):
    details = toml.load(path).get("project")
    name = details.get("name")
    version = details.get("version")
    dependencies = details.get("dependencies")
    return name, version, dependencies
    

pyproject_file = os.path.join(os.getcwd(), "pyproject.toml")

package_name, package_version, package_dependencies = retrieve_details(pyproject_file)

print(f"\n\n{TerminalStyle.ORANGE}Testing...{TerminalStyle.RESET}")
print(f"{TerminalStyle.YELLOW}Package: {TerminalStyle.BLUE}{package_name}{TerminalStyle.RESET}")
print(f"{TerminalStyle.YELLOW}Version: {TerminalStyle.BLUE}{package_version}{TerminalStyle.RESET}")
print(f"{TerminalStyle.YELLOW}Dependencies: {TerminalStyle.BLUE}{package_dependencies}{TerminalStyle.RESET}")


@pytest.mark.parametrize("path", [pyproject_file])
def test_install(path):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e ."])
