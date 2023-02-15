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
    print(f"\n{TerminalStyle.ORANGE}Testing...{TerminalStyle.RESET}\n")
    print(f"{TerminalStyle.YELLOW}Package: {TerminalStyle.BLUE}{name}{TerminalStyle.RESET}\n")
    print(f"{TerminalStyle.YELLOW}Version: {TerminalStyle.BLUE}{version}{TerminalStyle.RESET}\n")
    print(f"{TerminalStyle.YELLOW}Dependencies: {TerminalStyle.BLUE}{dependencies}{TerminalStyle.RESET}\n")
    return name


pyproject_file = "".join([os.path.dirname(os.getcwd()), "\\pyproject.toml"])


package_name = retrieve_details(pyproject_file)

uninstall_instructions = [pyproject_file, package_name]


@pytest.mark.parametrize(("path", "name"), [uninstall_instructions])
def test_uninstall(path, name):
    original_path = os.getcwd()
    os.chdir("../")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", name])
    os.chdir(original_path)
