import subprocess
import sys
from PPVD.style import TerminalStyle
import pytest
import os
from pathlib import Path
import toml


def retrieve_details(path):
    details = toml.load(path).get("project")
    name = details.get("name")
    version = details.get("version")
    dependencies = details.get("dependencies")
    return name, version, dependencies


def retrieve_project_directory(path):
    return Path(path).parent


def retrieve_project_file():
    project_file = os.path.join(os.getcwd(), "pyproject.toml")
    if not os.path.exists(project_file):
        project_file = os.path.join(Path(os.getcwd()).parent, "pyproject.toml")
    if not os.path.exists(project_file):
        raise FileNotFoundError("Can't find project file")
    return project_file


def collect_project():
    project_file = retrieve_project_file()
    project_directory = retrieve_project_directory(project_file)
    package_name, package_version, package_dependencies = retrieve_details(project_file)
    return project_directory, project_file, package_name, package_version, package_dependencies


# get project information and work from correct directory
project_dir, project_file, package_name, package_version, package_dependencies = collect_project()
os.chdir(project_dir)


print(f"\n{TerminalStyle.YELLOW}Package: {TerminalStyle.BLUE}{package_name}{TerminalStyle.RESET}")
print(f"{TerminalStyle.YELLOW}Version: {TerminalStyle.BLUE}{package_version}{TerminalStyle.RESET}")
print(f"{TerminalStyle.YELLOW}Dependencies: {TerminalStyle.BLUE}{package_dependencies}{TerminalStyle.RESET}")


@pytest.mark.parametrize("path", [project_file])
def test_install(path):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e ."])
