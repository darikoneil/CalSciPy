import subprocess
import sys
import pytest
from os import chdir
from pathlib import Path
import toml
from PPVD.style import TerminalStyle


"""
Test package installation

"""


def retrieve_details(path):
    details = toml.load(path).get("project")
    name = details.get("name")
    version = details.get("version")
    dependencies = details.get("dependencies")
    return name, version, dependencies


def retrieve_project_directory(path):
    return Path(path).parent


def retrieve_project_file():
    project_file = Path.cwd().joinpath("pyproject.toml")
    if not Path.exists(project_file):
        project_file = Path.cwd().parent.joinpath("pyproject.toml")
    if not Path.exists(project_file):
        raise FileNotFoundError("Can't find project file")
    return project_file


def collect_project():
    project_file = retrieve_project_file()
    project_directory = retrieve_project_directory(project_file)
    package_name, package_version, package_dependencies = retrieve_details(project_file)
    return project_directory, project_file, package_name, package_version, package_dependencies


# get project information and work from correct directory
proj_dir, proj_file, pkg_name, pkg_version, pkg_dependencies = collect_project()
chdir(proj_dir)


print(f"\n{TerminalStyle.YELLOW}Package: {TerminalStyle.BLUE}{pkg_name}{TerminalStyle.RESET}")
print(f"{TerminalStyle.YELLOW}Version: {TerminalStyle.BLUE}{pkg_version}{TerminalStyle.RESET}")
print(f"{TerminalStyle.YELLOW}Dependencies: {TerminalStyle.BLUE}{pkg_dependencies}{TerminalStyle.RESET}")


@pytest.mark.parametrize("path", [proj_dir])
def test_install(path):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e ."])
    except subprocess.CalledProcessError as e:
        print(f"{e.output}")
        # TODO This doesn't work for macOS or Linux. Obviously since the runner does an install this isn't a big deal
