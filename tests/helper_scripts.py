import os
import toml
from pathlib import Path


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
