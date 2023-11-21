from __future__ import annotations
import tkinter as tk
from tkinter.filedialog import askdirectory
from pathlib import Path
from tqdm import tqdm
from shutil import copytree, copy2, rmtree
from ._validators import convert_permitted_types_to_required


def select_directory(**kwargs) -> str:
    """
    Interactive tool for directory selection. All keyword arguments are
    passed to `tkinter.filedialog.askdirectory <https://docs.python.org/3/library/tk.html>`_

    :param kwargs: keyword arguments passed to tkinter.filedialog.askdirectory
    :return: absolute path to directory
    :rtype: str
    """
    # Make Root
    root = tk.Tk()

    # collect path & format
    # noinspection PyArgumentList
    path = askdirectory(**kwargs)
    path = Path(path)

    if str(path) == ".":
        raise FileNotFoundError

    # destroy root
    root.destroy()
    return path


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=1)
def verbose_copying(source: Path, dest: Path, content_string: str = "") -> None:
    """

    Verbose copying from source to dest

    :param source: source directory
    :type source: pathlib.Path
    :param dest: destination directory
    :type dest: pathlib.Path
    :param content_string: content to display in loading bar
    :type content_string: str
    :rtype: None
    """
    def copy_(source_: str, dest_: str) -> None:
        copy2(source_, dest_)
        pbar.update(1)

    num_files = sum([1 for file in source.rglob("*") if file.is_file()])
    pbar = tqdm(total=num_files)
    pbar.set_description("".join(["Copying ", content_string, " files"]))
    try:
        copytree(source, dest, copy_function=copy_, dirs_exist_ok=True)
    except TypeError:
        rmtree(dest)
        copytree(source, dest, copy_function=copy_)