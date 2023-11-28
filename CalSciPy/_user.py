from __future__ import annotations
import tkinter as tk
from tkinter.filedialog import askdirectory
from pathlib import Path
from shutil import copy2
from functools import partial

from tqdm import tqdm
from joblib import Parallel, delayed

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
def verbose_copying(source: Path, dest: Path, content_string: str = "") -> int:
    """

    Verbose copying from source to dest

    :param source: source directory
    :type source: pathlib.Path
    :param dest: destination directory
    :type dest: pathlib.Path
    :param content_string: content to display in loading bar
    :type content_string: str
    :returns: 0 if successful
    """

    def copy_(dest_: Path, source_: Path, file_: Path) -> None:
        destination = dest_.joinpath(file_.relative_to(source_))
        copy2(file_, destination)

    # collect files
    files = [file for file in source.rglob("*") if file.is_file()]

    # collect folders
    folders = [folder for folder in source.rglob("*") if not folder.is_file()]

    # make tree
    dest.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        dest_folder = dest.joinpath(folder.relative_to(source))
        dest_folder.mkdir(parents=True, exist_ok=True)

    # create func handle for parallel
    func_handle = partial(copy_, dest, source)

    # copy
    _ = (Parallel(n_jobs=-1, backend="loky")
         (delayed(func_handle)(file) for file in tqdm(files,
                                                      total=len(files),
                                                      desc=f"Copying {content_string} files")))  # noqa: F841

    return 0
