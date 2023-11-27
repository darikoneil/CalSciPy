from __future__ import annotations
from typing import Optional
from pathlib import Path

from ..._validators import convert_permitted_types_to_required, validate_filename
from ..factories import BrukerXMLFactory
# noinspection PyProtectedMember
from ..xml.xmlobj import _BrukerObject


# DEFAULT LOCATION / NAME FOR SAVING PROTOCOLS
_DEFAULT_PATH = Path.cwd().joinpath("prairieview_protocol.xml")


@validate_filename(pos=1)
@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=1)
def write_protocol(protocol: _BrukerObject,
                   file_path: Path = _DEFAULT_PATH,
                   ext: str = ".xml",
                   name: Optional[str] = None) -> None:
    """
    Write prairieview protocol to file

    :param protocol: prairieview object
    :param file_path: file path for saving file
    :param name: name of protocol
    :param ext: file extension for saving file
    """

    # If name  provided, append name
    if name is not None:
        file_path = file_path.joinpath(name)

    # If no extension, append ext
    if file_path.suffix == '':
        file_path = file_path.with_suffix(ext)

    factory = BrukerXMLFactory()
    lines = factory.constructor(protocol)
    with open(file_path, "w+") as file:
        for line in lines:
            file.write(line)
