"""
File-level read/write functionality.
"""
from typing import List, Dict, Tuple, Optional, BinaryIO, TypeVar, Type
import io
from datetime import datetime
from dataclasses import dataclass

from .basic import KlamathError
from .record import Record

from .records import HEADER, BGNLIB, ENDLIB, UNITS, LIBNAME
from .records import BGNSTR, STRNAME, ENDSTR
from .records import BOX, BOUNDARY, NODE, PATH, TEXT, SREF, AREF
from .elements import Element, Reference, Text, Box, Boundary, Path, Node


FH = TypeVar('FH', bound='FileHeader')


@dataclass
class FileHeader:
    """
    Representation of the GDS file header.

    File header records: HEADER BGNLIB LIBNAME UNITS
       Optional record are ignored if present and never written.

    Version is written as `600`.
    """
    name: bytes
    """ Library name """

    user_units_per_db_unit: float
    """ Number of user units in one database unit """

    meters_per_db_unit: float
    """ Number of meters in one database unit """

    mod_time: datetime = datetime(1900, 1, 1)
    """ Last-modified time """

    acc_time: datetime = datetime(1900, 1, 1)
    """ Last-accessed time """

    @classmethod
    def read(cls: Type[FH], stream: BinaryIO) -> FH:
        """
        Read and construct a header from the provided stream.

        Args:
            stream: Seekable stream to read from

        Returns:
            FileHeader object
        """
        version = HEADER.read(stream)[0]
        mod_time, acc_time = BGNLIB.read(stream)
        name = LIBNAME.skip_and_read(stream)
        uu, dbu = UNITS.skip_and_read(stream)

        return cls(mod_time=mod_time, acc_time=acc_time, name=name,
                   user_units_per_db_unit=uu, meters_per_db_unit=dbu)

    def write(self, stream: BinaryIO) -> int:
        """
        Write the header to a stream

        Args:
            stream: Stream to write to

        Returns:
            number of bytes written
        """
        b = HEADER.write(stream, 600)
        b += BGNLIB.write(stream, (self.mod_time, self.acc_time))
        b += LIBNAME.write(stream, self.name)
        b += UNITS.write(stream, (self.user_units_per_db_unit, self.meters_per_db_unit))
        return b


def scan_structs(stream: BinaryIO) -> Dict[bytes, int]:
    """
    Scan through a GDS file, building a table of
      {b'structure_name': byte_offset}.
    The intent of this function is to enable random access
     and/or partial (structure-by-structure) reads.

    Args:
        stream: Seekable stream to read from. Should be positioned
                before the first structure record, but possibly
                already past the file header.
    """
    positions = {}

    size, tag = Record.read_header(stream)
    while tag != ENDLIB.tag:
        stream.seek(size, io.SEEK_CUR)
        if tag == BGNSTR.tag:
            name = STRNAME.read(stream)
            if name in positions:
                raise KlamathError(f'Duplicate structure name: {name!r}')
            positions[name] = stream.tell()
        size, tag = Record.read_header(stream)

    return positions


def try_read_struct(stream: BinaryIO) -> Optional[Tuple[bytes, List[Element]]]:
    """
    Skip to the next structure and attempt to read it.

    Args:
        stream: Seekable stream to read from.

    Returns:
        (name, elements) if a structure was found.
        None if no structure was found before the end of the library.
    """
    if not BGNSTR.skip_past(stream):
        return None
    name = STRNAME.read(stream)
    elements = read_elements(stream)
    return name, elements


def write_struct(stream: BinaryIO,
                 name: bytes,
                 elements: List[Element],
                 cre_time: datetime = datetime(1900, 1, 1),
                 mod_time: datetime = datetime(1900, 1, 1),
                 ) -> int:
    """
    Write a structure to the provided stream.

    Args:
        name: Structure name (ascii-encoded).
        elements: List of Elements containing the geometry and text in this struct.
        cre_time: Creation time (optional).
        mod_time: Modification time (optional).

    Return:
        Number of bytes written
    """
    b = BGNSTR.write(stream, (cre_time, mod_time))
    b += STRNAME.write(stream, name)
    b += sum(el.write(stream) for el in elements)
    b += ENDSTR.write(stream, None)
    return b


def read_elements(stream: BinaryIO) -> List[Element]:
    """
    Read elements from the stream until an ENDSTR
      record is encountered. The ENDSTR record is also
      consumed.

    Args:
        stream: Seekable stream to read from.

    Returns:
        List of element objects.
    """
    data: List[Element] = []
    size, tag = Record.read_header(stream)
    while tag != ENDSTR.tag:
        if tag == BOUNDARY.tag:
            data.append(Boundary.read(stream))
        elif tag == PATH.tag:
            data.append(Path.read(stream))
        elif tag == NODE.tag:
            data.append(Node.read(stream))
        elif tag == BOX.tag:
            data.append(Box.read(stream))
        elif tag == TEXT.tag:
            data.append(Text.read(stream))
        elif tag == SREF.tag:
            data.append(Reference.read(stream))
        elif tag == AREF.tag:
            data.append(Reference.read(stream))
        else:
            # don't care, skip
            stream.seek(size, io.SEEK_CUR)
        size, tag = Record.read_header(stream)
    return data
