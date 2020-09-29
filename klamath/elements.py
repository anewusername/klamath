"""
Functionality for reading/writing elements (geometry, text labels,
 structure references) and associated properties.
"""
from typing import Dict, Tuple, Optional, BinaryIO, TypeVar, Type
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy        # type: ignore

from .basic import KlamathError
from .record import Record

from .records import BOX, BOUNDARY, NODE, PATH, TEXT, SREF, AREF
from .records import DATATYPE, PATHTYPE, BOXTYPE, NODETYPE, TEXTTYPE
from .records import LAYER, XY, WIDTH, COLROW, PRESENTATION, STRING
from .records import STRANS, MAG, ANGLE, PROPATTR, PROPVALUE
from .records import ENDEL, BGNEXTN, ENDEXTN, SNAME


E = TypeVar('E', bound='Element')

R = TypeVar('R', bound='Reference')
B = TypeVar('B', bound='Boundary')
P = TypeVar('P', bound='Path')
N = TypeVar('N', bound='Node')
T = TypeVar('T', bound='Text')
X = TypeVar('X', bound='Box')



def read_properties(stream: BinaryIO) -> Dict[int, bytes]:
    """
    Read element properties.

    Assumes PROPATTR records have unique values.
    Stops reading after consuming ENDEL record.

    Args:
        stream: Stream to read from.

    Returns:
        {propattr: b'propvalue'} mapping.
    """
    properties = {}

    size, tag = Record.read_header(stream)
    while tag != ENDEL.tag:
        if tag == PROPATTR.tag:
            key = PROPATTR.read_data(stream, size)[0]
            value = PROPVALUE.read(stream)
            if key in properties:
                raise KlamathError(f'Duplicate property key: {key!r}')
            properties[key]  = value
        size, tag = Record.read_header(stream)
    return properties


def write_properties(stream: BinaryIO, properties: Dict[int, bytes]) -> int:
    """
    Write element properties.

    This is does _not_ write the ENDEL record.

    Args:
        stream: Stream to write to.
    """
    b = 0
    for key, value in properties.items():
        b += PROPATTR.write(stream, key)
        b += PROPVALUE.write(stream, value)
    return b


class Element(metaclass=ABCMeta):
    """
    Abstract method definition for GDS structure contents
    """
    @classmethod
    @abstractmethod
    def read(cls: Type[E], stream: BinaryIO) -> E:
        """
        Read from a stream to construct this object.
        Consumes up to (and including) the ENDEL record.

        Args:
            Stream to read from.

        Returns:
            Constructed object.
        """
        pass

    @abstractmethod
    def write(self, stream: BinaryIO) -> int:
        """
        Write this element to a stream.
        Finishes with an ENDEL record.

        Args:
            Stream to write to.

        Returns:
            Number of bytes written
        """
        pass


@dataclass
class Reference(Element):
    """
    Datastructure representing
      an instance of a structure (SREF / structure reference) or
      an array of instances      (AREF / array reference).
    Type is determined by the presence of the `colrow` tuple.

    Transforms are applied to each individual instance (_not_
      to the instance's origin location or array vectors).
    """
    __slots__ = ('struct_name', 'invert_y', 'mag', 'angle_deg', 'xy', 'colrow', 'properties')

    struct_name: bytes
    """ Name of the structure being referenced. """

    invert_y: bool
    """ Whether to mirror the pattern (negate y-values / flip across x-axis). Default False. """

    mag: float
    """ Scaling factor (default 1) """

    angle_deg: float
    """ Rotation (degrees counterclockwise) """

    xy: numpy.ndarray
    """
     (For SREF) Location in the parent structure corresponding to the instance's origin (0, 0).
     (For AREF) 3 locations:
                [`offset`,
                 `offset + col_basis_vector * colrow[0]`,
                 `offset + row_basis_vector * colrow[1]`]
                which define the first instance's offset and the array's basis vectors.
        Note that many GDS implementations only support manhattan basis vectors, and some
          assume a certain axis mapping (e.g. x->columns, y->rows) and "reinterpret" the
          basis vectors to match it.
    """

    colrow: Optional[Tuple[int, int]]
    """ Number of columns and rows (AREF) or None (SREF) """

    properties: Dict[int, bytes]
    """ Properties associated with this reference. """

    @classmethod
    def read(cls: Type[R], stream: BinaryIO) -> R:
        invert_y = False
        mag = 1
        angle_deg = 0
        colrow = None
        struct_name = SNAME.skip_and_read(stream)

        size, tag = Record.read_header(stream)
        while tag != XY.tag:
            if tag == STRANS.tag:
                strans = STRANS.read_data(stream, size)
                invert_y = bool(0x8000 & strans)
            elif tag == MAG.tag:
                mag = MAG.read_data(stream, size)[0]
            elif tag == ANGLE.tag:
                angle_deg = ANGLE.read_data(stream, size)[0]
            elif tag == COLROW.tag:
                colrow = COLROW.read_data(stream, size)
            else:
                raise KlamathError(f'Unexpected tag {tag:04x}')
            size, tag = Record.read_header(stream)
        xy = XY.read_data(stream, size).reshape(-1, 2)
        properties = read_properties(stream)
        return cls(struct_name=struct_name, xy=xy, properties=properties, colrow=colrow,
                   invert_y=invert_y, mag=mag, angle_deg=angle_deg)

    def write(self, stream: BinaryIO) -> int:
        b = 0
        if self.colrow is None:
            b += SREF.write(stream, None)
        else:
            b += AREF.write(stream, None)

        b += SNAME.write(stream, self.struct_name)
        if self.angle_deg != 0 or self.mag != 1 or self.invert_y:
            b += STRANS.write(stream, int(self.invert_y) << 15)
            if self.mag != 1:
                b += MAG.write(stream, self.mag)
            if self.angle_deg !=0:
                b += ANGLE.write(stream, self.angle_deg)

        if self.colrow is not None:
            b += COLROW.write(stream, self.colrow)

        b += XY.write(stream, self.xy)
        b += write_properties(stream, self.properties)
        b += ENDEL.write(stream, None)
        return b

    def check(self) -> None:
        if self.colrow is not None:
            if self.xy.size != 6:
                raise KlamathError(f'colrow is not None, so expected size-6 xy. Got {self.xy}')
        else:
            if self.xy.size != 2:
                raise KlamathError(f'Expected size-2 xy. Got {self.xy}')


@dataclass
class Boundary(Element):
    """
    Datastructure representing a Boundary element.
    """
    __slots__ = ('layer', 'xy', 'properties')

    layer: Tuple[int, int]
    """ (layer, data_type) tuple """

    xy: numpy.ndarray
    """ Ordered vertices of the shape. First and last points should be identical. """

    properties: Dict[int, bytes]
    """ Properties for the element. """

    @classmethod
    def read(cls: Type[B], stream: BinaryIO) -> B:
        layer = LAYER.skip_and_read(stream)[0]
        dtype = DATATYPE.read(stream)[0]
        xy = XY.read(stream).reshape(-1, 2)
        properties = read_properties(stream)
        return cls(layer=(layer, dtype), xy=xy, properties=properties)

    def write(self, stream: BinaryIO) -> int:
        b = BOUNDARY.write(stream, None)
        b += LAYER.write(stream, self.layer[0])
        b += DATATYPE.write(stream, self.layer[1])
        b += XY.write(stream, self.xy)
        b += write_properties(stream, self.properties)
        b += ENDEL.write(stream, None)
        return b


@dataclass
class Path(Element):
    """
    Datastructure representing a Path element.

    If `path_type < 4`, `extension` values are not written.
    During read, `exension` defaults to (0, 0) even if unused.
    """
    __slots__ = ('layer', 'xy', 'properties', 'path_type', 'width', 'extension')

    layer: Tuple[int, int]
    """ (layer, data_type) tuple """

    path_type: int
    """ End-cap type (0: flush, 1: circle, 2: square, 4: custom) """

    width: int
    """ Path width """

    extension: Tuple[int, int]
    """ Extension when using path_type=4. Ignored otherwise. """

    xy: numpy.ndarray
    """ Path centerline coordinates """

    properties: Dict[int, bytes]
    """ Properties for the element. """

    @classmethod
    def read(cls: Type[P], stream: BinaryIO) -> P:
        path_type = 0
        width = 0
        bgn_ext = 0
        end_ext = 0
        layer = LAYER.skip_and_read(stream)[0]
        dtype = DATATYPE.read(stream)[0]

        size, tag = Record.read_header(stream)
        while tag != XY.tag:
            if tag == PATHTYPE.tag:
                path_type = PATHTYPE.read_data(stream, size)[0]
            elif tag == WIDTH.tag:
                width = WIDTH.read_data(stream, size)[0]
            elif tag == BGNEXTN.tag:
                bgn_ext = BGNEXTN.read_data(stream, size)[0]
            elif tag == ENDEXTN.tag:
                end_ext = ENDEXTN.read_data(stream, size)[0]
            else:
                raise KlamathError(f'Unexpected tag {tag:04x}')
            size, tag = Record.read_header(stream)
        xy = XY.read_data(stream, size).reshape(-1, 2)
        properties = read_properties(stream)
        return cls(layer=(layer, dtype), xy=xy,
                   properties=properties, extension=(bgn_ext, end_ext),
                   path_type=path_type, width=width)

    def write(self, stream: BinaryIO) -> int:
        b = PATH.write(stream, None)
        b += LAYER.write(stream, self.layer[0])
        b += DATATYPE.write(stream, self.layer[1])
        if self.path_type != 0:
            b += PATHTYPE.write(stream, self.path_type)
        if self.width != 0:
            b += WIDTH.write(stream, self.width)

        if self.path_type < 4:
            bgn_ext, end_ext = self.extension
            if bgn_ext != 0:
                b += BGNEXTN.write(stream, bgn_ext)
            if end_ext != 0:
                b += ENDEXTN.write(stream, end_ext)
        b += XY.write(stream, self.xy)
        b += write_properties(stream, self.properties)
        b += ENDEL.write(stream, None)
        return b


@dataclass
class Box(Element):
    """
    Datastructure representing a Box element. Rarely used.
    """
    __slots__ = ('layer', 'xy', 'properties')

    layer: Tuple[int, int]
    """ (layer, box_type) tuple """

    xy: numpy.ndarray
    """ Box coordinates (5 pairs) """

    properties: Dict[int, bytes]
    """ Properties for the element. """

    @classmethod
    def read(cls: Type[X], stream: BinaryIO) -> X:
        layer = LAYER.skip_and_read(stream)
        dtype = BOXTYPE.read(stream)
        xy = XY.read(stream).reshape(-1, 2)
        properties = read_properties(stream)
        return cls(layer=(layer, dtype), xy=xy, properties=properties)

    def write(self, stream: BinaryIO) -> int:
        b = BOX.write(stream, None)
        b += LAYER.write(stream, self.layer[0])
        b += BOXTYPE.write(stream, self.layer[1])
        b += XY.write(stream, self.xy)
        b += write_properties(stream, self.properties)
        b += ENDEL.write(stream, None)
        return b


@dataclass
class Node(Element):
    """
    Datastructure representing a Node element. Rarely used.
    """
    __slots__ = ('layer', 'xy', 'properties')

    layer: Tuple[int, int]
    """ (layer, node_type) tuple """

    xy: numpy.ndarray
    """ 1-50 pairs of coordinates. """

    properties: Dict[int, bytes]
    """ Properties for the element. """

    @classmethod
    def read(cls: Type[N], stream: BinaryIO) -> N:
        layer = LAYER.skip_and_read(stream)
        dtype = NODETYPE.read(stream)
        xy = XY.read(stream).reshape(-1, 2)
        properties = read_properties(stream)
        return cls(layer=(layer, dtype), xy=xy, properties=properties)

    def write(self, stream: BinaryIO) -> int:
        b = NODE.write(stream, None)
        b += LAYER.write(stream, self.layer[0])
        b += NODETYPE.write(stream, self.layer[1])
        b += XY.write(stream, self.xy)
        b += write_properties(stream, self.properties)
        b += ENDEL.write(stream, None)
        return b


@dataclass
class Text(Element):
    """
    Datastructure representing a Node element. Rarely used.
    """
    __slots__ = ('layer', 'xy', 'properties', 'presentation', 'path_type',
                 'width', 'invert_y', 'mag', 'angle_deg', 'string')

    layer: Tuple[int, int]
    """ (layer, node_type) tuple """

    presentation: int
    """ Bit array. Default all zeros.
        bits 0-1: 00 left/01 center/10 right
        bits 2-3: 00 top/01 middle/10 bottom
        bits 4-5: font number
    """

    path_type: int
    """ Default 0 """

    width: int
    """ Default 0 """

    invert_y: bool
    """ Vertical inversion. Default False. """

    mag: float
    """ Scaling factor. Default 1. """

    angle_deg: float
    """ Rotation (ccw). Default 0. """

    xy: numpy.ndarray
    """ Position (1 pair only) """

    string: bytes
    """ Text content """

    properties: Dict[int, bytes]
    """ Properties for the element. """

    @classmethod
    def read(cls: Type[T], stream: BinaryIO) -> T:
        path_type = 0
        presentation = 0
        invert_y = False
        width = 0
        mag = 1
        angle_deg = 0
        layer = LAYER.skip_and_read(stream)
        dtype = TEXTTYPE.read(stream)

        size, tag = Record.read_header(stream)
        while tag != XY.tag:
            if tag == PRESENTATION.tag:
                presentation = PRESENTATION.read_data(stream, size)
            elif tag == PATHTYPE.tag:
                path_type = PATHTYPE.read_data(stream, size)[0]
            elif tag == WIDTH.tag:
                width = WIDTH.read_data(stream, size)[0]
            elif tag == STRANS.tag:
                strans = STRANS.read_data(stream, size)
                invert_y = bool(0x8000 & strans)
            elif tag == MAG.tag:
                mag = MAG.read_data(stream, size)[0]
            elif tag == ANGLE.tag:
                angle_deg = ANGLE.read_data(stream, size)[0]
            else:
                raise KlamathError(f'Unexpected tag {tag:04x}')
            size, tag = Record.read_header(stream)
        xy = XY.read_data(stream, size).reshape(-1, 2)

        string = STRING.read(stream)
        properties = read_properties(stream)
        return cls(layer=(layer, dtype), xy=xy, properties=properties,
                   string=string, presentation=presentation, path_type=path_type,
                   width=width, invert_y=invert_y, mag=mag, angle_deg=angle_deg)

    def write(self, stream: BinaryIO) -> int:
        b = TEXT.write(stream, None)
        b += LAYER.write(stream, self.layer[0])
        b += TEXTTYPE.write(stream, self.layer[1])
        if self.presentation != 0:
            b += PRESENTATION.write(stream, self.presentation)
        if self.path_type != 0:
            b += PATHTYPE.write(stream, self.path_type)
        if self.width != 0:
            b += WIDTH.write(stream, self.width)
        if self.angle_deg != 0 or self.mag != 1 or self.invert_y:
            b += STRANS.write(stream, int(self.invert_y) << 15)
            if self.mag != 1:
                b += MAG.write(stream, self.mag)
            if self.angle_deg !=0:
                b += ANGLE.write(stream, self.angle_deg)
        b += XY.write(stream, self.xy)
        b += STRING.write(stream, self.string)
        b += write_properties(stream, self.properties)
        b += ENDEL.write(stream, None)
        return b
