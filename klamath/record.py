"""
Generic record-level read/write functionality.
"""
from typing import IO, ClassVar, Self, Generic, TypeVar
from collections.abc import Sequence
import struct
import io
from datetime import datetime
from abc import ABCMeta, abstractmethod

import numpy
from numpy.typing import NDArray

from .basic import KlamathError
from .basic import parse_int2, parse_int4, parse_real8, parse_datetime, parse_bitarray
from .basic import pack_int2, pack_int4, pack_real8, pack_datetime, pack_bitarray
from .basic import parse_ascii, pack_ascii, read


_RECORD_HEADER_FMT = struct.Struct('>HH')
II = TypeVar('II')  # Input type
OO = TypeVar('OO')  # Output type


def write_record_header(stream: IO[bytes], data_size: int, tag: int) -> int:
    record_size = data_size + 4
    if record_size > 0xFFFF:
        raise KlamathError(f'Record size is too big: {record_size}')
    header = _RECORD_HEADER_FMT.pack(record_size, tag)
    return stream.write(header)


def read_record_header(stream: IO[bytes]) -> tuple[int, int]:
    """
    Read a record's header (size and tag).
    Args:
        stream: stream to read from
    Returns:
        data_size: size of data (not including header)
        tag: Record type tag
    """
    header = read(stream, 4)
    record_size, tag = _RECORD_HEADER_FMT.unpack(header)
    if record_size < 4:
        raise KlamathError(f'Record size is too small: {record_size} @ pos 0x{stream.tell():x}')
    if record_size % 2:
        raise KlamathError(f'Record size is odd: {record_size} @ pos 0x{stream.tell():x}')
    data_size = record_size - 4     # substract header size
    return data_size, tag


def expect_record(stream: IO[bytes], tag: int) -> int:
    data_size, actual_tag = read_record_header(stream)
    if tag != actual_tag:
        raise KlamathError(f'Unexpected record! Got tag 0x{actual_tag:04x}, expected 0x{tag:04x}')
    return data_size


class Record(Generic[II, OO], metaclass=ABCMeta):
    tag: ClassVar[int] = -1
    expected_size: ClassVar[int | None] = None

    @classmethod
    def check_size(cls: type[Self], size: int) -> None:
        if cls.expected_size is not None and size != cls.expected_size:
            raise KlamathError(f'Expected size {cls.expected_size}, got {size}')

    @classmethod
    def check_data(cls: type[Self], data: II) -> None:
        pass

    @classmethod
    @abstractmethod
    def read_data(cls: type[Self], stream: IO[bytes], size: int) -> OO:
        pass

    @classmethod
    @abstractmethod
    def pack_data(cls: type[Self], data: II) -> bytes:
        pass

    @staticmethod
    def read_header(stream: IO[bytes]) -> tuple[int, int]:
        return read_record_header(stream)

    @classmethod
    def write_header(cls: type[Self], stream: IO[bytes], data_size: int) -> int:
        return write_record_header(stream, data_size, cls.tag)

    @classmethod
    def skip_past(cls: type[Self], stream: IO[bytes]) -> bool:
        """
        Skip to the end of the next occurence of this record.

        Args:
            stream: Seekable stream to read from.

        Return:
            True if the record was encountered and skipped.
            False if the end of the library was reached.
        """
        from .records import ENDLIB
        size, tag = Record.read_header(stream)
        while tag != cls.tag:
            stream.seek(size, io.SEEK_CUR)
            if tag == ENDLIB.tag:
                return False
            size, tag = Record.read_header(stream)
        stream.seek(size, io.SEEK_CUR)
        return True

    @classmethod
    def skip_and_read(cls: type[Self], stream: IO[bytes]) -> OO:
        size, tag = Record.read_header(stream)
        while tag != cls.tag:
            stream.seek(size, io.SEEK_CUR)
            size, tag = Record.read_header(stream)
        data = cls.read_data(stream, size)
        return data

    @classmethod
    def read(cls: type[Self], stream: IO[bytes]) -> OO:
        size = expect_record(stream, cls.tag)
        data = cls.read_data(stream, size)
        return data

    @classmethod
    def write(cls: type[Self], stream: IO[bytes], data: II) -> int:
        data_bytes = cls.pack_data(data)
        b = cls.write_header(stream, len(data_bytes))
        b += stream.write(data_bytes)
        return b


class NoDataRecord(Record[None, None]):
    expected_size: ClassVar[int | None] = 0

    @classmethod
    def read_data(cls: type[Self], stream: IO[bytes], size: int) -> None:
        stream.read(size)

    @classmethod
    def pack_data(cls: type[Self], data: None) -> bytes:
        if data is not None:
            raise KlamathError('?? Packing {data} into NoDataRecord??')
        return b''


class BitArrayRecord(Record[int, int]):
    expected_size: ClassVar[int | None] = 2

    @classmethod
    def read_data(cls: type[Self], stream: IO[bytes], size: int) -> int:        # noqa: ARG003  size unused
        return parse_bitarray(read(stream, 2))

    @classmethod
    def pack_data(cls: type[Self], data: int) -> bytes:
        return pack_bitarray(data)


class Int2Record(Record[NDArray[numpy.integer] | Sequence[int] | int, NDArray[numpy.int16]]):
    @classmethod
    def read_data(cls: type[Self], stream: IO[bytes], size: int) -> NDArray[numpy.int16]:
        return parse_int2(read(stream, size))

    @classmethod
    def pack_data(cls: type[Self], data: NDArray[numpy.integer] | Sequence[int] | int) -> bytes:
        return pack_int2(data)


class Int4Record(Record[NDArray[numpy.integer] | Sequence[int] | int, NDArray[numpy.int32]]):
    @classmethod
    def read_data(cls: type[Self], stream: IO[bytes], size: int) -> NDArray[numpy.int32]:
        return parse_int4(read(stream, size))

    @classmethod
    def pack_data(cls: type[Self], data: NDArray[numpy.integer] | Sequence[int] | int) -> bytes:
        return pack_int4(data)


class Real8Record(Record[Sequence[float] | float, NDArray[numpy.float64]]):
    @classmethod
    def read_data(cls: type[Self], stream: IO[bytes], size: int) -> NDArray[numpy.float64]:
        return parse_real8(read(stream, size))

    @classmethod
    def pack_data(cls: type[Self], data: Sequence[float] | float) -> bytes:
        return pack_real8(data)


class ASCIIRecord(Record[bytes, bytes]):
    @classmethod
    def read_data(cls: type[Self], stream: IO[bytes], size: int) -> bytes:
        return parse_ascii(read(stream, size))

    @classmethod
    def pack_data(cls: type[Self], data: bytes) -> bytes:
        return pack_ascii(data)


class DateTimeRecord(Record[Sequence[datetime], list[datetime]]):
    @classmethod
    def read_data(cls: type[Self], stream: IO[bytes], size: int) -> list[datetime]:
        return parse_datetime(read(stream, size))

    @classmethod
    def pack_data(cls: type[Self], data: Sequence[datetime]) -> bytes:
        return pack_datetime(data)
