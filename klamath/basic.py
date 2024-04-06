"""
Functionality for encoding/decoding basic datatypes
"""
from typing import Sequence, IO
import struct
import logging
from datetime import datetime

import numpy
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class KlamathError(Exception):
    pass


#
# Parse functions
#
def parse_bitarray(data: bytes) -> int:
    if len(data) != 2:
        raise KlamathError(f'Incorrect bitarray size ({len(data)}). Data is {data!r}.')
    (val,) = struct.unpack('>H', data)
    return val


def parse_int2(data: bytes) -> NDArray[numpy.int16]:
    data_len = len(data)
    if data_len == 0 or (data_len % 2) != 0:
        raise KlamathError(f'Incorrect int2 size ({len(data)}). Data is {data!r}.')
    return numpy.frombuffer(data, dtype='>i2', count=data_len // 2)


def parse_int4(data: bytes) -> NDArray[numpy.int32]:
    data_len = len(data)
    if data_len == 0 or (data_len % 4) != 0:
        raise KlamathError(f'Incorrect int4 size ({len(data)}). Data is {data!r}.')
    return numpy.frombuffer(data, dtype='>i4', count=data_len // 4)


def decode_real8(nums: NDArray[numpy.uint64]) -> NDArray[numpy.float64]:
    """ Convert GDS REAL8 data to IEEE float64. """
    nums = nums.astype(numpy.uint64)
    neg = nums & 0x8000_0000_0000_0000
    exp = (nums >> 56) & 0x7f
    mant = (nums & 0x00ff_ffff_ffff_ffff).astype(numpy.float64)
    mant[neg != 0] *= -1
    return numpy.ldexp(mant, 4 * (exp - 64) - 56, signature=(float, int, float))


def parse_real8(data: bytes) -> NDArray[numpy.float64]:
    data_len = len(data)
    if data_len == 0 or (data_len % 8) != 0:
        raise KlamathError(f'Incorrect real8 size ({len(data)}). Data is {data!r}.')
    ints = numpy.frombuffer(data, dtype='>u8', count=data_len // 8)
    return decode_real8(ints)


def parse_ascii(data: bytes) -> bytes:
    if len(data) == 0:
        return b''
    if data[-1:] == b'\0':
        return data[:-1]
    return data


def parse_datetime(data: bytes) -> list[datetime]:
    """ Parse date/time data (12 byte blocks) """
    if len(data) == 0 or len(data) % 12 != 0:
        raise KlamathError(f'Incorrect datetime size ({len(data)}). Data is {data!r}.')
    dts = []
    for ii in range(0, len(data), 12):
        year, *date_parts = parse_int2(data[ii:ii + 12])
        try:
            dt = datetime(year + 1900, *date_parts)
        except ValueError as err:
            dt = datetime(1900, 1, 1, 0, 0, 0)
            logger.warning(f'Invalid date {[year] + date_parts}, setting {dt} instead')
        dts.append(dt)
    return dts


#
# Pack functions
#
def pack_bitarray(data: int) -> bytes:
    if data > 65535 or data < 0:
        raise KlamathError(f'bitarray data out of range: {data}')
    return struct.pack('>H', data)


def pack_int2(data: Sequence[int]) -> bytes:
    arr = numpy.array(data)
    if (arr > 32767).any() or (arr < -32768).any():
        raise KlamathError(f'int2 data out of range: {arr}')
    return arr.astype('>i2').tobytes()


def pack_int4(data: Sequence[int]) -> bytes:
    arr = numpy.array(data)
    if (arr > 2147483647).any() or (arr < -2147483648).any():
        raise KlamathError(f'int4 data out of range: {arr}')
    return arr.astype('>i4').tobytes()


def encode_real8(fnums: NDArray[numpy.float64]) -> NDArray[numpy.uint64]:
    """ Convert from float64 to GDS REAL8 representation. """
    # Split the ieee float bitfields
    ieee = numpy.atleast_1d(fnums.astype(numpy.float64).view(numpy.uint64))
    sign = ieee & numpy.uint64(0x8000_0000_0000_0000)
    ieee_exp = (ieee >> numpy.uint64(52)).astype(numpy.int32) & numpy.int32(0x7ff)
    ieee_mant = ieee & numpy.uint64(0xf_ffff_ffff_ffff)

    subnorm = (ieee_exp == 0) & (ieee_mant != 0)
    zero = (ieee_exp == 0) & (ieee_mant == 0)

    # IEEE normal double is (1 + ieee_mant / 2^52) * 2^(ieee_exp - 1023)
    # IEEE subnormal double is (ieee_mant / 2^52) * 2^(-1022)
    # GDS real8 is (gds_mant / 2^(7*8)) * 16^(gds_exp - 64)
    #            = (gds_mant / 2^56) * 2^(4 * gds_exp - 256)

    # Convert exponent.
    exp2 = ieee_exp + 1 - 1023   # +1 is due to mantissa (1.xxxx in IEEE vs 0.xxxxx in GDSII)
    exp2[subnorm] = -1022
    exp16, rest = numpy.divmod(exp2, 4)

    # Compensate for exponent coarseness
    comp = (rest != 0)
    exp16[comp] += 1

    shift = rest.copy().astype(numpy.int8)
    shift[comp] = 4 - rest[comp]
    shift -= 3      # account for gds bit position

    # add leading one
    gds_mant_unshifted = ieee_mant.copy()
    gds_mant_unshifted[~subnorm] += 0x10_0000_0000_0000

    rshift = (shift > 0)
    gds_mant = numpy.empty_like(ieee_mant)
    gds_mant[~rshift] = gds_mant_unshifted[~rshift] << (-shift[~rshift]).astype(numpy.uint16)
    gds_mant[ rshift] = gds_mant_unshifted[ rshift] >> ( shift[ rshift]).astype(numpy.uint16)

    # add gds exponent bias
    gds_exp = exp16 + 64

    neg_biased = (gds_exp < 0)
    gds_mant[neg_biased] >>= (gds_exp[neg_biased] * 4).astype(numpy.uint16)
    gds_exp[neg_biased] = 0

    too_big = (gds_exp > 0x7f) & ~(zero | subnorm)
    if too_big.any():
        raise KlamathError(f'Number(s) too big for real8 format: {fnums[too_big]}')

    gds_exp_bits = gds_exp.astype(numpy.uint64) << 56

    real8 = sign | gds_exp_bits | gds_mant
    real8[zero] = 0
    real8[gds_exp < -14] = 0  # number is too small

    return real8.astype(numpy.uint64, copy=False)


def pack_real8(data: Sequence[float]) -> bytes:
    return encode_real8(numpy.array(data)).astype('>u8').tobytes()


def pack_ascii(data: bytes) -> bytes:
    size = len(data)
    if size % 2 != 0:
        return data + b'\0'
    return data


def pack_datetime(data: Sequence[datetime]) -> bytes:
    """ Pack date/time data (12 byte blocks) """
    parts = sum(((d.year - 1900, d.month, d.day, d.hour, d.minute, d.second)
                 for d in data), start=())
    return pack_int2(parts)


def read(stream: IO[bytes], size: int) -> bytes:
    """ Read and check for failure """
    data = stream.read(size)
    if len(data) != size:
        raise EOFError
    return data
