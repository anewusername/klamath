import struct

import pytest       # type: ignore
import numpy        # type: ignore
from numpy.testing import assert_array_equal        # type: ignore

from .basic import parse_bitarray, parse_int2, parse_int4, parse_real8, parse_ascii
from .basic import pack_bitarray, pack_int2, pack_int4, pack_real8, pack_ascii
from .basic import decode_real8, encode_real8

from .basic import KlamathError


def test_parse_bitarray():
    assert(parse_bitarray(b'59') == 13625)
    assert(parse_bitarray(b'\0\0') == 0)
    assert(parse_bitarray(b'\xff\xff') == 65535)

    # 4 bytes (too long)
    with pytest.raises(KlamathError):
        parse_bitarray(b'4321')

    # empty data
    with pytest.raises(KlamathError):
        parse_bitarray(b'')


def test_parse_int2():
    assert_array_equal(parse_int2(b'59\xff\xff\0\0'), (13625, -1, 0))

    # odd length
    with pytest.raises(KlamathError):
        parse_int2(b'54321')

    # empty data
    with pytest.raises(KlamathError):
        parse_int2(b'')


def test_parse_int4():
    assert_array_equal(parse_int4(b'4321'), (875770417,))

    # length % 4 != 0
    with pytest.raises(KlamathError):
        parse_int4(b'654321')

    # empty data
    with pytest.raises(KlamathError):
        parse_int4(b'')


def test_decode_real8():
    # zeroes
    assert(decode_real8(numpy.array([0x0])) == 0)
    assert(decode_real8(numpy.array([1<<63])) == 0) # negative
    assert(decode_real8(numpy.array([0xff << 56])) == 0) # denormalized

    assert(decode_real8(numpy.array([0x4110 << 48])) == 1.0)
    assert(decode_real8(numpy.array([0xC120 << 48])) == -2.0)


def test_parse_real8():
    packed = struct.pack('>3Q', 0x0, 0x4110_0000_0000_0000, 0xC120_0000_0000_0000)
    assert_array_equal(parse_real8(packed), (0.0, 1.0, -2.0))

    # length % 8 != 0
    with pytest.raises(KlamathError):
        parse_real8(b'0987654321')

    # empty data
    with pytest.raises(KlamathError):
        parse_real8(b'')


def test_parse_ascii():
    # empty data
    with pytest.raises(KlamathError):
        parse_ascii(b'')

    assert(parse_ascii(b'12345') == b'12345')
    assert(parse_ascii(b'12345\0') == b'12345') # strips trailing null byte


def test_pack_bitarray():
    packed = pack_bitarray(321)
    assert(len(packed) == 2)
    assert(packed == struct.pack('>H', 321))


def test_pack_int2():
    packed = pack_int2((3, 2, 1))
    assert(len(packed) == 3*2)
    assert(packed == struct.pack('>3h', 3, 2, 1))
    assert(pack_int2([-3, 2, -1]) == struct.pack('>3h', -3, 2, -1))


def test_pack_int4():
    packed = pack_int4((3, 2, 1))
    assert(len(packed) == 3*4)
    assert(packed == struct.pack('>3l', 3, 2, 1))
    assert(pack_int4([-3, 2, -1]) == struct.pack('>3l', -3, 2, -1))


def test_encode_real8():
    assert(encode_real8(numpy.array([0.0])) == 0)
    arr = numpy.array((1.0, -2.0, 1e-9, 1e-3, 1e-12))
    assert_array_equal(decode_real8(encode_real8(arr)), arr)


def test_pack_real8():
    reals = (0, 1, -1, 0.5, 1e-9, 1e-3, 1e-12)
    packed = pack_real8(reals)
    assert(len(packed) == len(reals) * 8)
    assert_array_equal(parse_real8(packed), reals)


def test_pack_ascii():
    assert(pack_ascii(b'4321') == b'4321')
    assert(pack_ascii(b'321') == b'321\0')
