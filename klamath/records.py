"""
Record type and tag definitions
"""
from typing import Sequence

from .record import NoDataRecord, BitArrayRecord, Int2Record, Int4Record, Real8Record
from .record import ASCIIRecord, DateTimeRecord


class HEADER(Int2Record):
    tag = 0x0002
    expected_size = 2


class BGNLIB(DateTimeRecord):
    tag = 0x0102
    expected_size = 6 * 2


class LIBNAME(ASCIIRecord):
    tag = 0x0206


class UNITS(Real8Record):
    """ (user_units_per_db_unit, db_units_per_meter) """
    tag = 0x0305
    expected_size = 8 * 2


class ENDLIB(NoDataRecord):
    tag = 0x0400


class BGNSTR(DateTimeRecord):
    tag = 0x0502
    expected_size = 6 * 2


class STRNAME(ASCIIRecord):
    """ Legal characters are `?A-Za-z0-9_$` """
    tag = 0x0606


class ENDSTR(NoDataRecord):
    tag = 0x0700


class BOUNDARY(NoDataRecord):
    tag = 0x0800


class PATH(NoDataRecord):
    tag = 0x0900


class SREF(NoDataRecord):
    tag = 0x0a00


class AREF(NoDataRecord):
    tag = 0x0b00


class TEXT(NoDataRecord):
    tag = 0x0c00


class LAYER(Int2Record):
    tag = 0x0d02
    expected_size = 2


class DATATYPE(Int2Record):
    tag = 0x0e02
    expected_size = 2


class WIDTH(Int4Record):
    tag = 0x0f03
    expected_size = 4


class XY(Int4Record):
    tag = 0x1003


class ENDEL(NoDataRecord):
    tag = 0x1100


class SNAME(ASCIIRecord):
    tag = 0x1206


class COLROW(Int2Record):
    tag = 0x1302
    expected_size = 4


class NODE(NoDataRecord):
    tag = 0x1500


class TEXTTYPE(Int2Record):
    tag = 0x1602
    expected_size = 2


class PRESENTATION(BitArrayRecord):
    tag = 0x1701


class SPACING(Int2Record):
    tag = 0x1802        #Not sure about 02; Unused


class STRING(ASCIIRecord):
    tag = 0x1906


class STRANS(BitArrayRecord):
    tag = 0x1a01


class MAG(Real8Record):
    tag = 0x1b05
    expected_size = 8


class ANGLE(Real8Record):
    tag = 0x1c05
    expected_size = 8


class UINTEGER(Int2Record):
    tag = 0x1d02    #Unused; not sure about 02


class USTRING(ASCIIRecord):
    tag = 0x1e06      #Unused; not sure about 06


class REFLIBS(ASCIIRecord):
    tag = 0x1f06

    @classmethod
    def check_size(cls, size: int):
        if size != 0 and size % 44 != 0:
            raise Exception(f'Expected size to be multiple of 44, got {size}')


class FONTS(ASCIIRecord):
    tag = 0x2006

    @classmethod
    def check_size(cls, size: int):
        if size != 0 and size % 44 != 0:
            raise Exception(f'Expected size to be multiple of 44, got {size}')


class PATHTYPE(Int2Record):
    tag = 0x2102
    expected_size = 2


class GENERATIONS(Int2Record):
    tag = 0x2202
    expected_size = 2

    @classmethod
    def check_data(cls, data: Sequence[int]):
        if len(data) != 1:
            raise Exception(f'Expected exactly one integer, got {data}')


class ATTRTABLE(ASCIIRecord):
    tag = 0x2306

    @classmethod
    def check_size(cls, size: int):
        if size > 44:
            raise Exception(f'Expected size <= 44, got {size}')


class STYPTABLE(ASCIIRecord):
    tag = 0x2406        #UNUSED, not sure about 06


class STRTYPE(Int2Record):
    tag = 0x2502        #UNUSED


class ELFLAGS(BitArrayRecord):
    tag = 0x2601


class ELKEY(Int2Record):
    tag = 0x2703        # UNUSED


class LINKTYPE(Int2Record):
    tag = 0x2803        # UNUSED


class LINKKEYS(Int2Record):
    tag = 0x2903        # UNUSED


class NODETYPE(Int2Record):
    tag = 0x2a02
    expected_size = 2


class PROPATTR(Int2Record):
    tag = 0x2b02
    expected_size = 2


class PROPVALUE(ASCIIRecord):
    tag = 0x2c06
    expected_size = 2


class BOX(NoDataRecord):
    tag = 0x2d00


class BOXTYPE(Int2Record):
    tag = 0x2e02
    expected_size = 2


class PLEX(Int4Record):
    tag = 0x2f03
    expected_size = 4


class BGNEXTN(Int4Record):
    tag = 0x3003


class ENDEXTN(Int4Record):
    tag = 0x3103


class TAPENUM(Int2Record):
    tag = 0x3202
    expected_size = 2


class TAPECODE(Int2Record):
    tag = 0x3302
    expected_size = 12


class STRCLASS(Int2Record):
    tag = 0x3401        # UNUSED


class RESERVED(Int2Record):
    tag = 0x3503        # UNUSED


class FORMAT(Int2Record):
    tag = 0x3602
    expected_size = 2

    @classmethod
    def check_data(cls, data: Sequence[int]):
        if len(data) != 1:
            raise Exception(f'Expected exactly one integer, got {data}')


class MASK(ASCIIRecord):
    """ List of layers and dtypes """
    tag = 0x3706


class ENDMASKS(NoDataRecord):
    """ End of MASKS records """
    tag = 0x3800


class LIBDIRSIZE(Int2Record):
    tag = 0x3902


class SRFNAME(ASCIIRecord):
    tag = 0x3a06


class LIBSECUR(Int2Record):
    tag = 0x3b02


class BORDER(NoDataRecord):
    tag = 0x3c00


class SOFTFENCE(NoDataRecord):
    tag = 0x3d00


class HARDFENCE(NoDataRecord):
    tag = 0x3f00


class SOFTWIRE(NoDataRecord):
    tag = 0x3f00


class HARDWIRE(NoDataRecord):
    tag = 0x4000


class PATHPORT(NoDataRecord):
    tag = 0x4100


class NODEPORT(NoDataRecord):
    tag = 0x4200


class USERCONSTRAINT(NoDataRecord):
    tag = 0x4300


class SPACERERROR(NoDataRecord):
    tag = 0x4400


class CONTACT(NoDataRecord):
    tag = 0x4500
