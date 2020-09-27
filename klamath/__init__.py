"""
`klamath` is a Python module for reading and writing to the GDSII file format.

The goal is to keep this library simple:
- Map data types directly wherever possible.
    * Presents an accurate representation of what is saved to the file.
    * Avoids excess copies / allocations for speed.
    * No "automatic" error checking, except when casting datatypes.
        If data integrity checks are provided at all, they must be
        explicitly run by the caller.
- Low-level functionality is first-class.
    * Meant for use-cases where the caller wants to read or write
         individual GDS records.
    * Offers complete control over the written file.
- Opinionated and limited high-level functionality.
    * Discards or ignores rarely-encountered data types.
    * Keeps functions simple and reusable.
    * Only de/encodes the file format, doesn't provide tools to modify
        the data itself.
    * Still requires explicit values for most fields.
- No compilation
    * Uses `numpy` for speed, since it's commonly available / pre-built.
    * Building this library should not require a compiler.

`klamath` was built to provide a fast and versatile GDS interface for
 [masque](https://mpxd.net/code/jan/masque), which provides higher-level
 tools for working with hierarchical design data and supports multiple
 file formats.
"""
import pathlib

from . import basic
from . import record
from . import records
from . import elements
from . import library


__author__ = 'Jan Petykiewicz'

with open(pathlib.Path(__file__).parent / 'VERSION', 'r') as f:
    __version__ = f.read().strip()
