# klamath README

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


### Alternatives
- [gdspy](https://github.com/heitzmann/gdspy)
    * Provides abstractions and methods for working with design data
        outside of the I/O process (e.g. polygon clipping).
    * Requires compilation (C++) to build from source.
    * Focused on high-level API
- [python-gdsii](https://pypi.org/project/python-gdsii)
    * Pure-python implementation. Can easily be altered to use `numpy`
        for speed, but is limited by object allocation overhead.
    * Focused on high-level API


### Links
- [Source repository](https://mpxd.net/code/jan/klamath)
- [PyPI](https://pypi.org/project/klamath)


## Installation

Requirements:
* python >= 3.7 (written and tested with 3.8)
* numpy


Install with pip:
```bash
pip3 install klamath
```

Alternatively, install from git
```bash
pip3 install git+https://mpxd.net/code/jan/klamath.git@release
```

## Examples
### Low-level

Filter which polygons are read based on layer:

```python3
import io
import klamath
from klamath import records
from klamath.record import Record

def read_polygons(stream, filter_layer_tuple=(4, 5)):
    """
    Given a stream positioned at the start of a record,
     return the vertices of all BOUNDARY records which match
     the provided `filter_layer_tuple`, up to the next
     ENDSTR record.
    """
    polys = []
    while True:
        size, tag = Record.read_header(stream)
        stream.seek(size, io.SEEK_CUR)      # skip to next header

        if tag == records.ENDEL.tag:
            break                           # If ENDEL, we are done

        if tag != records.BOUNDARY.tag:
            continue                        # Skip until we find a BOUNDARY

        layer = records.LAYER.skip_and_read(stream)[0]  # skip to LAYER
        dtype = records.DATATYPE.read(stream)[0]

        if (layer, dtype) != filter_layer_tuple:
            continue                        # Skip reading XY unless layer matches

        xy = XY.read(stream).reshape(-1, 2)
        polys.append(xy)
    return polys
```

### High-level

Write an example GDS file:

```python3
import klamath
from klamath.elements import Boundary, Text, Path, Reference

stream = open('example.gds', 'wb')

header = klamath.library.FileHeader(
                name=b'example',
                meters_per_db_unit=1e-9,      # 1 nm DB unit
                user_units_per_db_unit=1e-3)  # 1 um (1000nm) display unit
header.write(stream)

elements_A = [
    Boundary(layer=(4, 18),
             xy=[[0, 0], [10, 0], [10, 20], [0, 20], [0, 0]],
             properties={1: b'prop1string', 2: b'some other string'}),
    Text(layer=(5, 5),
         xy=[[5, 10]],
         string=b'center position',
         properties={},        # Remaining args are set to default values
         presentation=0,       #   and will be omitted when writing
         angle_deg=0,
         invert_y=False,
         width=0,
         path_type=0,
         mag=1),
    Path(layer=(4, 20),
         xy=[[0, 0], [10, 10], [0, 20]],
         path_type=0,
         width=0,
         extension=(0, 0),     # ignored since path_type=0
         properties={}),
    ]
klamath.library.write_struct(stream, name=b'my_struct', elements=elements_A)

elements_top = [
    Reference(struct_name=b'my_struct',
              xy=[[30, 30]],
              colrow=None,   # not an array
              angle_deg=0,
              invert_y=True,
              mag=1.5,
              properties={}),
    Reference(struct_name=b'my_struct',
              colrow=(3, 2),                    # 3x2 array at (0, 50)
              xy=[[0, 50], [60, 50], [30, 50]], #   with basis vectors
              angle_deg=30,                     #   [20, 0] and [0, 30]
              invert_y=False,
              mag=1,
              properties={}),
    ]
klamath.library.write_struct(stream, name=b'top', elements=elements_top)

klamath.records.ENDLIB.write(stream, None)
stream.close()
```

Read back the file:

```python3
import klamath

stream = open('example.gds', 'rb')
header = klamath.library.FileHeader.read(stream)

structs = {}

struct = klamath.library.try_read_struct(stream)
while struct is not None:
    name, elements = struct
    structs[name] = elements
    struct = klamath.library.try_read_struct(stream)

stream.close()

print(structs)
```

Read back a single struct by name:

```python3
import klamath

stream = open('example.gds', 'rb')

header = klamath.library.FileHeader.read(stream)
struct_positions = klamath.library.scan_structs(stream)

stream.seek(struct_positions[b'my_struct'])
elements_A = klamath.library.try_read_struct(stream)

stream.close()

print(elements_A)
```
