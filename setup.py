#!/usr/bin/env python3

from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    long_description = f.read()

with open('klamath/VERSION', 'r') as f:
    version = f.read().strip()

setup(name='klamath',
      version=version,
      description='GDSII format reader/writer',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/code/jan/klamath',
      packages=find_packages(),
      package_data={
          'klamath': ['VERSION',
                     'py.typed',
                     ]
      },
      install_requires=[
            'numpy',
      ],
      classifiers=[
            'Programming Language :: Python :: 3',
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Information Technology',
            'Intended Audience :: Manufacturing',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)',
      ],
      keywords=[
          'layout',
          'design',
          'CAD',
          'EDA',
          'electronics',
          'photonics',
          'IC',
          'mask',
          'pattern',
          'drawing',
          'lithography',
          'litho',
          'geometry',
          'geometric',
          'polygon',
          'gds',
          'gdsii',
          'gds2',
          'stream',
          'vector',
          'freeform',
          'manhattan',
          'angle',
          'Calma',
      ],
      )
