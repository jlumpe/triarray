"""Setuptools installation script for triarray package."""

import os
from setuptools import setup, find_packages
from distutils.util import convert_path


# Directory of script
root_dir = os.path.dirname(__file__)


# Get package version without importing it
version_ns = dict()
with open(convert_path('triarray/version.py')) as fobj:
	exec(fobj.read(), version_ns)
version = version_ns['__version__']


# Dynamic download URL based off current version - git tag should match
download_url = (
	'https://github.com/jlumpe/triarray/archive/{}.tar.gz'
	.format(version)
)


# Read readme file for long description
with open(os.path.join(root_dir, 'README.md')) as fobj:
	long_description = fobj.read()


setup(
	name='triarray',
	version=version,
	description='Tools for working with symmetric matrices in non-redundant format.',
	long_description=long_description,
	author='Jared Lumpe',
	url='https://github.com/jlumpe/triarray',
	download_url=download_url,
	license='MIT',
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Developers',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 3.6',
		'Topic :: Utilities',
		'Topic :: Scientific/Engineering',
		'Topic :: Scientific/Engineering :: Mathematics',
	],
	keywords='numpy array matrix symmetric pairwise distance similarity',
	packages=find_packages(),
	install_requires=[
		'numpy>=1.11',
		'numba>=0.30',
	],
)
