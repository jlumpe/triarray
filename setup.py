"""Setuptools installation script for triarray package."""

from setuptools import setup, find_packages
from distutils.util import convert_path


# Get package version without importing it
version_ns = dict()
with open(convert_path('triarray/version.py')) as fobj:
	exec(fobj.read(), version_ns)
version = version_ns['__version__']


setup(
	name='triarray',
	version=version,
	description=(
		'Tools for working with symmetric matrices in non-redundant format.',
	),
	author='Jared Lumpe',
	license='MIT',
	packages=find_packages(),
	install_requires=[
		'numpy>=1.11',
		'numba>=0.30',
	],
)
