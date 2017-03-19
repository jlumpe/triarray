# triarray

Python package for working with symmetric matrices in non-redundant format.

See full documentation [here](http://triarray.readthedocs.io/en/latest/).


## Overview

The `triarray` package contains tools for working with large symmetric matrices while only storing the elements in the upper or lower triangle, thus halving memory requirements.

`triarray` uses [Numba](http://numba.pydata.org/)'s just-in-time compilation to generate high-performance C code that works with any data type and is easily extendable (including within a Jupyter notebook).


## Requirements

* Numpy 1.11 or above
* Numba 0.30 or above


## Installation

The easiest way is to use pip:

    pip install triarray
    
or you can clone the repository and run the setup script:

    cd path/to/triarray
    python setup.py install


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
