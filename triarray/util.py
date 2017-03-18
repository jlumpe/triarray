"""Misc utility code."""


def in_notebook():
	"""Check if code is being executed in a Jupyter notebook.

	:rtype: bool
	"""
	try:
		from IPython import get_ipython
	except ImportError:
		return False

	ipy = get_ipython()
	if not ipy:
		return False

	return type(ipy).__name__ == 'ZMQInteractiveShell'


def get_tqdm():
	"""Get the appropriate tqdm() function for the current execution environment.

	If running in a notebook, imports and returns :func:`tqdm.tqdm_notebook`,
	otherwise returns :func:`tqdm.tqdm`.
	"""

	if in_notebook():
		from tqdm import tqdm_notebook as tqdm
	else:
		from tqdm import tqdm

	return tqdm
