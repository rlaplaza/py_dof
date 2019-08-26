import numpy as np
import os
from pprint import pprint
from iodata import load_one, dump_one, IOData


class readererror(Exception):
    """ Exception class for errors in the reader module.
    """

    pass


def read_one(filename: str, verb=0):
    """Very simple wrapper for iodata load_one.

    Parameters
    ----------
    filename
        A string that contains the path to an input file.
    verb
        Verbosity level integer flag.

    Returns
    -------
    mol
        An IOdata molecule object.

    Raises
    ------
    readereerror
        If the file is not found or is not a file, or
        does not contain the basis set information needed
        to calculate the one-particle density matrix etc.

    """
    path = os.path.abspath(filename)
    try:
        assert os.path.exists(path)
        assert os.path.isfile(path)
    except:
        raise readererror("Could not load the file {0}.".format(path))
    mol = load_one(path)
    try:
        mol.mo.coeffsa
    except:
        raise readererror(
            "Basis set coefficients were not understood or are not present."
        )
    if verb > 1:
        print("File loaded using IOData.")
    if verb > 2:
        pprint(mol)
    return mol
