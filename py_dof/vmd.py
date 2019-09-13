import os
import numpy as np


def _vmd_script_start():
    """Generate part of the beginning part of the VMD script.
    Returns
    -------
    VMD script responsible for beginning of VMD script
    """
    return (
        "#!/usr/local/bin/vmd\n"
        "# VMD version: 1.8.6\n"
        "#\n"
        "# Display settings\n"
        "display projection Perspective\n"
        "display nearclip set 0.000000\n"
        "display shadow off\n"
        "color Element {C} silver\n"
        "color Element {Cl} green\n"
        "axes location Off\n"
        "color Display Background white\n"
        "light 2 on\n"
        "light 3 on\n"
        "#\n"
    )


def _vmd_script_molecule(mole, filename="molecule.xyz"):
    """Generate part of the VMD script that loads the molecule information.
    Parameters
    ----------
    mole :
        PySCF molecule object. Will be translated to xyz format,
        and written, so that it can be used.
    filename :
        Filename for the xyz file that will be used as intermediary.
    Returns
    -------
    Part of the VMD script that constructs the molecule

    Raises
    ------
    ValueError
        If no coordinates are given.
    TypeError
        If not fed a molecule object file. 
    """
    output = "# load new molecule\n"
    if len(mole.atom) == 0:
        raise ValueError("Need at least one molecule file with coordinates.")
    atoms = mole.atom
    natoms = len(mole.atom[0:, 0])
    f = open(filename, "w")
    f.write(str(natoms) + "\n\n")
    for i in range(0, natoms):
        symb = str(atoms[i, 0])
        coord = " ".join(map(str, atoms[i, 1].tolist()))
        f.write(symb + " " + coord + "\n")
    f.close()
    output += (
        "mol {0} {1} type {2} first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all"
        "\n".format("new", filename, "{xyz}")
    )
    output += "#\n" "# representation of the atoms\n"
    output += "mol representation CPK 1.000000 0.300000 118.000000 131.000000\n"
    output += (
        "mol delrep 0 top\n"
        "mol color Element\n"
        "mol selection {{all}}\n"
        "mol material Opaque\n"
        "mol addrep top\n"
        "#\n"
    )
    return output


def _vmd_script_vectors(mole, grads, colorid, scalex):
    """Generate part of the VMD script that prints gradients as vectors
    for the visualization of your gradients on top of your molecule.

    Parameters
    ----------
    mole :
        PySCF molecule object. Will be translated to xyz format,
        and written, so that it can be used.
    grads :
        Gradients in the PySCF format, either nucleear gradients or
        gradients per one MO.
    colorid :
        Color ID in VMD for the vector arrows. 
    scalex :
        Multiplication factor for the vectors. 
    Returns
    -------
    Part of the VMD script that prints the vectors.
   """
    if not isinstance(grads, np.ndarray):
        raise TypeError("Grads should be a numpy ndarray object.")
    natoms = len(mole.atom[0:, 0])
    origins = mole.atom[0:, 1]
    output = "# create arrow function\n"
    output += "proc vmd_draw_arrow {mol start end} {\n"
    output += "   set middle [vecadd $start [vecscale 0.9 [ vecsub $end $start]]]\n"
    output += "   graphics $mol cylinder $start $middle radius 0.05 resolution 75 filled yes\n"
    output += "   graphics $mol cone $middle $end radius 0.075 resolution 75\n"
    output += "}\n"
    output += "graphics 0 color {0}\n".format(colorid)
    output += "#create arrows for each atom\n"
    for i in range(0, natoms):
        origin = np.around(origins[i], 4)
        end = np.around(np.add(origin, scalex * np.asarray(grads[i, :])), 4)
        output += "draw arrow {{{0} {1} {2}}} {{{3} {4} {5}}}\n".format(
            origin[0], origin[1], origin[2], end[0], end[1], end[2]
        )
    return output


def gen_vmd_script(
    mole,
    grads,
    filename="molecule.xyz",
    scriptname="molecule.vmd",
    colorid=1,
    scalex=50,
):
    """Generates a VMD (Visual Molecular Dynamics) script
    for the visualization of your gradients on top of your molecule.

    Parameters
    ----------
    mole :
        PySCF molecule object. Will be translated to xyz format,
        and written, so that it can be used.
    grads :
        Gradients in the PySCF format, either nucleear gradients or
        gradients per one MO.
    filename :
        Filename for the xyz file that will be used as intermediary.
    scriptname :
        Name of the vmd script file.
    colorid :
        Color ID in VMD for the vector arrows. 1 is blue (default)
    scalex :
        Multiplication factor for the vectors. Default is 50.
    """
    if colorid > 15:
        colorid = 0
    output = _vmd_script_start()
    output += _vmd_script_molecule(mole, filename)
    output += _vmd_script_vectors(mole, grads, colorid, scalex)
    with open(scriptname, "w") as f:
        f.write(output)
