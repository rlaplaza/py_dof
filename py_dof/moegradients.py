import numpy as np
from py_dof.wrappyscf import solver as runscf


def calc_moe_gradients_4p(
    mole, one_dm: np.ndarray, mf, occs, energies, maxstep=0.00001, verbose=0
):
    """Calculate gradients of MO energies with respect to nuclear displacement.
    Currently uses an 4-point stencil finite-difference approach.

    Parameters
    ----------
    mole
        A PySCF mole object.
    one_dm
        One-particle density matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array)
    mf
        A PySCF mean field object. Its only use is method specification.
    occs
        Occupation numbers formatted in a PySCF-compatible way 
        (single 1d array)
    energies
        MO energies formatted in a PySCF-compatible way 
        (single 1d array)
    maxstep
        Maximum distance (Bohr) that each nuclei will be displaced
        in every cartesian dimension. A small value of 0.0001-0.00001 is recommended.
    verb_lvl
        Verbosity level integer flag.

    Returns
    -------
    grad_moe
        Differences in the MO energies (final-beginning)
        with respect to the displacement.

    Raises
    ------
    moegerror 
        If an unrestricted molecule is input.

    """
    if len(one_dm) == 2:
        raise moegerror(
            "MO energy gradients are only available for restricted systems."
        )
    grad_moe = [None] * mole.natm
    h = maxstep / 2
    for i in range(0, mole.natm):
        grad_moe[i] = np.zeros((energies.size, 3))  # This is a massive object
        if verbose > 1:
            print("Atom being displaced: {0}".format(mole.atom[i]))
        if verbose > 1:
            print("Max step size: {0}".format(maxstep))
        for j in range(0, 3):
            if verbose > 2:
                print("Original coordinate: {0}".format(mole.atom[i][1][j]))
            mole_copy = mole.copy()
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] + 2 * h
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_pp = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] + h
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_p = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] - h
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_m = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] - 2 * h
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_mm = runscf(mole_copy, one_dm, mf, verbose)
            grad_moe[i][:, j] = (
                energies_mm + 8 * energies_p - energies_pp - 8 * energies_m
            ) / (12 * h)
    return grad_moe


def calc_moe_gradients_2p(
    mole, one_dm: np.ndarray, mf, occs, energies, maxstep=0.000005, verbose=0
):
    """Calculate gradients of MO energies with respect to nuclear displacement.
    Currently uses an 2-point stencil finite-difference approach.

    Parameters
    ----------
    mole
        A PySCF mole object.
    one_dm
        One-particle density matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array)
    mf
        A PySCF mean field object. Its only use is method specification.
    occs
        Occupation numbers formatted in a PySCF-compatible way 
        (single 1d array)
    energies
        MO energies formatted in a PySCF-compatible way 
        (single 1d array)
    maxstep
        Maximum distance (Bohr) that each nuclei will be displaced
        in every cartesian dimension. A small value of 0.0001-0.00001 is recommended.
    verb_lvl
        Verbosity level integer flag.

    Returns
    -------
    grad_moe
        Differences in the MO energies (final-beginning)
        with respect to the displacement.

    Raises
    ------
    moegerror 
        If an unrestricted molecule is input.
        If the system is an atom and hence has no possible internal coordinates.

    """
    if len(one_dm) == 2:
        raise moegerror("MO energy gradients are only available for restricted wfns.")
    grad_moe = [None] * mole.natm
    h = maxstep
    for i in range(0, mole.natm):
        grad_moe[i] = np.zeros((energies.size, 3))  # This is a massive object
        if verbose > 1:
            print("Atom being displaced: {0}".format(mole.atom[i]))
        if verbose > 1:
            print("Max step size: {0}".format(maxstep))
        for j in range(0, 3):
            if verbose > 2:
                print("Original coordinate: {0}".format(mole.atom[i][1][j]))
            mole_copy = mole.copy()
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] + h
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_p = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] - h
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_m = runscf(mole_copy, one_dm, mf, verbose)
            grad_moe[i][:, j] = (energies_p - energies_m) / (2 * h)
    return grad_moe


def internal_coordinates(mole):
    zmat_atoms = cart2zmat(mole.atom)
    if len(zmat_atoms) > 2:
        nics = (len(zmat_atoms) * 3) - 6
    elif len(zmat_atoms) == 2:
        nics = 1
    elif len(zmat_atoms) == 1:
        nics = 0
        raise moegerror(
            "There are no possible derivatives with respect to nuclear coordinates, this system is an atom."
        )
    cart_atoms = zmat2cart(zmat_atoms)
    mole.atom = cart_atoms
    return nics


def cart2zmat(atoms):
    symb = atoms[0:, 0]
    j = 0
    coord = atoms[:, 1]
    zmat = []
    zmat.append([symb[j], 1])
    j += 1
    if len(coord) > 1:
        r1 = coord[1] - coord[0]
        nr1 = np.linalg.norm(r1)
        zmat.append([symb[j], 1, nr1])
        j += 1
    if len(coord) > 2:
        r2 = coord[2] - coord[0]
        nr2 = np.linalg.norm(r2)
        a = np.arccos(np.dot(r1, r2) / (nr1 * nr2))
        zmat.append([symb[j], 1, nr2, 2, a * 180 / np.pi])
        j += 1
    if len(coord) > 3:
        o0, o1, o2 = coord[:3]
        p0, p1, p2 = 1, 2, 3
        for k, c in enumerate(coord[3:]):
            r0 = c - o0
            nr0 = np.linalg.norm(r0)
            r1 = o1 - o0
            nr1 = np.linalg.norm(r1)
            a1 = np.arccos(np.dot(r0, r1) / (nr0 * nr1))
            b0 = np.cross(r0, r1)
            nb0 = np.linalg.norm(b0)

            if abs(nb0) < 1e-7:  # o0, o1, c in line
                a2 = 0
                zmat.append([symb[j], p0, nr0, p1, a1 * 180 / np.pi, p2, a2])
                j += 1
            else:
                b1 = np.cross(o2 - o0, r1)
                nb1 = np.linalg.norm(b1)

                if abs(nb1) < 1e-7:  # o0 o1 o2 in line
                    a2 = 0
                    zmat.append([symb[j], p0, nr0, p1, a1 * 180 / np.pi, p2, a2])
                    j += 1
                    o2 = c
                    p2 = 4 + k
                else:
                    if np.dot(np.cross(b1, b0), r1) < 0:
                        a2 = np.arccos(np.dot(b1, b0) / (nb0 * nb1))
                    else:
                        a2 = -np.arccos(np.dot(b1, b0) / (nb0 * nb1))
                    zmat.append(
                        [symb[j], p0, nr0, p1, a1 * 180 / np.pi, p2, a2 * 180 / np.pi]
                    )
                    j += 1

    return zmat


def zmat2cart(zmat):
    from pyscf.symm import rotation_mat

    symb = []
    coord = []
    min_items_per_line = 1
    for i in range(len(zmat)):
        rawd = zmat[i]
        print(rawd)
        if rawd and rawd[0] != "#":
            assert len(rawd) >= min_items_per_line
            symb.append(rawd[0])
            if len(rawd) < 3:
                coord.append(np.zeros(3))
                min_items_per_line = 3
            elif len(rawd) == 3:
                coord.append(np.array((float(rawd[2]), 0, 0)))
                min_items_per_line = 5
            elif len(rawd) == 5:
                bonda = int(rawd[1]) - 1
                bond = float(rawd[2])
                anga = int(rawd[3]) - 1
                ang = float(rawd[4]) / 180 * np.pi
                assert ang >= 0
                v1 = coord[anga] - coord[bonda]
                if not np.allclose(v1[:2], 0):
                    vecn = np.cross(v1, np.array((0.0, 0.0, 1.0)))
                else:  # on z
                    vecn = np.array((0.0, 0.0, 1.0))
                rmat = rotation_mat(vecn, ang)
                c = np.dot(rmat, v1) * (bond / np.linalg.norm(v1))
                coord.append(coord[bonda] + c)
                min_items_per_line = 7
            else:
                bonda = int(rawd[1]) - 1
                bond = float(rawd[2])
                anga = int(rawd[3]) - 1
                ang = float(rawd[4]) / 180 * np.pi
                assert ang >= 0 and ang <= np.pi
                v1 = coord[anga] - coord[bonda]
                v1 /= np.linalg.norm(v1)
                if ang < 1e-7:
                    c = v1 * bond
                elif np.pi - ang < 1e-7:
                    c = -v1 * bond
                else:
                    diha = int(rawd[5]) - 1
                    dih = float(rawd[6]) / 180 * np.pi
                    v2 = coord[diha] - coord[anga]
                    vecn = np.cross(v2, -v1)
                    vecn_norm = np.linalg.norm(vecn)
                    if vecn_norm < 1e-7:
                        if not np.allclose(v1[:2], 0):
                            vecn = np.cross(v1, np.array((0.0, 0.0, 1.0)))
                        else:  # on z
                            vecn = np.array((0.0, 0.0, 1.0))
                        rmat = rotation_mat(vecn, ang)
                        c = np.dot(rmat, v1) * bond
                    else:
                        rmat = rotation_mat(v1, -dih)
                        vecn = np.dot(rmat, vecn) / vecn_norm
                        rmat = rotation_mat(vecn, ang)
                        c = np.dot(rmat, v1) * bond
                coord.append(coord[bonda] + c)
    cart_atoms = list(zip(symb, coord))
    return cart_atoms


def calc_moe_gradients_i4p(
    mole, one_dm: np.ndarray, mf, occs, energies, maxstep=0.00001, verbose=0
):
    """Calculate gradients of MO energies with respect to nuclear displacement.
    Currently uses an improved 4-point stencil finite-difference approach.

    Parameters
    ----------
    mole
        A PySCF mole object.
    one_dm
        One-particle density matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array)
    mf
        A PySCF mean field object. Its only use is method specification.
    occs
        Occupation numbers formatted in a PySCF-compatible way 
        (single 1d array)
    energies
        MO energies formatted in a PySCF-compatible way 
        (single 1d array)
    maxstep
        Maximum distance (Bohr) that each nuclei will be displaced
        in every cartesian dimension. A small value of 0.0001-0.00001 is recommended.
    verb_lvl
        Verbosity level integer flag.

    Returns
    -------
    grad_moe
        Differences in the MO energies (final-beginning)
        with respect to the displacement.

    Raises
    ------
    moegerror 
        If an unrestricted molecule is input.

    """
    if len(one_dm) == 2:
        raise moegerror(
            "MO energy gradients are only available for restricted systems."
        )
    grad_moe = [None] * mole.natm
    t = maxstep / 3
    for i in range(0, mole.natm):
        grad_moe[i] = np.zeros((energies.size, 3))  # This is a massive object
        if verbose > 1:
            print("Atom being displaced: {0}".format(mole.atom[i]))
        if verbose > 1:
            print("Max step size: {0}".format(maxstep))
        for j in range(0, 3):
            if verbose > 2:
                print("Original coordinate: {0}".format(mole.atom[i][1][j]))
            mole_copy = mole.copy()
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] + 3 * t
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_pp = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] + t
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_p = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] - t
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_m = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] - 3 * t
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_mm = runscf(mole_copy, one_dm, mf, verbose)
            grad_moe[i][:, j] = 9 * (energies_p - energies_m) / (16 * t) - (
                energies_pp - energies_mm
            ) / (48 * t)
    return grad_moe


def calc_moe_gradients_i6p(
    mole, one_dm: np.ndarray, mf, occs, energies, maxstep=0.00001, verbose=0
):
    """Calculate gradients of MO energies with respect to nuclear displacement.
    Currently uses an improved 6-point stencil finite-difference approach.
    This is obviously extremely expensive, but also extremely robust.

    Parameters
    ----------
    mole
        A PySCF mole object.
    one_dm
        One-particle density matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array)
    mf
        A PySCF mean field object. Its only use is method specification.
    occs
        Occupation numbers formatted in a PySCF-compatible way 
        (single 1d array)
    energies
        MO energies formatted in a PySCF-compatible way 
        (single 1d array)
    maxstep
        Maximum distance (Bohr) that each nuclei will be displaced
        in every cartesian dimension. A small value of 0.0001-0.00001 is recommended.
    verb_lvl
        Verbosity level integer flag.

    Returns
    -------
    grad_moe
        Differences in the MO energies (final-beginning)
        with respect to the displacement.

    Raises
    ------
    moegerror 
        If an unrestricted molecule is input.

    """
    if len(one_dm) == 2:
        raise moegerror(
            "MO energy gradients are only available for restricted systems."
        )
    grad_moe = [None] * mole.natm
    t = maxstep / 5
    for i in range(0, mole.natm):
        grad_moe[i] = np.zeros((energies.size, 3))  # This is a massive object
        if verbose > 1:
            print("Atom being displaced: {0}".format(mole.atom[i]))
        if verbose > 1:
            print("Max step size: {0}".format(maxstep))
        for j in range(0, 3):
            if verbose > 2:
                print("Original coordinate: {0}".format(mole.atom[i][1][j]))
            mole_copy = mole.copy()
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] + 5 * t
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_ppp = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] + 3 * t
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_pp = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] + t
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_p = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] - t
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_m = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] - 3 * t
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_mm = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] - 5 * t
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_mmm = runscf(mole_copy, one_dm, mf, verbose)
            grad_moe[i][:, j] = (
                75 * (energies_p - energies_m) / (64 * 2 * t)
                - 25 * (energies_pp - energies_mm) / (384 * 2 * t)
                + 3 * (energies_ppp - energies_mmm) / (640 * 2 * t)
            )
    return grad_moe


def calc_moe_gradients_6p(
    mole, one_dm: np.ndarray, mf, occs, energies, maxstep=0.00001, verbose=0
):
    """Calculate gradients of MO energies with respect to nuclear displacement.
    Currently uses a simple 6-point stencil finite-difference approach.
    This is obviously extremely expensive, but also extremely robust.

    Parameters
    ----------
    mole
        A PySCF mole object.
    one_dm
        One-particle density matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array)
    mf
        A PySCF mean field object. Its only use is method specification.
    occs
        Occupation numbers formatted in a PySCF-compatible way 
        (single 1d array)
    energies
        MO energies formatted in a PySCF-compatible way 
        (single 1d array)
    maxstep
        Maximum distance (Bohr) that each nuclei will be displaced
        in every cartesian dimension. A small value of 0.0001-0.00001 is recommended.
    verb_lvl
        Verbosity level integer flag.

    Returns
    -------
    grad_moe
        Differences in the MO energies (final-beginning)
        with respect to the displacement.

    Raises
    ------
    moegerror 
        If an unrestricted molecule is input.

    """
    if len(one_dm) == 2:
        raise moegerror(
            "MO energy gradients are only available for restricted systems."
        )
    grad_moe = [None] * mole.natm
    h = maxstep / 3
    for i in range(0, mole.natm):
        grad_moe[i] = np.zeros((energies.size, 3))  # This is a massive object
        if verbose > 1:
            print("Atom being displaced: {0}".format(mole.atom[i]))
        if verbose > 1:
            print("Max step size: {0}".format(maxstep))
        for j in range(0, 3):
            if verbose > 2:
                print("Original coordinate: {0}".format(mole.atom[i][1][j]))
            mole_copy = mole.copy()
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] + 3 * h
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_ppp = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] + 2 * h
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_pp = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] + h
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_p = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] - h
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_m = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] - 2 * h
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_mm = runscf(mole_copy, one_dm, mf, verbose)
            mole_copy.atom[i][1][j] = mole.atom[i][1][j] - 3 * h
            if verbose > 2:
                print("New coordinate: {0}".format(mole_copy.atom[i][1][j]))
            energies_mmm = runscf(mole_copy, one_dm, mf, verbose)
            grad_moe[i][:, j] = (
                1 * (energies_p - energies_m) / (60 * t)
                - 3 * (energies_pp - energies_mm) / (20 * h)
                + 3 * (energies_ppp - energies_mmm) / (4 * h)
            )
    return grad_moe


def calc_moe_gradients_np(
    mole, one_dm: np.ndarray, mf, occs, energies, imaxstep=0.00001, verbose=0
):
    """Calculate gradients of MO energies with respect to nuclear displacement.
    Will start with an initial maxtep and the low cost 2-point formula,
    then check against higher order formulae until convergence is reached;
    if unsuccessful the maxstep variable will be decreased and the show
    will go on. The reduction will be by two orders of magnitude max.
    This is obviously extremely expensive, but basically should always converge
    to the appropiate solution in molecular systems.
    Could be made much more efficient by cacheing some of the displacements.

    Parameters
    ----------
    mole
        A PySCF mole object.
    one_dm
        One-particle density matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array)
    mf
        A PySCF mean field object. Its only use is method specification.
    occs
        Occupation numbers formatted in a PySCF-compatible way 
        (single 1d array)
    energies
        MO energies formatted in a PySCF-compatible way 
        (single 1d array)
    imaxstep
        Maximum distance (Bohr) that each nuclei will be displaced
        in every cartesian dimension in the first iteration. 
        A small value of 0.0001-0.00001 is recommended.
    verbose
        Verbosity level integer flag.

    Returns
    -------
    grad_moe
        Differences in the MO energies (final-beginning)
        with respect to the displacement.

    Raises
    ------
    moegerror 
        If an unrestricted molecule is input.
        If convergence is never reached within the allocated reductions.

    """
    if len(one_dm) == 2:
        raise moegerror(
            "MO energy gradients are only available for restricted systems."
        )
    grads1 = [None] * mole.natm
    grads2 = [None] * mole.natm
    for i in range(0, mole.natm):
        grads1[i] = np.zeros((energies.size, 3))  # These are massive objects
        grads2[i] = np.zeros((energies.size, 3))
    decr = [1, 5, 10, 20, 100]
    for i in decr:
        step = imaxstep / float(i)
        grads1 = calc_moe_gradients_2p(
            mole, one_dm, mf, occs, energies, maxstep=step, verbose=verbose
        )
        if np.allclose(np.asarray(grads1), np.asarray(grads2), rtol=1e-4, atol=1e-6):
            return grads1
        grads2 = calc_moe_gradients_4p(
            mole, one_dm, mf, occs, energies, maxstep=step, verbose=verbose
        )
        if np.allclose(np.asarray(grads1), np.asarray(grads2), rtol=1e-4, atol=1e-6):
            return grads2
        else:
            grads1 = calc_moe_gradients_i4p(
                mole, one_dm, mf, occs, energies, maxstep=step, verbose=verbose
            )
            if np.allclose(
                np.asarray(grads1), np.asarray(grads2), rtol=1e-4, atol=1e-6
            ):
                return grads1
            else:
                grads2 = calc_moe_gradients_i6p(
                    mole, one_dm, mf, occs, energies, maxstep=step, verbose=verbose
                )
                if np.allclose(
                    np.asarray(grads1), np.asarray(grads2), rtol=1e-4, atol=1e-6
                ):
                    return grads2
        if verbose > 2:
            print("Initial step size decreased.")
    raise moegerror("MO energy gradients could not be converged.")
