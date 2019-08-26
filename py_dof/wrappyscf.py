import numpy as np
from numpy import linalg as la
from gbasis.parsers import parse_gbs, make_contractions
from iodata import load_one, dump_one, IOData
from gbasis.wrappers import from_iodata, from_pyscf

# Evals and integrals
from gbasis.evals.eval import evaluate_basis
from gbasis.evals.eval_deriv import evaluate_deriv_basis
from gbasis.evals.density import evaluate_density
from gbasis.evals.density import evaluate_deriv_density
from gbasis.evals.density import evaluate_density_gradient
from gbasis.evals.density import evaluate_density_laplacian
from gbasis.evals.density import evaluate_density_hessian
from gbasis.evals.stress_tensor import evaluate_stress_tensor
from gbasis.evals.stress_tensor import evaluate_ehrenfest_force
from gbasis.evals.electrostatic_potential import electrostatic_potential

# Integrals
from gbasis.integrals.overlap import overlap_integral
from gbasis.integrals.overlap_asymm import overlap_integral_asymmetric

# From PYSCF
from pyscf import gto, scf, grad, tools, ao2mo
from pyscf.grad import uhf as uhf_grad

# Done with importing stuff
# For my own testing purposes here
import os
import py_dof.orthogonalization as orth
import pprint

pp = pprint.PrettyPrinter(indent=4)


class wrappyscferror(Exception):
    """ Exception class for errors in the wrappyscf module.
    Such error generally have to do with incorrect information being passed.
    Which may have to do with IOData problems or problems in the 
    checkpoint/molden file that is being analyzed. 
    """

    pass


def spin_dm(mol):
    """Generate density matrices from basis set information
    recorded in a molecule object from IOData.

    Parameters
    ----------
    mol
        An IOData molecule object.

    Returns
    -------
    dm_a, dm_b, dm
        Alpha, Beta and total density matrices (2d arrays) in IOdata format.

    """
    c_a = mol.mo.coeffsa
    c_b = mol.mo.coeffsb
    dm_a = c_a * mol.mo.occsa @ c_a.conj().T
    dm_b = c_b * mol.mo.occsb @ c_b.conj().T
    dm = dm_a + dm_b
    return dm_a, dm_b, dm


def project(basis_2, basis_1):
    """Simple function to generate the projection matrix
    between two basis sets. 
 
    Parameters
    ----------
    basis_2, basis_1
        A gbasis basis object.

    Returns
    -------
    coeff_2_proj
        Projection matrix from basis_1 onto basis_2.
    s_2
        Overlap matrix of basis_2.
    s_1
        Overlap matrix of basis_1.

    Raises
    ------
    wrappyscferror 
        If linear dependencies are found in the projection.

    """
    s_1 = overlap_integral(basis_1, coord_type="spherical")
    s_2 = overlap_integral(basis_2, coord_type="spherical")
    s_2_1 = overlap_integral_asymmetric(basis_2, basis_1)
    s_2_inv = orth.power_symmetric(s_2, -1)
    coeff_2_proj = s_2_inv.dot(s_2_1)
    coeff_2_proj = coeff_2_proj[:, np.any(coeff_2_proj, axis=0)]
    olp_proj_proj = coeff_2_proj.T.dot(s_2).dot(coeff_2_proj)
    norm = np.diag(olp_proj_proj) ** (-0.5)
    coeff_2_proj *= norm
    rank = np.linalg.matrix_rank(coeff_2_proj)
    if rank < coeff_2_proj.shape[-1]:
        raise wrappyscferror("Linearly dependent basis set projection. WARNING!")
    assert np.allclose(
        coeff_2_proj @ s_1 @ np.linalg.inv(coeff_2_proj), s_2, rtol=1e-8, atol=1e-6
    )
    return coeff_2_proj, s_2, s_1


def topyscf(mol, verb_lvl=0):
    """Wrapper to pass an IOdata molecule to a PySCF Mole object.
    The goal is to pass all the information, including 
    everything gbasis can understand, 
    to a format that PySCF can use.

    Parameters
    ----------
    mol
        IOData molecule object,
    verb_lvl
        Verbosity level integer flag.

    Returns
    -------
    mole
        PySCF Mole object with the atom list and coordinates
        from mol.
    one_dm
        One-particle density matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array or list for a,b 2d arrays)
    coeffs
        AO coefficient matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array or list for a,b 2d arrays)
    occs
        Occupation numbers formatted in a PySCF-compatible way 
        (single 1d array or list for a,b 1d arrays)
    energies
        MO energies formatted in a PySCF-compatible way 
        (single 1d array or list for a,b 1d arrays)

    Raises
    ------
    wrappyscferror 
        If something other than an IOData mol object is passed.
        If the basis set is not understood (unavailable in PySCF)

    """
    if not isinstance(mol, IOData):
        raise wrappyscferror("Something other than an IOData mol object passed.")
    mole = gto.Mole()
    if mol.nelec:
        mole.spin = int(mol.nelec) % 2
    else:
        print(
            "No number of electrons was explicitely recorded in the input file. WARNING!"
        )
    if mol.charge:
        mole.charge = mol.charge
    else:
        print("No charge was explicitely recorded in the input file. WARNING!")
    coords = []
    coords = np.asarray(
        [
            [mol.atnums[i], mol.atcoords[i] * 0.52917721092]
            for i in range(mol.atnums.size)
        ]
    )
    mole.atom = coords
    if mol.obasis_name:
        mole.basis = mol.obasis_name
    else:
        print("No basis set name was recorded. Using default cc-pvtz. WARNING!")
        mole.basis = "cc-pvtz"
    try:
        mole.build()
    except:
        raise wrappyscf(
            "The basis set name could not be understood by PySCF. Use a different basis set or set it manually."
        )
    one_dm, coeffs, occs, energies = get_one_dm(mol, mole, verb_lvl)
    return mole, one_dm, coeffs, occs, energies


def get_one_dm(mol, mole, verb_lvl=0):
    """Function to pass the one-particle density matrix
    from a molecule IOdata object
    to the format that PySCF uses. 
    Should work for restricted and unrestricted 
    The MO information is used to generate the density matrix.

    Parameters
    ----------
    mol
        IOdata molecule object.
    mole
        PySCF mole object with the same or different basis set. 
    verb_lvl
        Verbosity level integer flag.

    Returns
    -------
    one_dm_2
        One-particle density matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array or list for a,b 2d arrays)
    coeffs
        AO coefficient matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array or list for a,b 2d arrays)
    occs
        Occupation numbers formatted in a PySCF-compatible way 
        (single 1d array or list for a,b 1d arrays)
    energies
        MO energies formatted in a PySCF-compatible way 
        (single 1d array or list for a,b 1d arrays)

    Raises
    ------
    wrappyscferror 
        If the mol.mo information is not understood. Can be set manually by modifying the
        mol.mo attribute in the mol object passed from IOData.

    """
    basis_1 = from_iodata(mol)
    basis_2 = from_pyscf(mole)
    proj, s_2, s_1 = project(basis_2, basis_1)
    try:
        if mol.mo[0] == "unrestricted":
            one_dm_a_1, one_dm_b_1, one_dm_1 = spin_dm(mol)
            one_dm_a_2 = proj @ one_dm_a_1 @ np.linalg.inv(proj)
            if verb_lvl > 0:
                print(
                    "Total number of alpha electrons is {0}".format(
                        checkrdm(one_dm_a_2, s_2)
                    )
                )
            one_dm_b_2 = proj @ one_dm_b_1 @ np.linalg.inv(proj)
            if verb_lvl > 0:
                print(
                    "Total number of beta  electrons is {0}".format(
                        checkrdm(one_dm_b_2, s_2)
                    )
                )
            one_dm_2 = one_dm_a_2 + one_dm_b_2
            if verb_lvl > 0:
                print(
                    "Total number of electrons is {0}".format(checkrdm(one_dm_2, s_2))
                )
            if np.allclose(one_dm_a_2, one_dm_b_2):
                if verb_lvl > 1:
                    print(
                        "This will be treated a restricted wave function, alpha and beta orbitals are the same."
                    )
                coeffs = proj @ mol.mo.coeffsa
                energies = np.asarray(mol.mo.energies[: mol.mo.norba])
                occs = np.asarray(np.around(mol.mo.occsa) * 2)
                return one_dm_2, coeffs, occs, energies
            else:
                if verb_lvl > 2:
                    print("This is a true unrestricted wave function.")
                energies_a = np.asarray(mol.mo.energies[: mol.mo.norba])
                energies_b = np.asarray(mol.mo.energies[mol.mo.norbb :])
                occs_a = np.asarray(np.around(mol.mo.occsa))
                occs_b = np.asarray(np.around(mol.mo.occsb))
                coeffs_a = proj @ mol.mo.coeffsa
                coeffs_b = proj @ mol.mo.coeffsb
                return (
                    np.array((one_dm_a_2, one_dm_b_2)),
                    np.array((coeffs_a, coeffs_b)),
                    np.array((occs_a, occs_b)),
                    np.array((energies_a, energies_b)),
                )
        if mol.mo[0] == "restricted":
            one_dm_1 = spin_dm(mol)[2]
            one_dm_2 = proj @ one_dm_1 @ np.linalg.inv(proj)
            coeffs = np.asarray(proj @ mol.mo.coeffsa)
            occs = np.asarray(np.around(mol.mo.occsa) * 2)
            energies = np.asarray(mol.mo.energies)
            if verb_lvl > 2:
                print("This is a restricted wave function.")
            if verb_lvl > 0:
                print(
                    "Total number of electrons is {0}".format(checkrdm(one_dm_2, s_2))
                )
            return one_dm_2, coeffs, occs, energies
    except:
        raise wrappyscferror(
            "Could not understand if the passed data is restricted or unrestricted. You may set it manually."
        )


def tellmem(a, name):
    print(
        "Lets check the size of the array {0}: {1} elements, {2} size.".format(
            name, a.size, a.shape
        )
    )
    pp.pprint(a)


def checkrdm(a: np.ndarray, b: np.ndarray):
    """Check that your density matrix makes sense
    by calculating the number of electrons.
   
    Parameters
    ----------
    a
        Density matrix in the AO basis.
        shape=(nbasis, nbasis)
    b
        Overlap matrix of the AO basis.
        shape=(nbasis, nbasis)

    Returns
    -------
    tr 
        Number of electrons (rounded). A check is in place to verify
        that it is an integer number (or very close to one)

    Raises
    ------
    wrappyscferror 
        If the one-particle density matrix does not contain an integer number of electrons.
        This might be caused by a wrong overlap matrix too.

    """
    c = a @ b
    tr = np.trace(c)
    try:
        assert np.isclose(np.abs(round(tr, 2) - tr), 1e-08)
    except:
        print("{0} is quite different from {1}!".format(tr, round(tr, 2)))
        raise wrappyscferror(
            "This density matrix does not seem to contain an integer number of electrons."
        )
    return round(tr, 6)


def calc_energy(mole, one_dm: np.ndarray, mf):
    """Calculate the energy according to a created PySCF mean field object.
    To do so, calculates all the integrals using PySCF functions.
    No SCF iterations are done.
      
    Parameters
    ----------
    mole
        A PySCF mole object.
    one_dm
        The PySCF formatted one particle density matrix.
    mf
        A PySCF mean field object. Its only use is method specification.

    Returns
    -------
    etot
        The total energy in atomic units.
    enuc
        The nuclear repulsion energy in a.u.
    e1e
        The 1-electron part of the total energy in a.u.
    e2e
        The 2-electron part of the total energy in a.u.
 
    """
    h1e = mf.get_hcore(mole)
    pot = mf.get_veff(mole, one_dm)
    ee, e2e = mf.energy_elec(one_dm)
    e1e = ee - e2e
    enuc = mf.energy_nuc()
    etot = e1e.real + e2e.real + enuc.real
    return etot, enuc, e1e, e2e


def solver(mole, one_dm, mf, verb_lvl):
    """Wrapper for a solver (PySCF mean field object smf) 
    used to evaluate the SCF solution at every geometry and
    return the MO energies. Could be replaced for any other approach.
    In fact, its not  very pretty as it is.

    Parameters
    ----------
    mole
        A PySCF mole object.
    one_dm
        One-particle density matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array) Used as initial guess for the SCF.
    mf
        A PySCF mean field object. Its only use is method specification.
    verb_lvl
        Verbosity level integer flag.

    Returns
    -------
    energies
        MO energies formatted in a PySCF-compatible way 
        (single 1d array)

    """
    mole.build()
    cmf = mf.__class__
    nmf = cmf(mole)
    try:
        nmf.xc = mf.xc
        nmf.grids.level = mf.grids.level
        if verb_lvl > 4:
            print("RKS solver built.")
    except:
        if verb_lvl > 4:
            print("RHF solver built.")
    nmf.verbose = verb_lvl
    nmf.max_cycle = 20
    nmf.conv_check = True
    nmf.run(one_dm)
    if verb_lvl > 3:
        print(
            "SCF converged: {0} , with total energy:{1} a.u.".format(
                nmf.converged, nmf.e_tot
            )
        )
    energies = nmf.mo_energy
    return energies


def calc_gradients_nuc(mole):
    """Calculate nuclear part of nuclear gradients from nuclear positions.

    Parameters
    ----------
    mole
        A PySCF mole object, which should contain the geometry information.

    Returns
    -------
    grad_nuc
        Nuclear gradients in a.u. per cartesian coordinate per atom.
        (Due to nuclear forces)

    """
    grad_nuc = np.zeros((mole.natm, 3))
    for j in range(mole.natm):
        q2 = mole.atom_charge(j)
        r2 = mole.atom_coord(j)
        for i in range(mole.natm):
            if i != j:
                q1 = mole.atom_charge(i)
                r1 = mole.atom_coord(i)
                r = np.sqrt(np.dot(r1 - r2, r1 - r2))
                grad_nuc[j] -= q1 * q2 * (r2 - r1) / r ** 3
    return grad_nuc


def calc_gradients_elec(mole, one_dm: np.ndarray, mf, coeffs, occs, energies):
    """Calculate electronic part of nuclear gradients.

    Parameters
    ----------
    mole
        A PySCF mole object.
    one_dm
        One-particle density matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array or list for a,b 2d arrays)
    mf
        A PySCF mean field object. Its only use is method specification.
    coeffs
        AO coefficient matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array or list for a,b 2d arrays)
    occs
        Occupation numbers formatted in a PySCF-compatible way 
        (single 1d array or list for a,b 1d arrays)
    energies
        MO energies formatted in a PySCF-compatible way 
        (single 1d array or list for a,b 1d arrays)


    Returns
    -------
    grad_elec
        Nuclear gradients in a.u. per cartesian coordinate per atom.
        (Due to electrons)

 """
    mo_energy = energies
    mo_coeff = coeffs
    mo_occ = occs
    if len(one_dm) == 2:
        try:
            grad_elec = grad.UKS(mf).grad_elec(
                mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ
            )
        except:
            grad_elec = grad.UHF(mf).grad_elec(
                mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ
            )
    else:
        try:
            grad_elec = grad.RKS(mf).grad_elec(
                mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ
            )
        except:
            grad_elec = grad.RHF(mf).grad_elec(
                mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ
            )
    return grad_elec


def calc_gradients(mole, one_dm: np.ndarray, mf, coeffs, occs, energies):
    """Calculate nuclear gradients according to a created PySCF mean field object.
    To do so, calculates all the integral derivatives using PySCF functions, 
    but no SCF iterations are done.
      
    Parameters
    ----------
    mole
        A PySCF mole object.
    one_dm
        One-particle density matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array or list for a,b 2d arrays)
    mf
        A PySCF mean field object. Its only use is method specification.
    coeffs
        AO coefficient matrix understood by IOdata 
        projected and formatted in a PySCF-compatible way 
        (single 2d array or list for a,b 2d arrays)
    occs
        Occupation numbers formatted in a PySCF-compatible way 
        (single 1d array or list for a,b 1d arrays)
    energies
        MO energies formatted in a PySCF-compatible way 
        (single 1d array or list for a,b 1d arrays)

    Returns
    -------
    grad_tot
        Nuclear gradients in a.u. per cartesian coordinate per atom.
 
    """
    grad_elec = calc_gradients_elec(mole, one_dm, mf, coeffs, occs, energies)
    grad_nuc = calc_gradients_nuc(mole)
    grad_tot = grad_elec + grad_nuc
    return grad_tot
