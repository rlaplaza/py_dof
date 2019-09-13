from py_dof import *
import numpy as np


def test_compare_moegradients():
    mol = read_one("water_dz.molden")
    mol.obasis_name = "aug-cc-pvdz"
    mole, one_dm, coeffs, occs, energies = topyscf(mol, verb_lvl=3)
    mf = scf.RHF(mole)

    # Reference gradients using the self-consistent formula
    grads1 = calc_moe_gradients_np(mole, one_dm, mf, occs, energies, imaxstep=0.00001)
    grads2 = calc_moe_gradients_i6p(mole, one_dm, mf, occs, energies)
    np.testing.assert_allclose(
        np.asarray(grads1), np.asarray(grads2), rtol=1e-4, atol=1e-6
    )

    # We try the self consistent formula with an excessively large step
    grads1 = calc_moe_gradients_np(mole, one_dm, mf, occs, energies, imaxstep=0.001)
    np.testing.assert_allclose(
        np.asarray(grads1), np.asarray(grads2), rtol=1e-4, atol=1e-6
    )
