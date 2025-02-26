from py_dof import *


def test_gradients_fchk():
    mol = read_one("water_opt.fchk")
    mole, one_dm, coeffs, occs, energies = topyscf(mol, verb_lvl=3)
    mf = scf.RHF(mole)
    grads = calc_gradients(mole, one_dm, mf, coeffs, occs, energies)
    zeros = np.zeros_like(grads)
    assert np.allclose(grads, zeros, rtol=1e-4, atol=1e-6, equal_nan=False)
