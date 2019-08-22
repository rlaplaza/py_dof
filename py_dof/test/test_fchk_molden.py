from py_dof import *
import numpy as np
mol = read_one('water_opt.fchk')
mole, one_dm, coeffs, occs, energies = topyscf(mol,verb_lvl=3)
mf = scf.RHF(mole)
etot1,enuc1,e1e1,e2e1 = calc_energy(mole,one_dm,mf)

mol = read_one('water_opt.molden')
mol.obasis_name = 'aug-cc-pvqz'
mole, one_dm, coeffs, occs, energies = topyscf(mol,verb_lvl=3)
mf = scf.RHF(mole)
etot2,enuc2,e1e2,e2e2 = calc_energy(mole,one_dm,mf)

assert(np.isclose(etot1,etot2))
assert(np.isclose(enuc1,enuc2))
assert(np.isclose(e1e1,e1e2))
assert(np.isclose(e2e1,e2e2))

