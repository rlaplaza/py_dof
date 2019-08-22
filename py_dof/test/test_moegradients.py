from py_dof import *
import numpy as np
mol = read_one('water_opt.molden')
mol.obasis_name = 'aug-cc-pvqz'
mole, one_dm, coeffs, occs, energies = topyscf(mol,verb_lvl=3)
mf = scf.RHF(mole)
grads1 = calc_moe_gradients_2p(mole,one_dm,mf,occs,energies)
grads2 = calc_moe_gradients_4p(mole,one_dm,mf,occs,energies)
grads3 = calc_moe_gradients_i4p(mole,one_dm,mf,occs,energies)
for j in range(0,np.count_nonzero(occs)) :
    for i in range(0,mole.natm) :
        assert(np.allclose(grads1[i][j,:],grads2[i][j,:],rtol=1e-4,atol=1e-6))
        assert(np.allclose(grads2[i][j,:],grads3[i][j,:],rtol=1e-4,atol=1e-6))


