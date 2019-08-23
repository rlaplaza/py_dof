from py_dof import *
import numpy as np

def test_compare_moegradients():
    mol = read_one('water_dz.molden')
    mol.obasis_name = 'aug-cc-pvdz'
    mole, one_dm, coeffs, occs, energies = topyscf(mol,verb_lvl=3)
    mf = scf.RHF(mole)

    # Reference gradients using the improved 6-point formula
    grads_ref = calc_moe_gradients_i6p(mole,one_dm,mf,occs,energies)

    # The 6-point formula is quite robust (less than 0.1% difference per gradient component)
    grads_ref_finer = calc_moe_gradients_i6p(mole,one_dm,mf,occs,energies,maxstep=0.000001)
    for j in range(0,2*np.count_nonzero(occs)) :
        for i in range(0,mole.natm) :
            assert(np.allclose(grads_ref[i][j,:],grads_ref_finer[i][j,:],rtol=1e-4,atol=1e-6))

    # Default setting, molecular orbital energy gradients are the same with the default 2-point formula
    grads2 = calc_moe_gradients_2p(mole,one_dm,mf,occs,energies)
    for j in range(0,2*np.count_nonzero(occs)) :
        for i in range(0,mole.natm) :
            assert(np.allclose(grads_ref[i][j,:],grads2[i][j,:],rtol=1e-4,atol=1e-6))
    
    # Increasing the step size leads to differences for the 2-point formula,
    # but is still quite valid with the improved 4-point formula
    grads2 = calc_moe_gradients_2p(mole,one_dm,mf,occs,energies,maxstep=0.0001)
    error2 = 0.0
    grads4 = calc_moe_gradients_4p(mole,one_dm,mf,occs,energies,maxstep=0.0001)
    error4 = 0.0
    gradsi4 = calc_moe_gradients_i4p(mole,one_dm,mf,occs,energies,maxstep=0.0001)
    errori4 = 0.0
    gradsi6 = calc_moe_gradients_i4p(mole,one_dm,mf,occs,energies,maxstep=0.0001)
    errori6 = 0.0
    for j in range(0,2*np.count_nonzero(occs)) :
        for i in range(0,mole.natm) :
            error2 += abs(grads_ref[i][j,2]-grads2[i][j,2])
            error4 += abs(grads_ref[i][j,2]-grads4[i][j,2])
            errori4 += abs(grads_ref[i][j,2]-gradsi4[i][j,2])
            errori6 += abs(grads_ref[i][j,2]-gradsi6[i][j,2])
    #print(error2,error4,errori4,errori6)
    # -> 0.2844535270133687 0.0948179026378593 0.035556753034606925 0.03555675519126715
    assert( error2 > 2*error4 and error4 > 2*errori4 and np.isclose(errori4,errori6) )
    assert(np.isclose(error2,0.2844535270133687))
    assert(np.isclose(error4,0.0948179026378593))
    assert(np.isclose(errori4,0.035556753034606925))
