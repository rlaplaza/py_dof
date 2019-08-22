import numpy as np
from py_dof.moegradients import *


def print_imoe_gradients_4p(mole,one_dm,mf,occs,energies,maxstep=0.0001,verbose=0,moegrads=None) :
    if moegrads == None :
        moegrads = calc_moe_gradients_4p(mole,one_dm,mf,occs,energies,maxstep,verbose)
    for j in range(0,np.count_nonzero(occs)) :
        print('(Inverse) Nuclear Gradients (x,y,z) of Molecular Orbital {0} with Energy {1} a.u. :'.format(j+1,np.around(energies[j],6)))
        for i in range(0,mole.natm) :
            print(mole.atom[i,0],np.around(moegrads[i][j,:],6)*(-1))


def print_imoe_gradients_2p(mole,one_dm,mf,occs,energies,maxstep=0.00005,verbose=0,moegrads=None) :
    if moegrads == None :
        moegrads = calc_moe_gradients_2p(mole,one_dm,mf,occs,energies,maxstep,verbose)
    for j in range(0,np.count_nonzero(occs)) :
        print('(Inverse) Nuclear Gradients (x,y,z) of Molecular Orbital {0} with Energy {1} a.u. :'.format(j+1,np.around(energies[j],6)))
        for i in range(0,mole.natm) :
            print(mole.atom[i,0],np.around(moegrads[i][j,:],6)*(-1))
