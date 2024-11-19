import numpy as np
import matplotlib.pyplot as plt
import os

#On NERSC
sim_dir = '/global/cfs/projectdirs/sobs/awg_bb/'
obsmat_dir = 'bbmaster_paper/obs_mat_nside128_fpthin8'
f_name = 'obsmat_coadd-full.npz'
path = os.path.join(sim_dir, obsmat_dir, f_name)

#Load as Regular Mat
#obs_mat = np.load(path)
#data,indices,ptrs = obs_mat['data'],obs_mat['indices'],obs_mat['indptr']

#plt.hist(data,bins=60)
#plt.yscale("log")
#plt.spy(data)


##Load as SparseMat
import scipy.sparse
import matspy
P = scipy.sparse.load_npz(path)
main_diags = P.diagonal()
print(main_diags)
#plt.spy(P)
#fig,ax = matspy.spy_to_mpl(P)
#fig.savefig('CarlosObsMat.png', dpi=300,bbox_inches='tight')
