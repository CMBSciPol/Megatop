import numpy as np
import matplotlib.pyplot as plt
import os

#On NERSC
sim_dir = '/global/cfs/projectdirs/sobs/awg_bb/'
obsmat_dir = 'bbmaster_paper/obs_mat_nside128_fpthin8'
f_name = 'obsmat_coadd-full.npz'
path = os.path.join(sim_dir, obsmat_dir, f_name)

#MSS2 path
mss2_dir = '/global/cfs/cdirs/sobs/sims/mss-0002/RC1.r01'
mss2_name = 'sobs_RC1.r01_SAT1_mission_f090_4way_coadd_sky_obsmat_healpix.npz'
mss2_path = os.path.join(mss2_dir, mss2_name)

#Load as Regular Mat
#obs_mat = np.load(path)
#data,indices,ptrs = obs_mat['data'],obs_mat['indices'],obs_mat['indptr']

#plt.hist(data,bins=60)
#plt.yscale("log")
#plt.spy(data)

##Load as SparseMat
import scipy.sparse
import matspy

P = scipy.sparse.load_npz(mss2_path)

size = 196608
# Make into TT only:
#P_TT = P[:size, :size]
#scipy.sparse.save_npz('MSS2_obs_TT.npz',P_TT)

#Save the diagonals:


#fig,ax = matspy.spy_to_mpl(P)
#fig.savefig('MSS2ObsMat.png', dpi=300,bbox_inches='tight')


