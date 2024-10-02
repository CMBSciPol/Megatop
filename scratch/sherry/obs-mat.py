import numpy as np
import matplotlib.pyplot as plt

freq = '090'

file_dir = '/lustre/work/SO-MEGATOP/MSS2'
inds = {'090':'SAT1',
	'150':'SAT1',
	'230':'SAT3',
	'290':'SAT3',
	'030':'SAT4',
	'040':'SAT4',
}
freqs = inds.keys()

path = file_dir + '/sobs_RC1.r01_{}_mission_f{}_4way_coadd_sky_obsmat_healpix.npz'.format(inds[freq],freq)
obs_mat = np.load(path)

data,indices,ptrs = obs_mat['data'],obs_mat['indices'],obs_mat['indptr']

nside = 128
npix = 12 * nside ** 2
test_size = 30000
output = np.array([])

for i in range(test_size):
    ind_s, ind_e = ptrs[i:i+2]
    new_row = np.zeros(npix)
    if ind_e > ind_s:
        for ind in indices[ind_s:ind_e]:
            if ind <= npix:
                new_row[ind] = data[ind]
        output = np.vstack((output,new_row))

plt.plot(output)
plt.show()
#plt.savefig('test.png')
