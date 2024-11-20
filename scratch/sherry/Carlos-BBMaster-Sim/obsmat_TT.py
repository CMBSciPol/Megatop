import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy.sparse
#import matspy

OBS = scipy.sparse.load_npz('obs_TT.npz')
r,c = OBS.nonzero()
non_zero_pixels = np.array(list(set(c))).astype(int)
nside = 128
npix = hp.nside2npix(nside)
mask = np.full(npix, True, dtype=bool)
mask[non_zero_pixels] = False

healpix_mask = hp.ma(np.ones(npix))  # Start with a full map of ones
healpix_mask.mask = mask  # Apply the mask

# Visualize the mask
hp.mollview(healpix_mask, nest=True, title="ObsMatAffectedPixels") #The pixels are nested.
plt.savefig('ObsMatPixels.png', dpi=300,bbox_inches='tight')
