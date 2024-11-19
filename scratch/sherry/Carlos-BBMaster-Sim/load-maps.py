import healpy as hp
import os

path = '/global/cfs/projectdirs/sobs/awg_bb/bbmaster_paper/Foregrounds/d1/f090'
filename = 'filterbin_coadd-full_map.fits'

f = os.path.join(path,filename)

hp.fitsfunc.read_map(f)
