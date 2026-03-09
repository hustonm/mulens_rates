import numpy as np
import pandas as pd
import pdb
from scipy.spatial import KDTree
from tqdm.auto import tqdm
#import h5py

def magsum(magss):
    return -2.5*np.log10(np.sum(10**(-0.4*magss),axis=0))

def calc_blends(dat0, blend_rad=None, filters=None):
    l = np.mean(dat0['l']); b = np.mean(dat0['b'])
    dat0.reset_index(drop=True,inplace=True)
    all_blends = np.zeros(len(dat0))
    dat_all = dat0.copy()
    dat_all.reset_index(inplace=True,names='orig_idx')
    all_l, all_b = dat_all['l'].to_numpy(), dat_all['b'].to_numpy()
    delta_l_cosb = ((all_l-360*(all_l>180))-l)*np.cos(all_b*np.pi/180)
    delta_b = all_b-b
    mags = dat_all[filters].to_numpy()
    mmin, mmax = np.floor(np.min(mags)), np.ceil(np.max(mags))
    
    blend_mags = []
    mlim = mmin
    pbar = tqdm(total=len(dat_all), position=0, leave=True)
    while len(dat_all)>0:
        # re-set arrays
        mags = dat_all[filters].to_numpy()
        all_l, all_b = dat_all['l'].to_numpy(), dat_all['b'].to_numpy()
        delta_l_cosb = ((all_l-360*(all_l>180))-l)*np.cos(all_b*np.pi/180)
        delta_b = all_b-b
        all_pts = np.transpose([all_l,all_b])
        kdt = KDTree(all_pts)

        # do the thing
        dmag = 0.5
        mlim = mlim + dmag
        bstars = np.where(mags<mlim)[0]
        all_qbp=[]
        if len(bstars)>0:
            qbp = kdt.query_ball_point(all_pts[bstars], blend_rad)
            all_qbp = np.concatenate(qbp)
            while len(np.unique(all_qbp)) != len(all_qbp):
                dmag = dmag/2
                mlim = mlim-dmag
                bstars = np.where(mags<mlim)[0]
                if len(bstars)==0:
                    all_qbp=[]
                    continue
                qbp = kdt.query_ball_point(all_pts[bstars], blend_rad)
                all_qbp = np.concatenate(qbp)
            for i,b in enumerate(bstars):
                #for
                bmag = magsum(mags[qbp[i],:])
                blend_mags.append(bmag)
            dat_all.drop(index=all_qbp, inplace=True)
            dat_all.reset_index(inplace=True, drop=True)
        pbar.update(len(all_qbp))
    return pd.DataFrame(data=np.array(blend_mags), columns=filters)

