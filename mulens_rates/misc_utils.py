import numpy as np
import pandas as pd
import pdb
from scipy.spatial import KDTree
from tqdm.auto import tqdm
#import h5py
from itertools import zip_longest
import warnings

def magsum(magss):
    return -2.5*np.log10(np.sum(10**(-0.4*magss),axis=1))

def mag_weighted_sum(mags, vals):
    fluxes = 10**(-0.4*mags)
    return (np.sum(fluxes.T*vals.T, axis=1)/np.sum(fluxes,axis=1)).T

"""
Use a KDTree to calculate unique point clusters with maximum radius
"""
def calc_clusters(points, radius):
    # Set up KDTree and array to note whether each point has been assigned a cluster
    tree = KDTree(points)
    n_points = len(points)
    assigned = np.zeros(n_points, dtype=bool)
    clusters = []
    for i in tqdm(range(n_points)):
        if assigned[i]:
            # Skip stars already assigned to a cluster
            continue
        # Find unassigned neighbors and save cluster
        neighbor_indices = tree.query_ball_point(points[i], r=radius)
        current_cluster = [idx for idx in neighbor_indices if not assigned[idx]]
        assigned[current_cluster] = True
        clusters.append(current_cluster)
    return clusters


"""
Take a resolved star catalog and blend it, summing magnitudes
and if desired calculating flux-weighted positions, proper
motions, and parallaxes.
"""
def calc_blends(dat_all, blend_rad, filters, 
                primary_filter=None, calc_params=True):
    # Clean up table and sort by magnitude
    if primary_filter is None:
        warnings.warn(f"No primary_filter provided. Using {filters[0]}"+
            " to sort stars and flux-weight additional parameters.")
        primary_filter = filters[0]
    dat_all = dat_all[~np.isnan(dat_all[primary_filter])].copy()
    dat_all.sort_values(by=primary_filter)
    dat_all.reset_index(drop=True,inplace=True)

    # Get coordinates into simplified delta_l_cosb and delta_b frame
    l_mean = np.mean(dat_all['l']); b_mean = np.mean(dat_all['b'])
    all_l, all_b = dat_all['l'].to_numpy(), dat_all['b'].to_numpy()
    delta_l_cosb = ((all_l-360*(all_l>180))-l_mean)*np.cos(all_b*np.pi/180)
    delta_b = all_b-b_mean
    all_pts = np.transpose([all_l,all_b])

    # Set up arrays with necessary data
    mags = dat_all[filters].to_numpy()
    if calc_params:
        mags_prim = dat_all[primary_filter].to_numpy()
        dat_all.loc[:,'plx'] = 1/dat_all['Dist']
        other_vals = dat_all[['l','b','mul','mub','plx']].to_numpy()
        blend_vals = []

    # Run the cluster calculation on the sorted/cleaned points
    clusters0 = calc_clusters(all_pts, blend_rad) 
    # Turn this into a masked array for easy mag + param sums
    clusters_arr = np.array(list(zip_longest(*clusters0, fillvalue=-1))).T
    clusters = np.ma.masked_values(clusters_arr, -1)
    #pdb.set_trace()

    # Compute the magnitude sums & other values if desired
    mags = np.append(mags, [np.repeat(np.inf, len(filters))], axis=0)
    bmags = magsum(mags[clusters,:])
    out = pd.DataFrame(data=bmags, columns=filters)
    if calc_params:
        mags = np.append(mags_prim, [np.inf])
        other_vals = np.append(other_vals, [np.repeat(0, 5)], axis=0)
        bvals = mag_weighted_sum(mags_prim[clusters], other_vals[clusters,:])
        out2 = pd.DataFrame(data=bvals, columns=['l','b','mul','mub','plx'])
        return pd.concat([out,out2],axis=1)
    return out



"""
Significantly less efficient version of calc_blends
"""
def calc_blends_OLD(dat0, blend_rad=None, filters=None, primary_filter=None):
    l = np.mean(dat0['l']); b = np.mean(dat0['b'])
    dat0.reset_index(drop=True,inplace=True)
    all_blends = np.zeros(len(dat0))
    dat_all = dat0.copy()
    dat_all.reset_index(inplace=True,names='orig_idx')
    all_l, all_b = dat_all['l'].to_numpy(), dat_all['b'].to_numpy()
    delta_l_cosb = ((all_l-360*(all_l>180))-l)*np.cos(all_b*np.pi/180)
    delta_b = all_b-b
    mags = dat_all[filters].to_numpy()
    if primary_filter:
        mags_prim = dat_all[primary_filter].to_numpy()
        dat_all.loc[:,'plx'] = 1/dat_all['Dist']
        other_vals = dat_all[['mul','mub','plx']].to_numpy()
        blend_vals = []
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
            #for i,b in enumerate(bstars):
            #    #for
            max_len = max(len(l) for l in qbp)
            qbp_tmp = np.array([l + [-1] * (max_len - len(l)) for l in qbp])
            qbp_new = np.ma.masked_equal(qbp_tmp, -1)
            bmag = magsum(mags[qbp_new,:])
            if primary_filter:
                bvals = mag_weighted_sum(mags_prim[qbp_new], other_vals[qbp_new,:])
                blend_vals.append(bvals)
            blend_mags.append(bmag)
            dat_all.drop(index=all_qbp, inplace=True)
            dat_all.reset_index(inplace=True, drop=True)
        pbar.update(len(all_qbp))
    out = pd.DataFrame(data=np.concatenate(blend_mags), columns=filters)
    out2 = pd.DataFrame(data=np.concatenate(blend_vals), columns=['mul','mub','plx'])
    return pd.concat([out,out2],axis=1)
