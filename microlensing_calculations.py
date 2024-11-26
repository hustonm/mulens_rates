import numpy as np
import pandas as pd

# Supress the large number of nan warnings
import warnings
warnings.filterwarnings('ignore')

# Some constants
c, G, mSun, pctom = 3*10**8, 6.674*10**-11, 2*10**30, 3.086*10**16
auinkpc = 4.84814e-9
maglim = 18

# Angular Einstein ring calculation
def thetaE(lmass, ldist, sdist):
    return np.nan_to_num(np.sqrt(4*G*mSun*lmass*(ldist**-1 - sdist**-1)/(1000*pctom*c**2)), nan=0)

# Relative proper motion calculation (takes mas/yr, returns rad/day)
def muRel(lmul, smul, lmub, smub):  
    mu = np.sqrt((lmul-smul)**2 + (lmub-smub)**2)
    #   mas/yr -> as/yr ->   as/day   ->  deg/day    ->   rad/day
    return mu * (1./1000.) * (1./365.25) * (1.0/60.0/60.0) * np.pi/180.0

# This function calculates a series of microlensing observables for an (l,b) pointing, corresponding to pre-made lens and source catalogs
def mulens_stats(len_dir, src_dir, l, b, outputs=['l','b','n_source','n_lens','sa_source','sa_lens',
                                                  'avg_tau','avg_t','avg_theta',
                                                  'eventrate_area','eventrate_source',
                                                  'avg_ds','avg_dl',
                                                  'stdev_ds','stdev_dl','stdev_t',
                                                  'frac_bulge_lens', 'frac_disk_lens','frac_bulge_source','frac_disk_source'
                                                 ],
                    nsd=False, mag_band=None, mag_cut=np.inf):
    if nsd:
        outputs = np.append(outputs,'frac_nsd_lens')
        outputs = np.append(outputs,'frac_nsd_source')
    # Set up dictionary for return values
    return_dict = {}
    return_dict['l'] = l
    return_dict['b'] = b
    # read in data
    f_lens = len_dir+'l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.csv'
    f_src = src_dir+'l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.csv'
    srcs = pd.read_csv(f_src)
    lens = pd.read_csv(f_lens)
    f_lens_param = len_dir+'l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.log'
    f_src_param = src_dir+'l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.log'
    sv = []
    sav=0
    with open(f_lens_param) as lop:
        for line in lop:
            if sav==1:
                sv.append(line)
                sav=0
            if ('set solid_angle to' in line):
                sav=1
    la = float(sv[-1].split(' ')[-2])*(np.pi/180)**2
    sv = []
    with open(f_src_param) as lop:
        for line in lop:
            if sav==1:
                sv.append(line)
                sav=0
            if ('set solid_angle to' in line):
                sav=1
    sa = float(sv[-1].split(' ')[-2])*(np.pi/180)**2
    # Get the number of stars in each catalog
    nl = len(lens.index)
    # Save lens and source numbers and solid angles for output
    return_dict['n_lens'] = nl
    return_dict['sa_source'] = sa * (180/np.pi)**2
    return_dict['sa_lens'] = la * (180/np.pi)**2
    # Setting up values for sums
    sum_theta2, sum_thetamu, sum_theta2mu = 0.0, 0.0, 0.0
    sum_dsthetamu, sum_dlthetamu = 0.0, 0.0
    sum_bulgelens_thetamu, sum_disklens_thetamu = 0.0, 0.0
    sum_bulgesrc_thetamu, sum_disksrc_thetamu = 0.0, 0.0
    sum_nsdlens_thetamu, sum_nsdsrc_thetamu = 0.0,0.0
    # Get necessary lens data
    lens_dists = lens['Dist']
    lens_masses = lens['Mass']
    lens_muls = lens['mul']
    lens_mubs = lens['mub']
    lens_inbulge = (lens['pop']==0.0).astype(int)
    if nsd:
        lens_innsd = (lens['pop']==2.0).astype(int)
        lens_indisk = (lens['pop']>=3.0).astype(int)
    else:
        lens_indisk = (lens['pop']>=2.0).astype(int)
    # Recut source mag if needed
    if mag_band is None:
        do_srcs = srcs.index
    else:
        do_srcs = srcs[srcs[mag_band]<mag_cut].index
    ns = len(do_srcs)
    return_dict['n_source'] = ns
    # Loop over each source in the catalog
    for i_src in do_srcs:
        # Source data
        src_dist = srcs['Dist'][i_src]
        src_mul = srcs['mul'][i_src]
        src_mub = srcs['mub'][i_src]
        src_inbulge = (srcs['pop'][i_src]==0.0).astype(int)
        if nsd:
            src_innsd =  (srcs['pop'][i_src]==2.0).astype(int)
            src_indisk =  (srcs['pop'][i_src]>=3.0).astype(int)
        else:
            src_indisk =  (srcs['pop'][i_src]>=2.0).astype(int)
        # Calculate values for summations
        dist_comp = (src_dist > lens_dists).astype(int) #source further than lens
        theta_e = thetaE(lens_masses,lens_dists,src_dist)
        mu_rel = muRel(lens_muls,src_mul,lens_mubs,src_mub)
        # Add to sums
        sum_theta2 += sum(dist_comp*theta_e**2)
        sum_thetamu += np.nansum(dist_comp*mu_rel * theta_e)
        sum_theta2mu += np.nansum(dist_comp*mu_rel * theta_e**2)
        sum_dsthetamu += np.nansum(dist_comp*mu_rel * theta_e * src_dist)
        sum_dlthetamu += np.nansum(dist_comp*mu_rel * theta_e * lens_dists)
        sum_bulgelens_thetamu += np.nansum(dist_comp*mu_rel * theta_e * lens_inbulge)
        sum_disklens_thetamu += np.nansum(dist_comp*mu_rel * theta_e * lens_indisk)
        sum_bulgesrc_thetamu += np.nansum(dist_comp*mu_rel * theta_e * src_inbulge)
        sum_disksrc_thetamu += np.nansum(dist_comp*mu_rel * theta_e * src_indisk)
        if nsd:
            sum_nsdsrc_thetamu += np.nansum(dist_comp*mu_rel * theta_e * src_innsd)
            sum_nsdlens_thetamu += np.nansum(dist_comp*mu_rel * theta_e * lens_innsd)

    # Calculate averages
    return_dict['avg_tau'] = np.pi*sum_theta2/(ns*la)
    avg_t = sum_theta2/sum_thetamu
    return_dict['avg_t'] = avg_t
    return_dict['avg_theta'] = sum_theta2mu/sum_thetamu
    avg_ds, avg_dl = sum_dsthetamu/sum_thetamu, sum_dlthetamu/sum_thetamu
    return_dict['avg_ds'] = avg_ds
    return_dict['avg_dl'] = avg_dl
    # Calculate event rates
    return_dict['eventrate_area'] = sum_thetamu*2/la/(sa/(np.pi/180)**2) *365
    return_dict['eventrate_source'] = sum_thetamu*2/la/ns*365
    # Calculate bulge/disk fractions
    return_dict['frac_bulge_lens'] = sum_bulgelens_thetamu / sum_thetamu
    return_dict['frac_disk_lens'] = sum_disklens_thetamu / sum_thetamu
    return_dict['frac_bulge_source'] = sum_bulgesrc_thetamu / sum_thetamu
    return_dict['frac_disk_source'] = sum_disksrc_thetamu / sum_thetamu    
    if nsd:
        return_dict['frac_nsd_lens'] = sum_nsdlens_thetamu / sum_thetamu
        return_dict['frac_nsd_source'] = sum_nsdsrc_thetamu / sum_thetamu

    # Calculate standard deviations if selected
    if 'stdev_ds' in outputs or 'stdev_dl' in outputs or 'stdev_t' in outputs:
        sum_dsvarthetamu, sum_dlvarthetamu = 0.0, 0.0
        sum_tvarthetamu = 0.0
        for i_src in do_srcs:
            # Source data
            src_dist = srcs['Dist'][i_src]
            src_mul = srcs['mul'][i_src]
            src_mub = srcs['mub'][i_src]
            # Calculate values for summations
            dist_comp = (src_dist > lens_dists).astype(int) #source further than lens
            theta_e = thetaE(lens_masses,lens_dists,src_dist)
            mu_rel = muRel(lens_muls,src_mul,lens_mubs,src_mub)
            # sums
            sum_dsvarthetamu += np.nansum(dist_comp*mu_rel * theta_e * (src_dist-avg_ds)**2)
            sum_dlvarthetamu += np.nansum(dist_comp*mu_rel * theta_e * (lens_dists-avg_dl)**2)
            sum_tvarthetamu += np.nansum(dist_comp*mu_rel * theta_e * (theta_e/mu_rel-avg_t)**2)
        return_dict['stdev_ds'] = np.sqrt(sum_dsvarthetamu/sum_thetamu)
        return_dict['stdev_dl'] = np.sqrt(sum_dlvarthetamu/sum_thetamu)
        return_dict['stdev_t'] = np.sqrt(sum_tvarthetamu/sum_thetamu)

    # Return selected outputs
    return [return_dict[output] for output in outputs], outputs

def mulens_hist(len_dir, src_dir, l, b, mag_band=None, mag_cut=np.inf, t=30):
    # Set up dictionary for return values
    return_dict = {}
    return_dict['l'] = l
    return_dict['b'] = b
    # read in data
    f_lens = len_dir+'l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.csv'
    f_src = src_dir+'l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.csv'
    srcs = pd.read_csv(f_src)
    lens = pd.read_csv(f_lens)
    f_lens_param = len_dir+'l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.log'
    f_src_param = src_dir+'l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.log'
    sv = []
    sav=0
    with open(f_lens_param) as lop:
        for line in lop:
            if sav==1:
                sv.append(line)
                sav=0
            if ('set solid_angle to' in line):
                sav=1
    la = float(sv[-1].split(' ')[-2])*(np.pi/180)**2
    sv = []
    with open(f_src_param) as lop:
        for line in lop:
            if sav==1:
                sv.append(line)
                sav=0
            if ('set solid_angle to' in line):
                sav=1
    sa = float(sv[-1].split(' ')[-2])*(np.pi/180)**2
    # Get the number of stars in each catalog
    nl = len(lens.index)
    # Save lens and source numbers and solid angles for output
    return_dict['n_lens'] = nl
    return_dict['sa_source'] = sa * (180/np.pi)**2
    return_dict['sa_lens'] = la * (180/np.pi)**2
    # Setting up values for sums
    sum_theta2, sum_thetamu, sum_theta2mu = 0.0, 0.0, 0.0
    sum_dsthetamu, sum_dlthetamu = 0.0, 0.0
    sum_bulgelens_thetamu, sum_disklens_thetamu = 0.0, 0.0
    sum_bulgesrc_thetamu, sum_disksrc_thetamu = 0.0, 0.0
    # Get necessary lens data
    lens_dists = lens['Dist']
    lens_masses = lens['Mass']
    lens_muls = lens['mul']
    lens_mubs = lens['mub']
    lens_inbulge = (lens['pop']==0.0).astype(int)
    lens_indisk = (lens['pop']>=2.0).astype(int)
    # Recut source mag if needed
    if mag_band is None:
        do_srcs = srcs.index
    else:
        do_srcs = srcs[srcs[mag_band]<mag_cut].index
    ns = len(do_srcs)
    return_dict['n_source'] = ns
    # Loop over each source in the catalog
    thetamu,tes = [],[]
    for i_src in do_srcs:
        # Source data
        src_dist = srcs['Dist'][i_src]
        src_mul = srcs['mul'][i_src]
        src_mub = srcs['mub'][i_src]
        src_inbulge = (srcs['pop'][i_src]==0.0).astype(int)
        src_indisk =  (srcs['pop'][i_src]>=2.0).astype(int)
        # Calculate values for summations
        dist_comp = (src_dist > lens_dists).astype(int) #source further than lens
        theta_e = thetaE(lens_masses,lens_dists,src_dist)
        mu_rel = muRel(lens_muls,src_mul,lens_mubs,src_mub)
        # Add to sums
        thetamu.append(dist_comp*(2*mu_rel*theta_e*t + np.pi*theta_e**2))
        tes.append(dist_comp* theta_e/mu_rel)
        
    thetamu = np.reshape(thetamu,-1)
    tes = np.reshape(tes,-1)
    thetamu = thetamu[tes != 0.0]
    tes = tes[tes != 0.0]

    # Return selected outputs
    return thetamu, tes


