import numpy as np
import pandas as pd
import pdb
import h5py

# Supress the large number of nan warnings
import warnings
warnings.filterwarnings('ignore')

# Some constants
c, G, mSun, pctom = 3*10**8, 6.674*10**-11, 2*10**30, 3.086*10**16
auinkpc = 4.84814e-9

# Units provided by this function
output_units = {'l':'deg', 'b':'deg', 
                'n_source':'', 'n_lens':'',
                'sa_source':'sr','sa_lens':'sr',
                'avg_tau':'', 'avg_t':'day', 'avg_logt':'log(day)',
                'avg_murel':'mas/yr', 'avg_theta':'mas',
                'eventrate_area':'/deg^2/yr', 'eventrate_source':'/yr',
                'stdev_t':'day', 'stdev_logt':'log(day)',
                'avg_ds':'kpc', 'avg_dl':'kpc', 'stdev_ds':'kpc', 'stdev_dl':'kpc',
                'frac_bulge_lens':'', 'frac_disk_lens':'', 'frac_bulge_source':'','frac_disk_source':'',
                'n_compact_obj':'', 'frac_compact_obj':'', 'frac_lowmass':'',
                'frac_nsd_lens':'', 'frac_nsd_source':'', 'frac_det_blue':''
                }

# Angular Einstein ring calculation
def calc_thetaE(lmass, ldist, sdist):
    return np.nan_to_num(np.sqrt(4*G*mSun*lmass*(ldist**-1 - sdist**-1)/(1000*pctom*c**2)), nan=0)

# Relative proper motion calculation (takes mas/yr, returns rad/day)
def calc_muRel(lmul, smul, lmub, smub):  
    mu = np.sqrt((lmul-smul)**2 + (lmub-smub)**2)
    #   mas/yr -> as/yr ->   as/day   ->  deg/day    ->   rad/day
    return mu * (1./1000.) * (1./365.25) * (1.0/60.0/60.0) * np.pi/180.0

'''
Inputs: lens and source directory and galactic coordinates for synthpop catalogs.
Optional: additional magnitude cuts to make and whether to include nsd component
Output: lists of microlensing rates and averages, values and descriptions
'''
def mulens_stats(l, b, f_lens, f_src, field_id=None, outputs=['l','b','n_source','n_lens','sa_source','sa_lens',
                                                  'avg_tau','avg_murel','avg_theta',
                                                  'eventrate_area','eventrate_source',
                                                  'avg_t','avg_logt',
                                                  'stdev_t','stdev_logt',
                                                  'avg_ds','avg_dl',
                                                  'stdev_ds','stdev_dl','stdev_t',
                                                  'frac_bulge_lens', 'frac_disk_lens',
                                                  'frac_bulge_source','frac_disk_source',
                                                  'n_compact_obj','frac_compact_obj', 'frac_lowmass',
                                                  'field_id', 'f_src', 'f_lens'
                                                 ],
                    nsd=False, roman_blue=False, mag_band=None, mag_cut=np.inf,
                    f_lens_kwargs={}, f_src_kwargs={},
                    sa_lens=None, sa_src=None,
                    tE_range=None, use_n=np.inf):
    if nsd:
        outputs = np.append(outputs,'frac_nsd_lens')
        outputs = np.append(outputs,'frac_nsd_source')
    if roman_blue:
        outputs = np.append(outputs, 'frac_det_blue')
    # Set up dictionary for return values
    return_dict = {}
    return_dict['l'] = l
    return_dict['b'] = b
    # read in data
    #f_lens = len_dir+'l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.csv'
    #f_src = src_dir+'l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.csv'
    l_cols = ['Mass','iMass','Dist','pop','mul','mub','phase']
    s_cols = ['Dist','pop','mul','mub']
    if mag_band is not None:
        s_cols.append(mag_band)
    if roman_blue:
        s_cols.append('Z087')
    if f_src.split('.')[-1]=='csv':
        srcs = pd.read_csv(f_src, usecols=s_cols, *f_src_kwargs)
    elif f_src.split('.')[-1]=='h5':
        srcs = pd.read_hdf(f_src, key='data', usecols=s_cols, *f_src_kwargs)
    if f_lens.split('.')[-1]=='csv':
        lens = pd.read_csv(f_lens, usecols=l_cols, *f_lens_kwargs)
    elif f_lens.split('.')[-1]=='h5':
        lens = pd.read_hdf(f_lens, key='data', usecols=l_cols, *f_lens_kwargs)

    # Get solid angle in steradians from file (convert from deg^2 if needed)
    f_lens_param = ".".join(f_lens.split('.')[:-1])+'.log'
    f_src_param = ".".join(f_src.split('.')[:-1])+'.log'
    sv = []
    sav=0
    if sa_lens is None:
        with open(f_lens_param) as lop:
            for line in lop:
                if sav==1:
                    sv.append(line)
                    sav=0
                if ('set solid_angle to' in line):
                    sav=1
        row = sv[-1].split(' ')
        la = float(row[-2]) * ((np.pi/180)**2)**int('deg^2' in row[-1])
    else:
        la = sa_lens * ((np.pi/180)**2)
    sv = []
    if sa_src is None:
        with open(f_src_param) as lop:
            for line in lop:
                if sav==1:
                    sv.append(line)
                    sav=0
                if ('set solid_angle to' in line):
                    sav=1
        row = sv[-1].split(' ')
        sa = float(row[-2]) * ((np.pi/180)**2)**int('deg^2' in row[-1])
    else:
        sa = sa_src * ((np.pi/180)**2)

    # Check for downsampling
    if len(lens)>use_n:
        la = la*use_n/len(lens)
        lens = lens.sample(n=use_n, ignore_index=True)
    else:
        lens.reset_index(drop=True, inplace=True)
    if len(srcs)>use_n:
        sa = sa*use_n/len(srcs)
        srcs = srcs.sample(n=use_n, ignore_index=True)
    else:
        srcs.reset_index(drop=True, inplace=True)

    # Get the number of stars in each catalog
    nl = len(lens.index)
    # Save lens and source numbers and solid angles for output
    return_dict['n_lens'] = nl
    return_dict['sa_source'] = sa 
    return_dict['sa_lens'] = la 

    # Get necessary lens data
    lens_idxs = np.array(lens.index)
    lens_dists = np.array(lens['Dist'])
    lens_masses = np.array(lens['Mass'])
    lens_muls = np.array(lens['mul'])
    lens_mubs = np.array(lens['mub'])
    lens_pops = np.array(lens['pop'])
    lens_cos = (np.array(lens['phase'])>100).astype(int)
    lens_lowmass = (np.array(lens['iMass'])<0.1).astype(int)
    lens_inbulge = np.array((lens_pops==0.0).astype(int))
    if nsd:
        lens_innsd = np.array((lens_pops==2.0).astype(int))
        lens_indisk = np.array((lens_pops>=3.0).astype(int))
    else:
        lens_indisk = np.array((lens_pops>=2.0).astype(int))
    
    # Recut source mag if needed
    if mag_band is None:
        do_srcs = srcs.index
    else:
        do_srcs = srcs[srcs[mag_band]<mag_cut].index
    # Loop over each source in the catalog
    do_srcs = srcs.loc[do_srcs].reset_index()
    src_idxs = np.array(do_srcs.index)
    ns = len(src_idxs)
    return_dict['n_source'] = ns
    srcs,lenses = np.meshgrid(src_idxs,lens_idxs)
    srcs,lenses = srcs.flatten(),lenses.flatten()
    src_dists = np.array(do_srcs['Dist'])
    src_muls = np.array(do_srcs['mul'])
    src_mubs = np.array(do_srcs['mub'])
    src_pops = np.array(do_srcs['pop'])
    if roman_blue:
        src_Z087 = (np.array(do_srcs['Z087'] < 24)).astype(int)
    
    src_inbulge = (src_pops==0.0).astype(int)
    if nsd:
        src_innsd =  (src_pops==2.0).astype(int)
        src_indisk =  (src_pops>=3.0).astype(int)
    else:
        src_indisk =  (src_pops>=2.0).astype(int)

    #pdb.set_trace()
    dist_comp = (src_dists[srcs] > lens_dists[lenses]) #source further than lens
    use_srcs = srcs[dist_comp]
    use_lens = lenses[dist_comp]
    
    theta_e = calc_thetaE(lens_masses[use_lens],lens_dists[use_lens],src_dists[use_srcs])
    mu_rel = calc_muRel(lens_muls[use_lens],src_muls[use_srcs],lens_mubs[use_lens],src_mubs[use_srcs])
    t_e = theta_e/mu_rel
    if tE_range is not None:
        good_tes = ((t_e>tE_range[0]) & (t_e<tE_range[1]))
        use_srcs = use_srcs[good_tes]
        use_lens = use_lens[good_tes]
        theta_e = calc_thetaE(lens_masses[use_lens],lens_dists[use_lens],src_dists[use_srcs])
        mu_rel = calc_muRel(lens_muls[use_lens],src_muls[use_srcs],lens_mubs[use_lens],src_mubs[use_srcs])
        #print('     tE cut keep fraction:',sum(good_tes)/len(good_tes))

    thetamu = theta_e*mu_rel
    sum_thetamu = np.sum(thetamu)

    return_dict['avg_t'] = np.average(theta_e/mu_rel, weights=theta_e*mu_rel)
    return_dict['avg_logt'] = np.average(np.log10(theta_e/mu_rel), weights=theta_e*mu_rel)
    return_dict['avg_theta'] = np.average(theta_e*180/np.pi*60*60*1000, weights=theta_e*mu_rel)
    return_dict['avg_ds'] = np.average(src_dists[use_srcs], weights=theta_e*mu_rel)
    return_dict['avg_dl'] = np.average(lens_dists[use_lens], weights=theta_e*mu_rel)
    return_dict['avg_tau'] = np.pi*np.sum(theta_e**2)/(ns*la)
    return_dict['avg_murel'] = np.average(mu_rel*(1000*365.25*60**2*180.0/np.pi), weights=theta_e*mu_rel) #mas/yr
    return_dict['eventrate_area'] = sum_thetamu*2/la/(sa/(np.pi/180)**2) *365.25
    return_dict['eventrate_source'] = sum_thetamu*2/la/ns*365.25
    
    return_dict['frac_bulge_lens'] = np.sum(lens_inbulge[use_lens]*thetamu) / sum_thetamu
    return_dict['frac_disk_lens'] = np.sum(lens_indisk[use_lens]*thetamu) / sum_thetamu
    return_dict['frac_bulge_source'] = np.sum(src_inbulge[use_srcs]*thetamu) / sum_thetamu
    return_dict['frac_disk_source'] = np.sum(src_indisk[use_srcs]*thetamu) / sum_thetamu    
    if nsd:
        return_dict['frac_nsd_lens'] = np.sum(lens_innsd[use_lens]*thetamu) / sum_thetamu
        return_dict['frac_nsd_source'] = np.sum(src_innsd[use_srcs]*thetamu) / sum_thetamu
    if roman_blue:
        return_dict['frac_det_blue'] = np.sum(src_Z087[use_srcs]*thetamu) / sum_thetamu
    return_dict['frac_compact_obj'] = np.sum(lens_cos[use_lens]*thetamu) / sum_thetamu    
    return_dict['frac_lowmass'] = np.sum(lens_lowmass[use_lens]*thetamu) / sum_thetamu    
    return_dict['n_compact_obj'] = np.sum(lens_cos)


    return_dict['stdev_t'] = np.sqrt(np.average((theta_e/mu_rel-return_dict['avg_t'])**2, weights=theta_e*mu_rel))
    return_dict['stdev_logt'] = np.sqrt(np.average((np.log10(theta_e/mu_rel)-return_dict['avg_logt'])**2, weights=theta_e*mu_rel))
    return_dict['stdev_ds'] = np.sqrt(np.average((src_dists[use_srcs]-return_dict['avg_ds'])**2, weights=theta_e*mu_rel))
    return_dict['stdev_dl'] = np.sqrt(np.average((lens_dists[use_lens]-return_dict['avg_dl'])**2, weights=theta_e*mu_rel))
    return_dict['stdev_t'] = np.sqrt(np.average((theta_e/mu_rel - return_dict['avg_t'])**2, weights=theta_e*mu_rel))

    return_dict['field_id'] = field_id
    return_dict['f_lens'] = f_lens
    return_dict['f_src'] = f_src

    # Return selected outputs'''
    return [return_dict[output] for output in outputs], outputs



# Switch from my output format to that needed for GULLS input
def prep_rates_for_gulls(rates_orig, chip_side=0.125, remove_dir_layers=0):
    rates_gulls_dict  = {'ID_src': rates_orig.field_id, 'l_src':rates_orig.l, 'b_src':rates_orig.b,
            'fa_l_src': np.ones(len(rates_orig))*chip_side, 'fa_b_src': np.ones(len(rates_orig))*chip_side, 
            'sa_src':rates_orig.sa_source*(180/np.pi)**2, 
            'file_src':['/'.join(x.split('/')[remove_dir_layers:]) for x in rates_orig.f_src],
            'ID_lens': rates_orig.field_id, 'l_lens':rates_orig.l, 'b_lens':rates_orig.b,
            'fa_l_lens': np.ones(len(rates_orig))*chip_side, 'fa_b_lens': np.ones(len(rates_orig))*chip_side, 
            'sa_lens':rates_orig.sa_lens*(180/np.pi)**2, 
            'file_lens':['/'.join(x.split('/')[remove_dir_layers:]) for x in rates_orig.f_lens],
            'nsource':rates_orig.n_source, 'source_area':rates_orig.sa_source*(180/np.pi)**2,
            'nlens':rates_orig.n_lens, 'lens_area':rates_orig.sa_lens*(180/np.pi)**2,
            'tau':rates_orig.avg_tau, 'tEmean':rates_orig.avg_t, 'murelmean':rates_orig.avg_murel, 
            'nevents_per_tile':rates_orig.eventrate_area*chip_side**2, 'nevents_per_source':rates_orig.eventrate_source,
            'nevents_per_deg2':rates_orig.eventrate_area, 'source_density':rates_orig.n_source/(rates_orig.sa_source*(180/np.pi*60)**2)
            }
    #gulls_output.to_csv('mulens_rates_gulls.txt', index=False, sep=' ')
    rates_gulls = pd.DataFrame(rates_gulls_dict)
    return rates_gulls


def mulens_events(len_dir, src_dir, output_dir, l, b,
                    mag_band=None, mag_cut=np.inf):
    #print('STARTING',l,b)
    # Set up dictionary for return values
    return_dict = {}
    return_dict['l'] = l
    return_dict['b'] = b
    # read in data
    f_lens = len_dir+'l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.csv'
    f_src = src_dir+'l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.csv'
    l_cols = ['Mass','Dist','pop','mul','mub','phase']
    s_cols = ['Dist','pop','mul','mub']
    if mag_band is not None:
        s_cols.append(mag_band)
    srcs = pd.read_csv(f_src, usecols=s_cols)
    lens = pd.read_csv(f_lens, usecols=l_cols)
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

    # Get necessary lens data
    lens_idxs = np.array(lens.index)
    lens_dists = np.array(lens['Dist'])
    lens_masses = np.array(lens['Mass'])
    lens_muls = np.array(lens['mul'])
    lens_mubs = np.array(lens['mub'])
    lens_pops = np.array(lens['pop'])
    lens_inbulge = np.array((lens_pops==0.0).astype(int))
    
    # Recut source mag if needed
    if mag_band is None:
        do_srcs = srcs.index
    else:
        do_srcs = srcs[srcs[mag_band]<mag_cut].index
    # Loop over each source in the catalog
    do_srcs = srcs.loc[do_srcs].reset_index()
    src_idxs = np.array(do_srcs.index)
    ns = len(src_idxs)
    return_dict['n_source'] = ns
    srcs,lenses = np.meshgrid(src_idxs,lens_idxs)
    srcs,lenses = srcs.flatten(),lenses.flatten()
    src_dists = np.array(do_srcs['Dist'])
    src_muls = np.array(do_srcs['mul'])
    src_mubs = np.array(do_srcs['mub'])
    src_pops = np.array(do_srcs['pop'])

    dist_comp = (src_dists[srcs] > lens_dists[lenses]) #source further than lens
    use_srcs = srcs[dist_comp]
    use_lens = lenses[dist_comp]
    
    theta_e = calc_thetaE(lens_masses[use_lens],lens_dists[use_lens],src_dists[use_srcs])
    mu_rel = calc_muRel(lens_muls[use_lens],src_muls[use_srcs],lens_mubs[use_lens],src_mubs[use_srcs])
    thetamu = theta_e*mu_rel
    sum_thetamu = np.sum(thetamu)
    
    event_ml = lens_masses[use_lens]
    event_dl = lens_dists[use_lens]
    event_ds = src_dists[use_srcs]
    event_lmul = lens_muls[use_lens]
    event_lmub = lens_mubs[use_lens]
    event_smul = src_muls[use_srcs]
    event_smub = src_mubs[use_srcs]
    event_lpop = lens_pops[use_lens]
    event_spop = src_pops[use_srcs]
    event_ltype = np.array(lens['phase'])[use_lens]

    dbins = np.arange(0,25.001,0.5)
    #ndls = np.transpose(np.array(list(map(lambda i: [sum((event_dl*thetamu)[(event_dl>dbins[i+0]) & (event_dl<dbins[i+1])]), 
    #                        sum((event_ds*thetamu)[(event_ds>dbins[i+0]) & (event_ds<dbins[i+1])])], 
    #                        range(len(dbins)-1))))/sum_thetamu)
    ndl = np.histogram(event_dl, bins=dbins, weights=thetamu)[0]/sum_thetamu
    nds = np.histogram(event_ds, bins=dbins, weights=thetamu)[0]/sum_thetamu
    
    df = pd.DataFrame(data=np.transpose([dbins[1:],ndl,nds]), 
                      columns=['dbins','fdl','fds'])
    df.to_csv(output_dir+'/'+'l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.csv', index=False)
    #print('saved to',output_dir+'/'+'l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.csv')

    # Return event list
    return [dbins[1:],ndl,nds], ['dbins','fdl','fds']



def events_for_popclass(h5_file, use_stars_per_bin=1e3):
    """
    Draw microlensing events for random lens, source pairs from a 
    PopSyCLE singles-only catalog. 
    
    Parameters
    ----------
    hdf5_file : str
        Filename of an hdf5 file.

    use_stars_per_bin : str
        Order of magnitude number of stars per bin to use for the 
        calculation. Prevents excessive memory/computation.
        Default is 1e3.

    Returns
    -------
    thetaEs : np.array
        array of Einstein ring radii for events (mas)
    piEs : np.array
        array of microlensing parallaxes for events (unitless)
    tEs : np.array
        array of timescales for events (days)
    weights : np.array
        array of relative weights for events (mu_rel * thetaE)
    """
    hf = h5py.File(h5_file, 'r')
    thetaEs = []
    piEs = []
    tEs = []
    weights = []
    for k in list(hf.keys()):
        if '_' not in k:
            print(k)
            dat = hf[k]

            if dat.shape[0] > 0:
                ds = int(np.floor(dat.shape[0]/use_stars_per_bin))
                patch = dat[::ds]
                print(len(patch))
                dists = patch['rad']
                mul = patch['mu_lcosb']
                mub = patch['mu_b']
                masses = patch['mass']
                idx = np.arange(len(masses))
                del patch

                src_idxs, lens_idxs = np.meshgrid(idx, idx)
                src_idxs, lens_idxs = src_idxs.ravel(), lens_idxs.ravel()

                dist_comp = (dists[src_idxs] > dists[lens_idxs]) #source further than lens
                use_srcs = src_idxs[dist_comp]
                use_lens = lens_idxs[dist_comp]

                # Microlensing math
                # get theta_e, convert rad to mas
                theta_e = calc_thetaE(masses[use_lens], dists[use_lens], dists[use_srcs]) * 180/np.pi * 60**2 * 1000
                # get mu_rel, TODO: confirm provided units are mas/yr
                mu_rel = np.sqrt((mul[use_lens]-mul[use_srcs])**2 + (mub[use_lens]-mub[use_srcs])**2)
                # pi_rel, TODO: confirm distance units
                pi_rel = (1/dists[use_lens] - 1/dists[use_srcs])
                pi_e = pi_rel / theta_e
                t_e = theta_e/mu_rel * 365.25 # years -> days
                thetamu = theta_e*mu_rel
                thetaEs.append(theta_e)
                piEs.append(pi_e)
                tEs.append(t_e)
                weights.append(thetamu)
    return np.concatenate(thetaEs), np.concatenate(piEs), np.concatenate(tEs), np.concatenate(weights)
