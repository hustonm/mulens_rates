import numpy as np
import synthpop as sp
import pandas as pd
import shutil
import h5py

old_cols = [ 'logL', 'logTeff', 'logg', 'Fe/H_evolved', 'log_radius']
new_cols = [ 'log_L', 'log_Teff', 'log_g', '[Fe/H]', 'log_R']

def prep_gulls(l,b):
    f_orig = dir_orig+'/Huston2025_l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.h5'
    f_new = dir_new+'/Huston2025_l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.csv'
    dat = pd.read_hdf(f_orig, key='data')
    xyz = mod.populations[0].coord_trans.dlb_to_xyz(*dat[['Dist','l','b']].to_numpy().T)
    dat.loc[:,'x'] = xyz[0]
    dat.loc[:,'y'] = xyz[1]
    dat.loc[:,'z'] = xyz[2]
    dat.rename(columns=dict(zip(old_cols,new_cols)), inplace=True)
    dat_new = mod.post_processing(dat)
    dat_new.to_csv(f_new,sep=' ', index=False)
    print(f'  processed l={l:2.3f}, b={b:2.3f}')
    n = len(dat)
    with h5py.File(f_orig, 'r') as hf:
        sa = hf['solid_angle'][()]
    return n, sa

sets = ['source', 'lens', 'faint', 'mid2', 'mid1', 'bright', 'extra-bright']

mod = sp.SynthPop('huston2025_defaults.synthpop_conf', extinction_map_kwargs={"name":"NoExtinction"},
                  post_processing_kwargs={"name":"GullsPostProcessing", "cat_type":"n/a"},
                 )
mod.init_populations()

df = pd.DataFrame()

for set_name in sets:
    dir_orig = 'roman_dense_grid/'+set_name
    dir_new = 'roman_dense_grid_gulls/'+set_name

    n_stars = []
    solangs = []
    ls, bs = [], []

    for b in np.arange(-2.20, -0.70+0.01, 0.05):
        for l in np.arange(-1.00, 1.80+0.01, 0.05):
            n, sa = prep_gulls(l,b)
            n_stars.append(n)
            solangs.append(sa)
            ls.append(np.round(l, 3))
            bs.append(np.round(b, 3))
    for b in np.arange(-0.70+0.05, 0.5+0.01, 0.05):
        for l in np.arange(-0.5, 0.5+0.01, 0.05):
            n, sa = prep_gulls(l,b)
            n_stars.append(n)
            solangs.append(sa)
            ls.append(np.round(l, 3))
            bs.append(np.round(b, 3))

    if set_name=='source':
        df['l'] = ls 
        df['b'] = bs 
    df['nstars_'+set_name] = n_stars
    df['sa_deg2_'+set_name] = solangs

    print(f'{set_name} complete')

df.to_csv('roman_dense_grid_gulls/catalog_info.txt', sep=' ', index=False)
