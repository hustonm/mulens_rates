import numpy as np
import synthpop as sp
import pandas as pd
import h5py
import pdb
import os

cat_type=['lens','source','extra-bright', 'bright', 'mid1', 'mid2', 'faint']
maglim_dim    = [None,  25,      12,      17, 20, 24, 35]
maglim_bright = [None, None, None, 12, 17, 20, 24]
goal_count = [25000, 25000, 2500, 5000, 10000, 10000, 5000]

def run_field(l,b, solang=np.pi*0.025**2):
    # rerun=False
    # for i, cat in enumerate(cat_type):
    #     out_file = f'/Volumes/g-drive/huston2026_synthpop/dense_grid_extcor/{cat}/Huston2025_l{l:2.3f}_b{b:2.3f}.h5'
    #     if os.path.isfile(out_file):
    #         mag_stars = pd.read_hdf(out_file)
    #         with h5py.File(out_file, 'a') as hf:
    #             use_solang = hf['solid_angle'][()]
    #         if ((len(mag_stars) < goal_count[i]*0.8) and (use_solang<solang)):
    #             rerun=True
    #         elif len(mag_stars)>(goal_count[i]*1.2):
    #             use_solang = goal_count[i]/len(mag_stars) * use_solang
    #             lim_ang_sep = np.sqrt(use_solang/np.pi)
    #             mag_stars_recut = mag_stars[mag_stars['ang_sep']<lim_ang_sep].copy()
    #             while (len(mag_stars_recut)<(goal_count[i]*0.8)) or \
    #                     (len(mag_stars_recut)>(goal_count[i]*1.2)):
    #                 use_solang = goal_count[i]/len(mag_stars_recut) * use_solang
    #                 lim_ang_sep = np.sqrt(use_solang/np.pi)
    #                 mag_stars_recut = mag_stars[mag_stars['ang_sep']<lim_ang_sep].copy()
    #             mag_stars = mag_stars_recut
    #             mag_stars.to_hdf(out_file, key='data')
    #             with h5py.File(out_file, 'a') as hf:
    #                 if 'solid_angle' in hf:
    #                     del hf['solid_angle']
    #                 dset = hf.create_dataset('solid_angle', data=use_solang, dtype='f')
    #             n_stars = len(mag_stars)
    #             print(f'    Re-saved {cat} catalog with {n_stars} stars and solid_angle={use_solang} deg^2.')
    #     else:
    #         rerun=True
    # if not rerun:
    #     print(f'Not regenerating l={l:0.3f}, b={b:0.3f}.')
    #     return

    df,_ = mod.process_location(l_deg=l, b_deg=b, solid_angle=solang, save_data=False)
    print(f'Generated full catalog for l={l:0.3f}, b={b:0.3f}.')
    for i, cat in enumerate(cat_type):
        out_file = f'roman_dense_grid/{cat}/Huston2025_l{l:2.3f}_b{b:2.3f}.h5'
        # Filter by magnitude for given catalog type
        if maglim_dim[i] is not None:
            not_too_dim = (df['W146']<=maglim_dim[i])
        else:
            not_too_dim = df['iMass']>0
        if maglim_bright[i] is not None:
            not_too_bright = (df['W146']>maglim_bright[i])
        else:
            not_too_bright = df['iMass']>0
        mag_stars = df[not_too_dim & not_too_bright].copy()
        use_solang = solang
        # Trim catalog if needed
        if len(mag_stars)>(goal_count[i]*1.2):
            use_solang = goal_count[i]/len(mag_stars) * solang
            mag_stars.loc[:,'l_sep'] = (mag_stars['l'] - l)
            mag_stars.loc[:,'l_sep'] -= (mag_stars['l_sep']>180)*360
            mag_stars.loc[:,'l_sep'] *= np.cos(b*np.pi/180)
            mag_stars.loc[:,'b_sep'] = (mag_stars['b'] - b)
            mag_stars.loc[:,'ang_sep'] = np.hypot(mag_stars['l_sep'], mag_stars['b_sep'])
            lim_ang_sep = np.sqrt(use_solang/np.pi)
            mag_stars_recut = mag_stars[mag_stars['ang_sep']<lim_ang_sep].copy()
            while (len(mag_stars_recut)<(goal_count[i]*0.8)) or \
                    (len(mag_stars_recut)>(goal_count[i]*1.2)):
                use_solang = goal_count[i]/len(mag_stars_recut) * use_solang
                lim_ang_sep = np.sqrt(use_solang/np.pi)
                mag_stars_recut = mag_stars[mag_stars['ang_sep']<lim_ang_sep].copy()
            mag_stars = mag_stars_recut
        # Save to file, along with solid angle used
        mag_stars.to_hdf(out_file, key='data')
        with h5py.File(out_file, 'a') as hf:
            if 'solid_angle' in hf:
                del hf['solid_angle']
            dset = hf.create_dataset('solid_angle', data=use_solang, dtype='f')
        n_stars = len(mag_stars)
        print(f'    Saved {cat} catalog with {n_stars} stars and solid_angle={use_solang} deg^2.')


#flds = pd.read_csv('overguide_chip_centers.dat', sep='\s+')
mod = sp.SynthPop('huston2025_defaults.synthpop_conf',
                  maglim=["W146", 99, "keep"],
                  #skip_lowmass_stars=True,
                  post_processing_kwargs=[{"name":"ProcessDarkCompactObjects", "remove":False}, 
                        {"name":"ConvertMistMags", "conversions":{"AB": ["R062", "Z087", "Y106", "J129", "W146", "H158", "F184"]}},
                        {"name": "EstimateRomanExtinction", "mag_sys":"AB"}],
                  name_for_output="Huston2025",
                  output_location="roman_dense_grid/tmp",
                  output_file_type="h5"
                 )
mod.init_populations()


for b in np.arange(-2.0, -0.70+0.01, 0.05):
    for l in np.arange(-1.00, 1.80+0.01, 0.05):
        run_field(l,b)
for b in np.arange(-0.70+0.05, 0.5+0.01, 0.05):
    for l in np.arange(-0.5, 0.5+0.01, 0.05):
        run_field(l,b)
