import numpy as np
from mulens_rates import microlensing_calculations
import pandas as pd
import h5py
from multiprocessing.pool import Pool
import time
import pdb

def run_func(chip_id, l,b):
    f_lens = f'/Volumes/g-drive/huston2026_synthpop/extinction_rate_comparison/lens_newext/Huston2025_l{l:2.3f}_b{b:2.3f}.h5'
    f_src = f'/Volumes/g-drive/huston2026_synthpop/extinction_rate_comparison/src_newext/Huston2025_l{l:2.3f}_b{b:2.3f}.h5'
    with h5py.File(f_lens, 'r') as hf:
        sa_lens = hf['solid_angle'][()]
    with h5py.File(f_src, 'r') as hf:
        sa_src = hf['solid_angle'][()]
    dat,output_cols = microlensing_calculations.mulens_stats(l, b, f_lens, f_src, 
                        nsd=True, roman_blue=True, field_id=chip_id,
                        sa_lens=sa_lens, sa_src=sa_src, tE_range=[0.04,np.inf])
    print(f'{l:.2f},{b:.2f}')
    return dat, output_cols

if __name__=='__main__':
    lbs = []
    i=0
    results = []
    for b in np.round(np.arange(-4.0, 4.0+0.01, 0.1),3):
        for l in np.round(np.arange(-4.00, 4.0+0.01, 0.1),3):
            lbs.append((i,l,b))
            i+=1
            #results.append(run_func(*lbs[-1]))

    pool = Pool(processes=4)
    results = pool.starmap(run_func,lbs)

    data_list = []
    for result in results:
        data_list.append(result[0])
    output_cols = results[0][1]
    print(data_list)
    print(output_cols)

    output = pd.DataFrame(data=data_list, columns=output_cols)
    output=output.sort_values(by=['l', 'b'])
    output.to_csv('mulens_rates_wide_newext_0.04tE.txt', index=False)

    # output_gulls = microlensing_calculations.prep_rates_for_gulls(output, chip_side=0.05, remove_dir_layers=1)
    # output_gulls.to_csv('roman_dense_grid_gulls/rates.txt', index=False, sep=' ')


