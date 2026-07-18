import numpy as np
from mulens_rates import microlensing_calculations
import pandas as pd
import h5py
from multiprocessing.pool import Pool
import os


def run_func(chip_id, l,b):
    print(f"Process ID: {os.getpid()} is processing args: ({l:.2f},{b:.2f})")
    f_lens = f'roman_dense_grid/lens/Huston2025_l{l:2.3f}_b{b:2.3f}.h5'
    f_src = f'roman_dense_grid/source/Huston2025_l{l:2.3f}_b{b:2.3f}.h5'
    with h5py.File(f_lens, 'r') as hf:
        sa_lens = hf['solid_angle'][()]
    with h5py.File(f_src, 'r') as hf:
        sa_src = hf['solid_angle'][()]
    dat,output_cols = microlensing_calculations.mulens_stats(l, b, f_lens, f_src, 
                        nsd=True, roman_blue=True, field_id=chip_id,
                        sa_lens=sa_lens, sa_src=sa_src)
    print(*dat[:4])
    return dat, output_cols

if __name__=='__main__':
    lbs = []
    i=0
    for b in np.arange(-2.20, -0.70+0.01, 0.05):
        for l in np.arange(-1.00, 1.80+0.01, 0.05):
            lbs.append((i,l,b))
            i+=1
    for b in np.arange(-0.70+0.05, 0.5+0.01, 0.05):
        for l in np.arange(-0.5, 0.5+0.01, 0.05):
            lbs.append((i,l,b))
            i+=1

    pool = Pool(processes=4)
    results = pool.starmap(run_func,lbs)

    data_list = []
    for result in results:
        data_list.append(result[0])
    output_cols = results[0][1]
    print(data_list)
    print(output_cols)

    output = pd.DataFrame(data=data_list, columns=output_cols)
    output.sort_values(by=['b', 'l'], inplace=True)
    output.to_csv('mulens_rates_dense_grid.txt', index=False)

    # output = pd.read_csv('mulens_rates_dense_grid.txt')
    # output.sort_values(by=['l', 'b'], inplace=True)
    # output.to_csv('mulens_rates_dense_grid_sorted.txt', index=False)

    output_gulls = microlensing_calculations.prep_rates_for_gulls(output, chip_side=0.05, remove_dir_layers=1)
    output_gulls.loc[:,'ID_src'] = np.arange(len(output_gulls))
    output_gulls.to_csv('roman_dense_grid_gulls/rates.txt', index=False, sep=' ')


