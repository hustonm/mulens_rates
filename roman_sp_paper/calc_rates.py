import numpy as np
from mulens_rates import microlensing_calculations
import pandas as pd

chips = pd.read_csv('overguide_chip_centers.dat', sep='\s+')
chip_side = 0.125 # chip width in degrees

data_list = []
for i in chips.index:
    l = chips.l[i]
    b = chips.b[i]
    chip_id = f'{chips.Nfield[i]}'+f'{chips.Nchip[i]}'.zfill(2)
    print(chip_id)
    f_lens = f'roman_chips/lens/Huston2025_l{l:2.3f}_b{b:2.3f}.h5'
    f_src = f'roman_chips/source/Huston2025_l{l:2.3f}_b{b:2.3f}.h5'
    dat,output_cols = microlensing_calculations.mulens_stats(l, b, f_lens, f_src)
    new_dats = [f_lens, f_src, chip_id]
    data_list.append(dat+new_dats)
    print(*dat)
new_cols = ['f_lens', 'f_src', 'chip_id']
output = pd.DataFrame(data=data_list, columns=output_cols+new_cols)
output.to_csv('mulens_rates_gulls_v0.txt', index=False)