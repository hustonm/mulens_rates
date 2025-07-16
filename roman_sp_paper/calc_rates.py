import numpy as np
from mulens_rates import microlensing_calculations
import pandas as pd

chips = pd.read_csv('overguide_chip_centers.dat', sep='\s+')
chip_side = 0.125 # chip width in degrees

data_list = []
for i in chips.index[:2]:
    l = chips.l[i]
    b = chips.b[i]
    chip_id = f'{chips.Nfield[i]}'+f'{chips.Nchip[i]}'.zfill(2)
    print(chip_id)
    f_lens = f'roman_chips/lens/Huston2025_l{l:2.3f}_b{b:2.3f}.h5'
    f_src = f'roman_chips/source/Huston2025_l{l:2.3f}_b{b:2.3f}.h5'
    dat,output_cols = microlensing_calculations.mulens_stats(l, b, f_lens, f_src, nsd=True, roman_blue=True, field_id=chip_id)
    data_list.append(dat)
    print(*dat)
output = pd.DataFrame(data=data_list, columns=output_cols)
output.to_csv('mulens_rates_test.txt', index=False)

output_gulls = microlensing_calculations.prep_rates_for_gulls(output, chip_side=0.125)
output_gulls.to_csv('rates_gulls_test.txt', index=False, sep=' ')
