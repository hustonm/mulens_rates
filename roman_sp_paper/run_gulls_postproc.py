import numpy as np
import synthpop as sp
import pandas as pd
import shutil

flds = pd.read_csv('overguide_chip_centers.dat', sep='\s+')
set_name='mid2'
dir_orig = 'roman_chips/'+set_name
dir_new = 'roman_chips_gulls/'+set_name

mod = sp.SynthPop('huston2025_defaults.synthpop_conf',
                  maglim=["W146", 23.975533+2, "remove"],
                  post_processing_kwargs={"name":"GullsPostProcessing", "cat_type":set_name},
                  name_for_output="Huston2025",
                  output_location="roman_chips/IGNORE",
                  output_file_type="csv"
                 )
mod.init_populations()

#postproc = sp.modules.post_processing.gulls_post_processing.GullsPostProcessing(mod, set_name)

for i in flds.index:
    l = flds.l[i]
    b = flds.b[i]
    f_orig = dir_orig+'/Huston2025_l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.h5'
    f_orig_log = dir_orig+'/Huston2025_l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.log'

    f_new = dir_new+'/Huston2025_l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.csv'
    f_new_log = dir_new+'/Huston2025_l'+f'{l:2.3f}'+'_b'+f'{b:2.3f}'+'.log'

    shutil.copyfile(f_orig_log, f_new_log, follow_symlinks=True)
    dat = pd.read_hdf(f_orig, key='data')
    dat_new = mod.post_processing(dat)
    dat_new.to_csv(f_new,sep=' ', index=False)
    print(f'processed l={l:2.3f}, b={b:2.3f}')
