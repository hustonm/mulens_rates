import numpy as np
import synthpop as sp
import pandas as pd

ulim = 30000
llim = 20000
alim = 25000

flds = pd.read_csv('overguide_chip_centers.dat', sep='\s+')
mod = sp.SynthPop('huston2025_defaults.synthpop_conf',
                  maglim=["W146", 23.975533, "remove"],
                  post_processing_kwargs=[{"name":"ProcessDarkCompactObjects", "remove":True}, 
                        {"name":"ConvertMistMags", "conversions":{"AB": ["R062", "Z087", "Y106", "J129", "W146", "H158", "F184"]}}],
                  name_for_output="Huston2025",
                  output_location="roman_chips/src25",
                  output_file_type="h5"
                 )
mod.init_populations()
solang = 3e-5

for i in flds.index:
    l = flds.l[i]
    b = flds.b[i]
    df1,_ = mod.process_location(l_deg=l, b_deg=b, solid_angle=solang)
    leng = len(df1)
    if leng>ulim or leng<llim:
        print('    length:',leng,", rerunning l=",l,' b=',b)
        solang = solang * alim/leng
        df1,_ = mod.process_location(l_deg=l, b_deg=b, solid_angle=solang)
        leng = len(df1)
    solang = solang * alim/leng