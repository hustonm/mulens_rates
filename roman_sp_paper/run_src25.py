import numpy as np
import synthpop as sp
import pandas as pd

def run_field(l,b, solang, max_solang=np.pi*0.05**2):
    df1,_ = mod.process_location(l_deg=l, b_deg=b, solid_angle=solang)
    leng = len(df1)
    if (leng>ulim or leng<llim) and (solang<max_solang):
        print('    length:',leng,", rerunning l=",l,' b=',b)
        solang = np.minimum(solang * alim/leng, max_solang)
        df1,_ = mod.process_location(l_deg=l, b_deg=b, solid_angle=solang)
        leng = len(df1)
    solang = np.minimum(solang * alim/leng, max_solang)
    return solang

ulim = 30000
llim = 20000
alim = 25000

#flds = pd.read_csv('overguide_chip_centers.dat', sep='\s+')
mod = sp.SynthPop('huston2025_defaults.synthpop_conf',
                  maglim=["W146", 23.975533+10, "remove"],
                  skip_lowmass_stars=True,
                  post_processing_kwargs=[{"name":"ProcessDarkCompactObjects", "remove":True}, 
                        {"name":"ConvertMistMags", "conversions":{"AB": ["R062", "Z087", "Y106", "J129", "W146", "H158", "F184"]}},
                        {"name": "CorrectF146Extinction", "mag_sys":"AB"},
                        {"name": "AdditionalCuts", "standard_cuts":[['W146', 'max', 25]]}],
                  name_for_output="Huston2025",
                  output_location="roman_wide_grid/src",
                  output_file_type="h5"
                 )
mod.init_populations()
solang = 3e-5

for b in np.arange(-4.0, 4.0+0.01, 0.1):
    for l in np.arange(-4.00, 4.0+0.01, 0.1):
        solang = run_field(l,b,solang)
