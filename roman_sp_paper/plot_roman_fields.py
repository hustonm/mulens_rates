import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

ccyc = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

# Imports
import pysiaf
from pysiaf.utils.rotations import attitude
# Read in the Roman SIAF
rsiaf = pysiaf.Siaf('Roman')
wfi_cen=rsiaf['WFI_CEN']
roman_apertures = [f'WFI{i + 1:02}_FULL' for i in range(18)]


def compile_edges(l,b, pa_v3):
    """
    Compile the outer borders of a Roman FoV for a given l, b and PA,
    enveloping all detectors.
    """
    edge_corners = [[7, 2], [9,1], [9,0], [6,1], [6,0], [3,1], [12,0], [15,1], [15,0],
                    [18,1], [18,0], [16,3], [16,2], [13,3], [13,2], [10,3], [1,2],
                    [4,3], [4,2], [7,3], [7,2]]
    all_corners = []
    coord = SkyCoord(l=l, b=b, unit=u.deg, frame='galactic').transform_to('icrs')
    boresight_ra = coord.ra.degree
    boresight_dec = coord.dec.degree
    att = attitude(wfi_cen.V2Ref, wfi_cen.V3Ref, boresight_ra, boresight_dec, pa_v3)

    for j,rap in enumerate(roman_apertures):
        wfi_ap = rsiaf[rap]
        wfi_ap.set_attitude_matrix(att)
        ra,dec = wfi_ap.closed_polygon_points('sky')
        lb = SkyCoord(ra=ra, dec=dec, unit=u.deg, frame='icrs').transform_to('galactic')
        mid_l = np.mean((lb.l.degree-360*(lb.l.degree>180))[:-1])
        mid_b = np.mean(lb.b.degree[:-1])
        all_corners.append(np.transpose([lb.l.degree-360*(lb.l.degree>180),lb.b.degree]))
    pt_corners = [all_corners[edge_corners[i][0]-1][edge_corners[i][1]] for i in range(len(edge_corners))]
    wfi_cen.set_attitude_matrix(att)
    radec = wfi_cen.idl_to_sky(0, 0)
    lb = SkyCoord(ra=radec[0],dec=radec[1], unit=u.deg, frame='icrs').transform_to('galactic')
    return np.transpose(pt_corners)

# This would also plot detectors but I think I broke it -- let me know if anyone wants it fixed rn
# def plot_fields(plot=plt, season='spring', custom_fields=None, plot_chips=False):
#     if season not in ['spring','fall']:
#         raise ValueError(f"season {season} not found, use spring or fall")
#     if custom_fields is not None:
#         flds = np.
#     frds = SkyCoord(l=flds[:,0], b=flds[:,1], unit=u.deg, frame='galactic').transform_to('icrs')
#     # Set the boresight position and roll angle
#     cent_list = np.zeros((len(frds)*len(roman_apertures),2))
#     field_list = []; chip_list = []
#     k=0
#     for i,frd in enumerate(frds):
#         boresight_ra = frd.ra.degree
#         boresight_dec = frd.dec.degree
#         pa_v3 = 91.5
#         att = attitude(wfi_cen.V2Ref, wfi_cen.V3Ref, boresight_ra, boresight_dec, pa_v3)

#         for j,rap in enumerate(roman_apertures):
#             wfi_ap = rsiaf[rap]
#             wfi_ap.set_attitude_matrix(att)
#             ra,dec = wfi_ap.closed_polygon_points('sky')
#             lb = SkyCoord(ra=ra, dec=dec, unit=u.deg, frame='icrs').transform_to('galactic')
#             mid_l = np.mean((lb.l.degree-360*(lb.l.degree>180))[:-1])
#             mid_b = np.mean(lb.b.degree[:-1])
#             cent_list[k]=[np.round(mid_l,3), np.round(mid_b,3)]
#             field_list.append(i+1), chip_list.append(j+1)
#             plot.plot(lb.l.degree-360*(lb.l.degree>180),lb.b.degree,c=ccyc[i])
#             plot.plot(mid_l,mid_b, '.', c=ccyc[i])
#             k+=1
#         wfi_cen.set_attitude_matrix(att)
#         radec = wfi_cen.idl_to_sky(0, 0)
#         lb = SkyCoord(ra=radec[0],dec=radec[1], unit=u.deg, frame='icrs').transform_to('galactic')
#         plot.plot(*compile_edges(lb.l.degree, lb.b.degree, pa_v3), c='k')
#         plot.plot(lb.l.degree-360*(lb.l.degree>180), lb.b.degree, 'k.', markersize=10)
#     #plot.gca().invert_xaxis()
    
#     df = pd.DataFrame({'Nfield': field_list,
#                    'Nchip': chip_list,
#                    'l': cent_list[:,0],
#                    'b': cent_list[:,1]})
#     return df

def plot_outline(plot=plt, season='spring', custom_fields=None):
    """
    Plot Roman fields outlines for a given season. I had the set Spring and Fall plans
    ready to go, but you can also put in your own file path for an alternate version.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if season not in ['spring','fall']:
        raise ValueError(f"season {season} not found, use spring or fall")
    elif season=='spring':
        flds = np.loadtxt(f"{dir_path}/field_centers_spring.txt", skiprows=1)
        pa_v3 = 90.75
    elif season=='fall':
        flds = np.loadtxt(f"{dir_path}/field_centers_fall.txt", skiprows=1)
        pa_v3 = 270.75
    if custom_fields is not None:
        flds = np.loadtxt(custom_fields, skiprows=1)
        
    frds = SkyCoord(l=flds[:,0], b=flds[:,1], unit=u.deg, frame='galactic').transform_to('icrs')
    # Set the boresight position and roll angle
    cent_list = np.zeros((len(frds)*len(roman_apertures),2))
    field_list = []; chip_list = []
    k=0
    for i,frd in enumerate(frds):
        boresight_ra = frd.ra.degree
        boresight_dec = frd.dec.degree
        att = attitude(wfi_cen.V2Ref, wfi_cen.V3Ref, boresight_ra, boresight_dec, pa_v3)
        wfi_cen.set_attitude_matrix(att)
        radec = wfi_cen.idl_to_sky(0, 0)
        lb = SkyCoord(ra=radec[0],dec=radec[1], unit=u.deg, frame='icrs').transform_to('galactic')
        plot.plot(*compile_edges(lb.l.degree, lb.b.degree, pa_v3), c='white')
