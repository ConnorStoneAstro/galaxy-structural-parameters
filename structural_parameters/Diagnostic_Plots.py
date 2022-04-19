import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from .Profile_Functions import (
    Courteau97_Model_Evaluate,
    Tanh_Model_Evaluate,    
)
from .Decorators import catch_errors, all_bands

band_colors = {'f': 'tab:pink', 'n': 'tab:blue', 
               'U': 'tab:pink', 'B': 'tab:blue', 'V': 'tab:purple', 
               'u': 'm', 'g': 'tab:green', 'r': 'tab:red', 'i': 'maroon', 'z': 'k',
               'w1': 'cyan', 'w2': 'orange'}

@catch_errors
def Plot_Photometry(G):

    if not bool(G['plot'].get('all') or G['plot'].get('photometry')):
        return G
    
    for b in G['photometry']:
        colour = band_colors[b] if b in band_colors else None
        plt.errorbar(G['photometry'][b]['R'], G['photometry'][b]['SB'], yerr = G['photometry'][b]['SB_e'], color = colour, markersize = 6, marker = '.', linewidth = 0, elinewidth = 1, label = f'SB {b}')
    plt.legend()
    plt.xlabel('Radius [arcsec]')
    plt.ylabel('Surface Brightness [mag arcsec$^{-2}$]')
    plt.gca().invert_yaxis()
    if 'saveto' in G['plot']:
        saveto = G['plot']['saveto']
    else:
        saveto = ''
    plt.savefig(f"{saveto}Plot_Photometry_SB_{G['name']}.jpg")
    plt.close()

    for b in G["photometry"]:
        colour = band_colors[b] if b in band_colors else None
        plt.scatter(G['photometry'][b]['R'], G['photometry'][b]['totmag'], label = f'mag {b}', color = colour, s = 6)
    plt.legend()
    plt.xlabel('Radius [arcsec]')
    plt.ylabel('Magnitude [mag]')
    plt.gca().invert_yaxis()
    if 'saveto' in G['plot']:
        saveto = G['plot']['saveto']
    else:
        saveto = ''
    plt.savefig(f"{saveto}Plot_Photometry_mag_{G['name']}.jpg")
    plt.close()

    return G

@catch_errors
def Plot_Radii(G, eval_in_band = None):

    if not bool(G['plot'].get('all') or G['plot'].get('radii')):
        return G
    
    clist = list(mcolors.TABLEAU_COLORS.values())
    appRs = list(filter(lambda k: 'E|' not in k and 'RI' not in k and 'Rlast' not in k, G['appR'].keys()))
    colour = band_colors[eval_in_band] if eval_in_band in band_colors else None
    plt.errorbar(G['photometry'][eval_in_band]['R'], G['photometry'][eval_in_band]['SB'], yerr = G['photometry'][eval_in_band]['SB_e'], color = colour, markersize = 6, marker = '.', linewidth = 0, elinewidth = 1, label = f'SB {eval_in_band}')
    cindex = 0
    for r in sorted(appRs, key = lambda k: float(k.split(':')[0][2:])):
        if r[:2] != 'Ri' or r == 'Rinf' or r.split(':')[1] != eval_in_band:
            continue
        plt.axvline(G['appR'][r], label = 'R$_{%s}$' % r.split(':')[0][2:], color = clist[cindex % len(clist)])
        cindex += 1
    plt.legend()
    plt.xlabel('Radius [arcsec]')
    plt.ylabel('Surface Brightness [mag arcsec$^{-2}$]')
    plt.gca().invert_yaxis()
    if 'saveto' in G['plot']:
        saveto = G['plot']['saveto']
    else:
        saveto = ''
    plt.savefig(f"{saveto}Plot_Radii_SB_{G['name']}_{eval_in_band}.jpg")
    plt.close()

    colour = band_colors[eval_in_band] if eval_in_band in band_colors else None
    plt.scatter(G['photometry'][eval_in_band]['R'], G['photometry'][eval_in_band]['totmag'], label = f'mag {eval_in_band}', color = colour, s = 6)
    cindex = 0
    for r in sorted(appRs):
        if r[:2] != 'Rp' or r.split(':')[1] != eval_in_band:
            continue
        plt.axvline(G['appR'][r], label = 'R$_{%s}$' % r.split(':')[0][2:], color = clist[cindex % len(clist)])
        cindex += 1
    plt.legend()
    plt.xlabel('Radius [arcsec]')
    plt.ylabel('Magnitude [mag]')
    plt.gca().invert_yaxis()
    if 'saveto' in G['plot']:
        saveto = G['plot']['saveto']
    else:
        saveto = ''
    plt.savefig(f"{saveto}Plot_Radii_mag_{G['name']}_{eval_in_band}.jpg")
    plt.close()
        
    return G

def Plot_Velocity(G):

    if not bool(G['plot'].get('all') or G['plot'].get('velocity')):
        return G

    plt.errorbar(G['rotation curve']['R'], G['rotation curve']['V'], yerr = G['rotation curve']['V_e'], color = 'k', markersize = 7, marker = '.', linewidth = 0, elinewidth = 1, label = 'RC data')
    rr = np.linspace(min(G['rotation curve']['R']), max(G['rotation curve']['R']), 1000)
    x = list(G['rc_model'][f"Tanh:{p}"] for p in G['rc_model']['Tanh:param order'])
    plt.plot(rr, Tanh_Model_Evaluate(x, rr), color = 'tan', label = 'Tanh Model', linewidth = 2)
    x = list(G['rc_model'][f"C97:{p}"] for p in G['rc_model']['C97:param order'])
    plt.plot(rr, Courteau97_Model_Evaluate(x, rr), color = 'tab:red', label = 'C97 Model', linewidth = 2)
    plt.legend()
    plt.xlabel('Radius [arcsec]')
    plt.ylabel('Velocity [km s$^{-1}$]')
    if 'saveto' in G['plot']:
        saveto = G['plot']['saveto']
    else:
        saveto = ''
    plt.savefig(f"{saveto}Plot_Velocity_{G['name']}.jpg")
    plt.close()

    return G
