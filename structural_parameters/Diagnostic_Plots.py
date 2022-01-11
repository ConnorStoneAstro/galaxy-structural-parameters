import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from .Profile_Functions import (
    Courteau97_Model_Evaluate,
    Tan_Model_Evaluate,    
)
band_colors = {'U': 'tab:pink', 'B': 'tab:blue', 'V': 'tab:purple', 
               'u': 'm', 'g': 'tab:green', 'r': 'tab:red', 'i': 'maroon', 'z': 'k'}

def Plot_Photometry(G):

    if not bool(G['plot'].get('all') or G['plot'].get('photometry')):
        return G
    
    for b in G['SB']:
        colour = band_colors[b] if b in band_colors else None
        plt.errorbar(G['SB'][b]['R'], G['SB'][b]['sb'], yerr = G['SB'][b]['sb E'], color = colour, markersize = 6, marker = '.', linewidth = 0, elinewidth = 1, label = f'SB {b}')
    plt.legend()
    plt.xlabel('Radius [arcsec]')
    plt.ylabel('Surface Brightness [mag arcsec$^{-2}$]')
    plt.gca().invert_yaxis()
    plt.savefig(f"Plot_Plotometry_SB_{G['name']}.jpg")
    plt.close()

    for b in G["SB"]:
        colour = band_colors[b] if b in band_colors else None
        plt.scatter(G['SB'][b]['R'], G['SB'][b]['m'], label = f'mag {b}', color = colour, s = 6)
    plt.legend()
    plt.xlabel('Radius [arcsec]')
    plt.ylabel('Magnitude [mag]')
    plt.gca().invert_yaxis()
    plt.savefig(f"Plot_Plotometry_mag_{G['name']}.jpg")
    plt.close()

    return G

def Plot_Radii(G):

    if not bool(G['plot'].get('all') or G['plot'].get('radii')):
        return G
    
    clist = list(mcolors.TABLEAU_COLORS.values())
    appRs = list(filter(lambda k: k.split(':')[0] != 'E', G['appR'].keys()))
    for b in G['SB']:
        colour = band_colors[b] if b in band_colors else None
        plt.errorbar(G['SB'][b]['R'], G['SB'][b]['sb'], yerr = G['SB'][b]['sb E'], color = colour, markersize = 6, marker = '.', linewidth = 0, elinewidth = 1, label = f'SB {b}')
        cindex = 0
        for r in sorted(appRs, key = lambda k: float(k.split(':')[0][2:])):
            if r[:2] != 'Ri' or r == 'Rinf' or r.split(':')[1] != b:
                continue
            plt.axvline(G['appR'][r], label = 'R$_{%s}$' % r.split(':')[0][2:], color = clist[cindex % len(clist)])
            cindex += 1
        plt.legend()
        plt.xlabel('Radius [arcsec]')
        plt.ylabel('Surface Brightness [mag arcsec$^{-2}$]')
        plt.gca().invert_yaxis()
        plt.savefig(f"Plot_Radii_SB_{G['name']}_{b}.jpg")
        plt.close()

    for b in G["SB"]:
        colour = band_colors[b] if b in band_colors else None
        plt.scatter(G['SB'][b]['R'], G['SB'][b]['m'], label = f'mag {b}', color = colour, s = 6)
        cindex = 0
        for r in sorted(G['appR']):
            if r[:2] != 'Re' or r.split(':')[1] != b:
                continue
            plt.axvline(G['appR'][r], label = 'R$_{%s}$' % r.split(':')[0][2:], color = clist[cindex % len(clist)])
            cindex += 1
        plt.legend()
        plt.xlabel('Radius [arcsec]')
        plt.ylabel('Magnitude [mag]')
        plt.gca().invert_yaxis()
        plt.savefig(f"Plot_Radii_mag_{G['name']}_{b}.jpg")
        plt.close()
        
    return G

def Plot_Velocity(G):

    if not bool(G['plot'].get('all') or G['plot'].get('velocity')):
        return G

    plt.errorbar(G['RC']['R'], G['RC']['v'], yerr = G['RC']['v E'], color = 'k', markersize = 7, marker = '.', linewidth = 0, elinewidth = 1, label = 'RC data')
    rr = np.linspace(min(G['RC']['R']), max(G['RC']['R']), 1000)
    x = list(G['Tan Model'][p] for p in G['Tan Model']['param order'])
    plt.plot(rr, Tan_Model_Evaluate(x, rr), color = 'tan', label = 'Tan Model', linewidth = 2)
    x = list(G['C97 Model'][p] for p in G['C97 Model']['param order'])
    plt.plot(rr, Courteau97_Model_Evaluate(x, rr), color = 'tab:red', label = 'C97 Model', linewidth = 2)
    plt.legend()
    plt.xlabel('Radius [arcsec]')
    plt.ylabel('Velocity [km s$^{-1}$]')
    plt.savefig(f"Plot_Velocity_{G['name']}.jpg")
    plt.close()

    return G
