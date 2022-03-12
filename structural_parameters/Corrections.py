import numpy as np
from .Profile_Functions import Isophotal_Radius
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .Supporting_Functions import v_to_z, mag_to_flux, flux_to_mag, fluxdens_to_fluxsum
from .Decorators import catch_errors, all_bands

@catch_errors
def Apply_Cosmological_Dimming(G):

    if not "redshift helio" in G:
        if "warnings" in G:
            G["warnings"].append(
                f"redshift helio not in {G['name']}, no cosmological diming correction applied"
            )
        return G

    G["cosmological dimming corr"] = -2.5 * np.log10((1 + v_to_z(G["redshift_helio"])) ** 3)

    for b in G["photometry"]:
        G["photometry"][b]["SB"] += G["cosmological dimming corr"]
        G["photometry"][b]["totmag"] += G["cosmological dimming corr"]

    return G

@all_bands
@catch_errors
def Apply_Extinction_Correction(G, eval_in_band = None):

    if not "extinction" in G or not eval_in_band in G["extinction"]:
        if "warnings" in G:
            G["warnings"].append(
                f"extinction not in {G['name']} for {eval_in_band}-band, no extinction correction applied"
            )
    else:
        G["photometry"][eval_in_band]["SB"] -= G["extinction"][eval_in_band]
        G["photometry"][eval_in_band]["totmag"] -= G["extinction"][eval_in_band]

    return G

@all_bands
@catch_errors
def Apply_K_Correction(G, eval_in_band = None):
    """
    Applies a k-correction to each band from: Blanton and Roweis 2007

    requires: Calc_Colour_Profile
    Note: must run Calc_Colour_Profile again
    """

    if not "redshift_helio" in G:
        if "warnings" in G:
            G["warnings"].append(
                f"redshift helio not in {G['name']}, no K-correction applied"
            )
        return G
    
    # SB profile correction
    try:
        kcorr = list(
            calc_kcor(
                eval_in_band,
                v_to_z(G["redshift_helio"]),
                "g - z",
                c,
            )
            for c in np.interp(
                G["photometry"][eval_in_band]["R"],
                G["Col_prof"]["z:g"]["R"],
                G["Col_prof"]["z:g"]["col"],
            )
        )
        G["photometry"][eval_in_band]["SB"] -= np.array(kcorr)
    except Exception:
        if "warnings" in G:
            G["warnings"].append(
                f"Could not apply K-correction to {eval_in_band}-band SB profile"
            )

    # Curve of Growth correction
    try:
        kcorr = list(
            calc_kcor(
                eval_in_band,
                v_to_z(G["redshift_helio"]),
                "g - z",
                c,
            )
            for c in np.interp(
                G["photometry"][eval_in_band]["R"],
                G["Col_prof"]["z:g"]["R"],
                G["Col_prof"]["z:g"]["totcol"],
            )
        )
        G["photometry"][eval_in_band]["totmag"] -= np.array(kcorr)
    except Exception:
        if "warnings" in G:
            G["warnings"].append(
                f"Could not apply K-correction to {eval_in_band}-band Curve of Growth"
            )

    return G

@all_bands
@catch_errors
def Apply_Profile_Truncation(G, eval_in_band):

    if eval_in_band in ['f', 'n', 'w1', 'w2']:
        lim = 0.5
    elif eval_in_band in ['g', 'r', 'z']:
        lim = 0.3
    else:
        lim = 1.
    CHOOSE = np.logical_and(G["photometry"][eval_in_band]["SB"] < 90, G["photometry"][eval_in_band]["SB_e"] < lim)
    if np.sum(CHOOSE) < 5:
        CHOOSE = G["photometry"][eval_in_band]["SB"] < 90
    if np.sum(CHOOSE) < 5:
        del G['photometry'][eval_in_band]
        raise KeyError(f'Low quality profile, cannot determine viable region for {eval_in_band}, kicking')
    truncR = G["photometry"][eval_in_band]["R"][-1] * 1.01

    outer_CHOOSE = [0]
    inner_zone = 5
    while np.sum(outer_CHOOSE) < 5 and inner_zone > 1:
        Rstart = Isophotal_Radius(
            G["photometry"][eval_in_band]["R"][CHOOSE],
            G["photometry"][eval_in_band]["SB"][CHOOSE],
            G["photometry"][eval_in_band]["SB"][CHOOSE][0] + inner_zone,
        )[0]
        outer_CHOOSE = np.logical_and(CHOOSE, G["photometry"][eval_in_band]["R"] > Rstart)
        inner_zone -= 1
    if inner_zone <= 1 or np.sum(outer_CHOOSE) < 5:
        outer_CHOOSE = CHOOSE

    if np.all(G["photometry"][eval_in_band]["SB"] > 27):
        del G['photometry'][eval_in_band]
        raise KeyError(f'Low quality profile, cannot determine viable region for {eval_in_band}, kicking')
    
    def linear_floor(x, R, SB):
        return np.mean(
            np.abs(SB - np.clip(x[0] * R + x[1], a_min=None, a_max=x[2]))
        )

    x0 = list(
        np.polyfit(G["photometry"][eval_in_band]["R"][outer_CHOOSE][:5], G["photometry"][eval_in_band]["SB"][outer_CHOOSE][:5], 1)
    ) + [np.median(G["photometry"][eval_in_band]["SB"][outer_CHOOSE][-5:])]
    res = minimize(
        linear_floor,
        x0=x0,
        args=(G["photometry"][eval_in_band]["R"][outer_CHOOSE], G["photometry"][eval_in_band]["SB"][outer_CHOOSE]),
        method="Nelder-Mead",
    )
    if res.success:
        truncR = (res.x[2] - res.x[1]) / res.x[0]
    else:
        truncR = G["photometry"][eval_in_band]["R"][-1]
        for i in range(
            np.argmin(np.abs(G["photometry"][eval_in_band]["R"] - Rstart)), len(G["photometry"][eval_in_band]["R"])
        ):
            if G["photometry"][eval_in_band]["SB"][i] > 90:
                truncR = G["photometry"][eval_in_band]["R"][i]
                break
    if not eval_in_band in ['f', 'n']:
        CHOOSE[G["photometry"][eval_in_band]["R"] > truncR] = False
    if np.sum(CHOOSE) < 5:
        if eval_in_band in ['f', 'n', 'w1', 'w2']:
            CHOOSE = np.logical_and(G["photometry"][eval_in_band]["SB"] < 26, G["photometry"][eval_in_band]["SB_e"] < 1)
        else:
            CHOOSE = np.logical_and(G["photometry"][eval_in_band]["SB"] < 27, G["photometry"][eval_in_band]["SB_e"] < 0.3)
                
    if np.sum(CHOOSE) < 5:
        del G['photometry'][eval_in_band]
        raise KeyError(f'Low quality profile, cannot determine viable region for {eval_in_band}, kicking')
    for k in G["photometry"][eval_in_band]:
        G["photometry"][eval_in_band][k] = G["photometry"][eval_in_band][k][CHOOSE]

    return G

@catch_errors
def Apply_Redshift_Velocity_Correction(G):

    if not "redshift_helio" in G:
        if "warnings" in G:
            G["warnings"].append(
                f"redshift helio not in {G['name']}, no redshift velocity correction applied"
            )
        return G
    redshift_corr = 1 + v_to_z(G["redshift_helio"])
    G["rotation curve"]["V"] /= redshift_corr
    G["rotation curve"]["V_e"] /= redshift_corr

    return G

@catch_errors
def Apply_Inclination_Correction(G, specification = None):
    
    if specification is None:
        return G
    for b in specification:
        correction = specification[b]['alpha'] + specification[b]['gamma']*np.clip(-np.log10(np.cos(np.interp(G['photometry'][b]['R'], G['inclination_prof']['R'], G['inclination_prof']['inclination']))), a_min = 0, a_max = 1)
        G['photometry'][b]['SB'] += correction        
        pretot = G['photometry'][b]['totmag'] + 10
        G['photometry'][b]['totmag'] = flux_to_mag(
            fluxdens_to_fluxsum(
                G['photometry'][b]['R'],
                mag_to_flux(G['photometry'][b]['SB'], 20),
                1 - G['photometry'][b]['ellip'],
            ), 20)

    return G


def Fit_Inclination_Correction(Glist, fit_bands = ['f', 'n', 'g', 'r', 'z'], refband = 'w1', eval_after_R = None, eval_after_band = None):

    spec = dict((b,{}) for b in fit_bands)
    for b in fit_bands:

        all_col = []
        all_incl = []
        for G in Glist:
            if G is None or b not in G['photometry']:
                continue
            if not eval_after_R is None:
                CHOOSE = G['photometry'][b]['R'] > G['appR'][f"{eval_after_R}:{eval_after_band}"]
            else:
                CHOOSE = np.ones(len(G['photometry'][b]['R']),dtype = bool)
            minr = min(G['photometry'][b]['R'][-1], G['photometry'][refband]['R'][-1])
            CHOOSE[G['photometry'][b]['R'] > minr] = False
            if np.sum(CHOOSE) < 5 or len(G['photometry'][refband]['R']) < 5:
                continue
            all_col += list(
                G['photometry'][b]['SB'][CHOOSE] - np.interp(
                    G['photometry'][b]['R'][CHOOSE],
                    G['photometry'][refband]['R'],
                    G['photometry'][refband]['SB'],
            ))
            all_incl += list(np.interp(
                    G['photometry'][b]['R'][CHOOSE],
                    G['inclination_prof']['R'],
                    G['inclination_prof']['inclination'],
            ))

        all_incl = -np.log10(np.cos(all_incl))
        all_col = np.array(all_col)
        lim = -np.log10(np.cos(80 * np.pi/180))
        p = np.polyfit(all_incl[all_incl < lim], all_col[all_incl < lim], 1)

        spec[b]['alpha'] = 0
        spec[b]['gamma'] = -p[0]
        
    return spec

def Decide_Bypass(self, G, check_key):
    """
    Descision node used to bypass the main pipeline thread if a data key is missing from the dictionary.

    """
    if check_key in G:
        return self.forward.name
    else:
        op = list(self.options.keys())
        op.pop(op.index(self.forward.name))
        return op[0]

    
