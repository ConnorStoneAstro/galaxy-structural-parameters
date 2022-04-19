import numpy as np
from .Profile_Functions import Isophotal_Radius
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .Supporting_Functions import v_to_z, mag_to_flux, flux_to_mag, fluxdens_to_fluxsum
from .Decorators import catch_errors, all_bands
from .K_correction import calc_kcor

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
        if eval_in_band in ['g', 'r', 'z']:
            kcorr = list(
                calc_kcor(
                    eval_in_band,
                    v_to_z(G["redshift_helio"]),
                    "g - z",
                    c,
                )
                for c in np.interp(
                        G["photometry"][eval_in_band]["R"],
                        G["Col_prof"]["g:z"]["R"],
                        np.clip(G["Col_prof"]["g:z"]["col"], a_min = -2, a_max = 3),
                )
            )
        else:
            return G
        G["photometry"][eval_in_band]["SB"] -= np.array(kcorr)
    except Exception as e:
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
                G["Col_prof"]["g:z"]["R"],
                G["Col_prof"]["g:z"]["totcol"],
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

    # SB error limits for the different bands
    if eval_in_band in ['f', 'n']:
        lim = 0.5
    elif eval_in_band in ['g', 'r', 'z', 'w1', 'w2']:
        lim = 0.3
    else:
        lim = 1.
    # Remove failed points and high error points
    CHOOSE = np.logical_and(G["photometry"][eval_in_band]["SB"] < 90, G["photometry"][eval_in_band]["SB_e"] < lim)
    if np.sum(CHOOSE) < 5:
        CHOOSE = G["photometry"][eval_in_band]["SB"] < 90
    # If too few points have reasonable values, kick the profile
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
    if eval_in_band not in ['g', 'r', 'z']: #inner_zone <= 1 or np.sum(outer_CHOOSE) < 15:
        outer_CHOOSE = CHOOSE

    # If all the data is very low S/N, kick out the profile
    if np.all(G["photometry"][eval_in_band]["SB"] > 27):
        del G['photometry'][eval_in_band]
        raise KeyError(f'Low quality profile, cannot determine viable region for {eval_in_band}, kicking')

    # Basic linear floor model
    def linear_floor(x, R, SB):
        return np.mean(
            np.abs(SB - np.clip(x[0] * R + x[1], a_min=None, a_max=x[2]))
        )

    # Try to fit the linear floor model, using every point in the profile as a starting point
    # Select the best fit out of all of them
    best_res = None
    for i in range(2,np.sum(outer_CHOOSE)-1):
        x0 = list(
            np.polyfit(G["photometry"][eval_in_band]["R"][outer_CHOOSE][:i],
                       G["photometry"][eval_in_band]["SB"][outer_CHOOSE][:i],
                       1
            )
        ) + [np.median(G["photometry"][eval_in_band]["SB"][outer_CHOOSE][i:])]
        res = minimize(
            linear_floor,
            x0=x0,
            args=(
                G["photometry"][eval_in_band]["R"][outer_CHOOSE],
                G["photometry"][eval_in_band]["SB"][outer_CHOOSE]
            ),
            method="Nelder-Mead",
        )
        if best_res is None:
            best_res = res
        elif res.success and best_res.fun > res.fun:
            best_res = res

    # If the best fitting results converged, set them as the truncation radius
    # Otherwise try to find a point where errors go too high
    if best_res.success:
        truncR = (best_res.x[2] - best_res.x[1]) / best_res.x[0]
    else:
        truncR = G["photometry"][eval_in_band]["R"][-1]
        for i in range(
                np.argmin(np.abs(G["photometry"][eval_in_band]["R"] - Rstart)), len(G["photometry"][eval_in_band]["R"])
        ):
            if G["photometry"][eval_in_band]["SB"][i] > 90 or G["photometry"][eval_in_band]["SB_e"][i] > lim:
                truncR = G["photometry"][eval_in_band]["R"][i]
                break

    # Apply the truncation, except for f,n bands which have too little S/N
    if not eval_in_band in ['f', 'n']:
        CHOOSE[G["photometry"][eval_in_band]["R"] > truncR] = False
    if np.sum(CHOOSE) < 5:
        CHOOSE = np.logical_and(G["photometry"][eval_in_band]["SB"] < 28, G["photometry"][eval_in_band]["SB_e"] < lim)

    # Remove stragler points that are isolated in the profile outskirts
    for i in range(len(G["photometry"][eval_in_band]["R"])-2,5,-1):
        if CHOOSE[i] and not CHOOSE[i+1] and not CHOOSE[i-1]:
            CHOOSE[i] = False
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

    
