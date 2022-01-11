"""Author: Connor Stone

Calculate galaxy structural parameters.

Functions here will take a galaxy object and compute some structural
parameter from the avaialble profiles. The parameter(s) will then be
added to the galaxy object and returned. Several calculations are
dependent on other calculations preceeding them, this is noted in
their documentation.
"""

from .Supporting_Functions import (
    app_mag_to_abs_mag,
    mag_to_L,
    L_to_mag,
    pc_to_arcsec,
    arcsec_to_pc,
    vcmb_to_z,
    H0,
    c,
    surface_brightness,
    flux_to_mag,
    mag_to_flux,
    Get_M2L,
    inclination,
)

from .Profile_Functions import (
    Evaluate_Magnitude,
    Isophotal_Radius,
    Evaluate_Surface_Brightness,
    Effective_Radius,
    bulgefit,
    Courteau97_Model_Fit,
    Courteau97_Model_Evaluate,
    Tan_Model_Evaluate,
    Tan_Model_Fit,
)    

from .K_correction import calc_kcor
from scipy.integrate import quad, trapz
import matplotlib.pyplot as plt
import logging
import numpy as np

def Calc_Apparent_Radius(G, eval_at_R="Ri23.5", eval_at_band="r"):
    """
    Compute the apparent radius for a given requested radius and band.

    references: None
    """
    if not "appR" in G:
        G["appR"] = {}

    if eval_at_R == "Rinf":
        Reval = [np.inf, 0.0]
    elif eval_at_R[:2] == "Ri":
        Reval = Isophotal_Radius(
            G["SB"][eval_at_band]["R"],
            G["SB"][eval_at_band]["sb"],
            float(eval_at_R[2:]),
            E=G["SB"][eval_at_band]["sb E"],
        )
    elif eval_at_R[:2] == "Re":
        Reval = Effective_Radius(
            G["SB"][eval_at_band]["R"],
            G["SB"][eval_at_band]["m"],
            E=G["SB"][eval_at_band]["m E"],
            ratio=float(eval_at_R[2:]) / 100,
        )
    else:
        raise ValueError(f"unrecognized evaluation radius: {eval_at_R}")

    G["appR"][f"{eval_at_R}:{eval_at_band}"] = Reval[0]
    G["appR"][f"E:{eval_at_R}:{eval_at_band}"] = Reval[1]

    return G


def Calc_Physical_Radius(G, eval_at_R="Ri23.5", eval_at_band="r"):
    """
    Compute the physical radius for a given requested radius and band.

    references: Calc_Apparent_Radius
    """
    if not "physR" in G:
        G["physR"] = {}

    Rphys = arcsec_to_pc(
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        G["D"],
        thetae=G["appR"][f"E:{eval_at_R}:{eval_at_band}"],
        De=G["D E"],
    )

    G["physR"][f"{eval_at_R}:{eval_at_band}"] = Rphys[0] / 1e3
    G["physR"][f"E:{eval_at_R}:{eval_at_band}"] = Rphys[1] / 1e3

    return G


def Calc_Axis_Ratio(G, eval_at_R="Ri23.5", eval_at_band="r"):
    """
    Interpolate the isophote axis ratio at a requested radius and band.

    references: Calc_Apparent_Radius
    """
    if not "q" in G:
        G["q"] = {}

    if G["appR"][f"{eval_at_R}:{eval_at_band}"] < G['SB'][eval_at_band]['R'][-1]:
        q = np.interp(
            G["appR"][f"{eval_at_R}:{eval_at_band}"],
            G["SB"][eval_at_band]["R"],
            G["SB"][eval_at_band]["q"],
        )
    else:
        q = np.median(G["SB"][eval_at_band]["q"][-5:])

    G["q"][f"{eval_at_R}:{eval_at_band}"] = q
    G["q"][f"E:{eval_at_R}:{eval_at_band}"] = max(
        0.05,
        np.std(G["SB"][eval_at_band]["q"][-5:]),
    )

    return G


def Calc_Inclination(G, eval_at_R="Ri23.5", eval_at_band="r"):
    """Compute the inclination at a requested radius and band. Accounting
    for intrinsic thickness of the disk.

    references: Calc_Axis_Ratio
    """
    if not "i" in G:
        G["i"] = {}

    G["i"][f"{eval_at_R}:{eval_at_band}"] = inclination(
        G["q"][f"{eval_at_R}:{eval_at_band}"], G["q0"]
    )
    G["i"][f"E:{eval_at_R}:{eval_at_band}"] = 5 * np.pi / 180  # fixme

    return G


def Calc_C97_Velocity_Fit(G):
    if not "C97 Model" in G:
        G["C97 Model"] = {}

    fixed_origin = False
    if 'fixed origin' in G['RC']:
        fixed_origin = G['RC']['fixed origin']
    x = Courteau97_Model_Fit(G['RC']['R'], G['RC']['v'], G['RC']['v E'], fixed_origin = fixed_origin)

    G['C97 Model']['param order'] = ['r_t', 'v_c', 'beta', 'gamma', 'v0', 'x0']
    G['C97 Model']['r_t'] = x[0]
    G['C97 Model']['v_c'] = x[1]
    G['C97 Model']['beta'] = x[2]
    G['C97 Model']['gamma'] = x[3]
    G['C97 Model']['v0'] = x[4]
    G['C97 Model']['x0'] = x[5]
        
    return G

def Calc_Tan_Velocity_Fit(G):
    if not "Tan Model" in G:
        G["Tan Model"] = {}

    fixed_origin = False
    if 'fixed origin' in G['RC']:
        fixed_origin = G['RC']['fixed origin']
    x = Tan_Model_Fit(G['RC']['R'], G['RC']['v'], G['RC']['v E'], fixed_origin = fixed_origin)

    G['Tan Model']['param order'] = ['r_t', 'v_c', 'v0', 'x0']
    G['Tan Model']['r_t'] = x[0]
    G['Tan Model']['v_c'] = x[1]
    G['Tan Model']['v0'] = x[2]
    G['Tan Model']['x0'] = x[3]
        
    return G

def Calc_Velocity(G, eval_at_R="Ri23.5", eval_at_band="r", eval_with_model = 'C97 Model'):
    """
    Compute the rotation velocity at a requested radius and band.

    references: Calc_C97_Velocity_Fit, Calc_Apparent_Radius
    """
    if not "V" in G:
        G["V"] = {}

    x = list(G[eval_with_model][p] for p in G[eval_with_model]['param order'])
    if eval_with_model == 'C97 Model':
        model = Courteau97_Model_Evaluate
    elif eval_with_model == 'Tan Model':
        model = Tan_Model_Evaluate
    else:
        raise ValueError(f'unrecognized model type {eval_with_model}')
    Vobs = model(
        x[:-2] + [0, 0], G["appR"][f"{eval_at_R}:{eval_at_band}"]
    )
    incl_corr = np.sin(G["i"][f"{eval_at_R}:{eval_at_band}"])
    Vcorr = abs(Vobs) / incl_corr
    Vcorr_E = G["RC"]["v E"][
        np.argmin(
            np.abs(np.abs(G["RC"]["R"]) - G["appR"][f"{eval_at_R}:{eval_at_band}"])
        )
    ]

    G["V"][f"{eval_at_R}:{eval_at_band}"] = Vcorr
    G["V"][f"E:{eval_at_R}:{eval_at_band}"] = Vcorr_E

    return G


def Calc_Apparent_Magnitude(G, eval_at_R="Ri23.5", eval_at_band="r"):
    """
    Compute the total apparent magnitude within a requested radius and at a requested band.

    references: Calc_Apparent_Radius
    """
    if not "appMag" in G:
        G["appMag"] = {}

    # if extrapolating to infinity, do evaluations at R23.5 and
    # extrapolate at the end
    if eval_at_R == "Rinf":
        eval_at_R = "Ri23.5"
        to_inf = True
    else:
        to_inf = False

    sbR = G["SB"][eval_at_band]["R"]
    sbSB = G["SB"][eval_at_band]["sb"]

    Mag = Evaluate_Magnitude(
        sbR,
        G["SB"][eval_at_band]["m"],
        G["appR"][f"{eval_at_R}:{eval_at_band}"],
        E=G["SB"][eval_at_band]["m E"],
    )

    # add extrapolation if integrating to infinity
    if to_inf:
        p = np.polyfit(
            sbR[np.logical_and(sbSB > 23, sbSB < 26)],
            sbSB[np.logical_and(sbSB > 23, sbSB < 26)],
            1,
        )
        pprime = (-p[0] / 2.5, (22.5 - p[1]) / 2.5)
        prefactor = -2 * np.pi * G["q"][f"{eval_at_R}:{eval_at_band}"]
        L_inf = (
            prefactor
            * (
                (
                    10
                    ** (
                        pprime[0] * G["appR"][f"{eval_at_R}:{eval_at_band}"] + pprime[1]
                    )
                )
                * (
                    pprime[0] * G["appR"][f"{eval_at_R}:{eval_at_band}"] * np.log(10)
                    - 1
                )
                / ((pprime[0] ** 2) * (np.log(10) ** 2))
            )
        )
        Mag = [flux_to_mag(L_inf + mag_to_flux(Mag[0], 22.5), 22.5), Mag[1]]
        eval_at_R = "Rinf"

    G["appMag"][f"{eval_at_R}:{eval_at_band}"] = Mag[0]
    G["appMag"][f"E:{eval_at_R}:{eval_at_band}"] = Mag[1]

    return G


def Calc_Absolute_Magnitude(G, eval_at_R="Ri23.5", eval_at_band="r"):
    """
    Compute the total absolute magnitude within a requested radius and at a requested band.

    references: Calc_Apparent_Magnitude (at multiple bands for K-correction)
    """

    if not "absMag" in G:
        G["absMag"] = {}

    Mag = app_mag_to_abs_mag(
        G["appMag"][f"{eval_at_R}:{eval_at_band}"],
        G["D"],
        G["appMag"][f"E:{eval_at_R}:{eval_at_band}"],
        G["D E"],
    )

    # determine K-correction
    try:
        if eval_at_band == "g":
            kcorr = calc_kcor(
                eval_at_band,
                G["zhel"],
                "g - z",
                G["appMag"][f"{eval_at_R}:g"] - G["appMag"][f"{eval_at_R}:z"],
            )
        else:
            kcorr = calc_kcor(
                eval_at_band,
                G["zhel"],
                "g - %s" % eval_at_band,
                G["appMag"][f"{eval_at_R}:g"]
                - G["appMag"][f"{eval_at_R}:{eval_at_band}"],
            )
        Mag = [Mag[0] + kcorr, Mag[1]]
    except KeyError:
        pass

    G["absMag"][f"{eval_at_R}:{eval_at_band}"] = Mag[0]
    G["absMag"][f"E:{eval_at_R}:{eval_at_band}"] = Mag[1]

    return G


def Calc_Luminosity(G, eval_at_R="Ri23.5", eval_at_band="r"):
    """
    Compute the total luminosity within a requested radius and at a requested band.

    references: Calc_Absolute_Magnitude
    """

    if not "L" in G:
        G["L"] = {}

    Lum = mag_to_L(
        G["absMag"][f"{eval_at_R}:{eval_at_band}"],
        eval_at_band,
        mage=G["absMag"][f"E:{eval_at_R}:{eval_at_band}"],
    )

    G["L"][f"{eval_at_R}:{eval_at_band}"] = Lum[0]
    G["L"][f"E:{eval_at_R}:{eval_at_band}"] = Lum[1]

    return G


def Calc_Colour(G, eval_at_R="Ri23.5", eval_at_band1="r", eval_at_band2="g"):
    """
    Compute the colour within a requested radius and at a requested pair of bands.

    references: Calc_Apparent_Magnitude
    """
    if not "Col" in G:
        G["Col"] = {}

    col = (
        G["appMag"]["%s:%s" % (eval_at_R, eval_at_band2)]
        - G["appMag"]["%s:%s" % (eval_at_R, eval_at_band1)]
    )
    colE = np.sqrt(
        G["appMag"]["E:%s:%s" % (eval_at_R, eval_at_band1)] ** 2
        + G["appMag"]["E:%s:%s" % (eval_at_R, eval_at_band2)] ** 2
    )  # fixme check

    G["Col"]["%s:%s:%s" % (eval_at_R, eval_at_band1, eval_at_band2)] = col
    G["Col"]["E:%s:%s:%s" % (eval_at_R, eval_at_band1, eval_at_band2)] = colE

    return G


def Calc_Mass_to_Light(G, eval_at_R="Ri23.5", eval_at_band1="r", eval_at_band2="g"):
    """
    Compute the colour based mass-to-light-ratio within a requested radius and at a requested pair of bands.

    references: Calc_Colour
    """

    if not "M2L" in G:
        G["M2L"] = {}

    M2L = Get_M2L(
        G["Col"]["%s:%s:%s" % (eval_at_R, eval_at_band1, eval_at_band2)],
        "%s-%s" % (eval_at_band2, eval_at_band1),
        eval_at_band1,
        colour_err=G["Col"]["E:%s:%s:%s" % (eval_at_R, eval_at_band1, eval_at_band2)],
    )

    G["M2L"]["%s:%s:%s" % (eval_at_R, eval_at_band1, eval_at_band2)] = M2L[0]
    G["M2L"]["E:%s:%s:%s" % (eval_at_R, eval_at_band1, eval_at_band2)] = M2L[1]

    return G


def Calc_Concentration(G, eval_at_R1="Re20", eval_at_R2="Re80", eval_at_band="r"):
    """
    Compute the light concentration for a requested pair of radii and at a requested band.

    references: Calc_Apparent_Radius
    """

    if not "Con" in G:
        G["Con"] = {}

    Con = 5 * np.log10(
        G["appR"]["%s:%s" % (eval_at_R2, eval_at_band)]
        / G["appR"]["%s:%s" % (eval_at_R1, eval_at_band)]
    )
    Con_E = 5 * np.sqrt(
        G["appR"]["E:%s:%s" % (eval_at_R1, eval_at_band)] ** 2
        + G["appR"]["E:%s:%s" % (eval_at_R2, eval_at_band)] ** 2
    )

    G["Con"]["%s:%s:%s" % (eval_at_R1, eval_at_R2, eval_at_band)] = Con
    G["Con"]["E:%s:%s:%s" % (eval_at_R1, eval_at_R2, eval_at_band)] = Con_E

    return G


def Calc_Sersic_Params(G, eval_at_band="r"):
    """
    Compute the sersic index for a light profile of a given band.

    references: None
    """

    if not "sersic" in G:
        G["sersic"] = {}

    sbR = G["SB"][eval_at_band]["R"]
    sbSB = G["SB"][eval_at_band]["sb"]
    S = bulgefit(
        {
            "R": G["SB"][eval_at_band]["R"],
            "SB": G["SB"][eval_at_band]["sb"],
            "SB_e": G["SB"][eval_at_band]["sb E"],
        },
        1 - np.median(G["SB"][eval_at_band]["q"][-5:]),
    )
    for key in S:
        G["sersic"][f"{key[8:]}:{eval_at_band}"] = S[key]
        G["sersic"][f"E:{key[8:]}:{eval_at_band}"] = 0.0  # fixme error

    return G


def Calc_Surface_Density(G, eval_at_R="Ri23.5", eval_at_band="r"):
    """
    Compute the light surface density within a requested radius and at a requested band.

    references: Calc_Apparent_Radius, Calc_Apparent_Magnitude
    """

    if not "SurfDen" in G:
        G["SurfDen"] = {}

    A = (
        G["appR"][f"{eval_at_R}:{eval_at_band}"] ** 2
    ) 
    Ae1 = (
        G["q"][f"E:{eval_at_R}:{eval_at_band}"]
        * A
        / G["q"][f"{eval_at_R}:{eval_at_band}"]
    )
    Ae2 = (
        2
        * G["appR"][f"E:{eval_at_R}:{eval_at_band}"]
        * A
        / G["appR"][f"{eval_at_R}:{eval_at_band}"]
    )
    SB = surface_brightness(
        m=G["appMag"][f"{eval_at_R}:{eval_at_band}"],
        A=A,
        me=G["appMag"][f"E:{eval_at_R}:{eval_at_band}"],
        Ae=np.sqrt(Ae1 ** 2 + Ae2 ** 2),
    )

    G["SurfDen"][f"{eval_at_R}:{eval_at_band}"] = SB[0]
    G["SurfDen"][f"E:{eval_at_R}:{eval_at_band}"] = SB[1]

    return G
