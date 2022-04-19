import numpy as np
from scipy.integrate import trapz, quad

Grav_C      = 4.302e3            # pc M_sun^-1 (m s^-1)^2
H0          = 70.4               # km s^-1 Mpc^-1
Au_to_Pc    = 4.84814e-6         # pc au^-1
Pc_to_m     = 3.086e16           # m pc^-1
Pc_to_Au    = 206265.            # Au pc^-1
Ly_to_Pc    = 0.306601           # pc ly^-1
Au_to_m     = 1.496e11           # m au^-1
D_sun       = 1.58e-5 * Ly_to_Pc # pc
c           = 299792.458         # km s^-1

# mips.as.arizona.edu/~cnaw/sun.html
# AB magnitude system
# also see: http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
Abs_Mag_Sun = {'f': 17.30,
               'n': 10.16,
               'u': 6.39,
               'g': 5.05,
               'r': 4.61,
               'i': 4.52,
               'z': 4.50,
               'U': 6.33,
               'B': 5.31,
               'V': 4.80,
               'R': 4.60,
               'I': 4.51,
               'J': 4.54,
               'H': 4.66,
               'K': 5.08,
               'w1': 5.92,
               'w2': 6.58,
               'w3': 8.48,
               'w4': 9.88,
               '3.6um': 3.24}

lambda_eff = {'f': 0.1549,
              'n': 0.2304,
              'u': 0.3881,
              'g': 0.4776,
              'r': 0.6374,
              'i': 0.7758,
              'z': 0.9139,
              'U': 0.3694,
              'B': 0.4390,
              'V': 0.5476,
              'R': 0.6492,
              'I': 0.7993,
              'J': 1.2321,
              'H': 1.6424,
              'K': 2.1558,
              'w1': 3.3387,
              'w2': 4.5870,
              'w3': 11.3086,
              'w4': 22.0230}

jansky_zp = {'f': 531.97,
             'n': 809.45,
             'u': 2735.17,
             'g': 4033.18,
             'r': 3137.37,
             'i': 2553.68,
             'z': 2304.99,
             'U': 1868.72,
             'B': 4085.60,
             'V': 3674.73,
             'R': 3080.98,
             'I': 2478.76,
             'J': 1628.84,
             'H': 1053.12,
             'K': 683.04,
             'w1': 314.69,
             'w2': 175.16,
             'w3': 29.79,
             'w4': 8.30}

# http://www.sdss.org/dr12/algorithms/magnitudes/
asinh_softening = {'u': 1.4e-10,
                   'g': 9e-11,
                   'r': 1.2e-10,
                   'i': 1.8e-10,
                   'z': 7.4e-10} 

#https://en.wikipedia.org/wiki/Galaxy_morphological_classification
HubbleTypeConversion_deVacouleurs = {'cE': -6,
                                     'E': -5,
                                     'E+':-4,
                                     'S0-':-3,
                                     'S0':-2,
                                     'S0+':-1,
                                     'S0/a':0,
                                     'Sa':1,
                                     'Sab':2,
                                     'Sb':3,
                                     'Sbc':4,
                                     'Sc':5,
                                     'Scd':6,
                                     'Sd':7,
                                     'Sdm':8,
                                     'Sm':9,
                                     'Im':10} 

HubbleTypeConversion_Hubble = {'E': -4,
                               'S0':-1,
                               'S0/a':0,
                               'Sa':1,
                               'Sa-b':2,
                               'Sb':3,
                               'Sb-c':4,
                               'Sc':7,
                               'Sc-Irr':8,
                               'Sc-Ir':8,
                               'Ir I':10,
                               'Ir':10,
                               'Irr':10,
                               'Irr I':10}

def ConvertHubbleType(HT, HT_type = None):
    special = {'Sb/Sc':4, 'S':1}
    not_needed = 'BpLN?():'
    for c in not_needed:
        HT = HT.replace(c,'')
    HT = HT.strip(' \n')
        
    if HT in special:
        return special[HT]
    
    if HT_type == 'deVacouleurs':
        try:
            return HubbleTypeConversion_deVacouleurs[HT]
        except:
            if 'D' in HT:
                return 11
            else:
                print('unidentified Hubble Type!: ', HT)
                return 12
    elif HT_type == 'Hubble':
        try:
            return HubbleTypeConversion_Hubble[HT]
        except:
            if 'D' in HT:
                return 11
            elif 'Ir' in HT:
                return 10
            else:
                print('unidentified Hubble Type!: ', HT)
                return 12
    else:
        try:
            return HubbleTypeConversion_deVacouleurs[HT]
        except:
            pass
        try:
            return HubbleTypeConversion_Hubble[HT]
        except:
            pass
        if 'Ir' in HT:
            return 10
        if 'D' in HT:
            if 'E' in HT:
                return -8
            if 'S' in HT:
                return 11
        print('unidentified Hubble Type!: ' + HT)
        return 12
        

allradii = list(f"Ri{rr:g}" for rr in np.arange(22, 26.5, 0.5)) + \
           list(f"Rp{rr:g}" for rr in np.arange(20, 90, 10)) + \
           list(f"Re{rr:g}" for rr in np.arange(1.5, 4.5, 0.5)) + \
           ['RI']
specialradii = list((f"R*{rr:g}", "*") for rr in np.arange(20, 90, 10)) +\
               list((f"Rd{rr:g}",'*') for rr in [500,100,50,10,5,1]) +\
               [('Rlast', 'rc')]
    
def mag_to_L(mag, band, mage = None, zeropoint = None):
    """
    Returns the luminosity (in solar luminosities) given the absolute magnitude and reference point.
    mag: Absolute magnitude
    band: Photometric band in which measurements were taken
    mage: uncertainty in absolute magnitude
    zeropoint: user defined zero point
    returns: Luminosity in solar luminosities
    """

    L = 10**(((Abs_Mag_Sun[band] if zeropoint is None else zeropoint) - mag)/2.5)
    if mage is None:
        return L
    else:
        Le = np.abs(L * mage * np.log(10) / 2.5)
        return L, Le

def L_to_mag(L, band, Le = None, zeropoint = None):
    """
    Returns the Absolute magnitude of a star given its luminosity and distance
    L: Luminosity in solar luminosities
    band: Photometric band in which measurements were taken
    Le: Uncertainty in luminosity
    zeropoint: user defined zero point
    
    returns: Absolute magnitude
    """

    mag = (Abs_Mag_Sun[band]if zeropoint is None else zeropoint) - 2.5 * np.log10(L)
    
    if Le is None:
        return mag
    else:
        mage = np.abs(2.5 * Le / (L * np.log(10)))
        return mag, mage

def app_mag_to_abs_mag(m, D, me = 0., De = 0.):
    """
    Converts an apparent magnitude to an absolute magnitude
    m: Apparent magnitude
    D: Distance to object in parsecs
    returns: Absolute magnitude at 10 parcecs
    """

    M = m - 5.0 * np.log10(D / 10.0)
    if np.all(me == 0) and np.all(De == 0):
        return M
    else:
        return M, np.sqrt(me**2 + (5. * De / (D * np.log(10)))**2)

def abs_mag_to_app_mag(M, D, Me = 0., De = 0.):
    """
    Converts an absolute magnitude to an apparent magnitude
    M: Absolute magnitude at 10 parcecs
    D: Distance to object in parsecs
    returns: Apparent magnitude
    """

    m = M + 5.0 * np.log10(D / 10.0)
    if np.all(Me == 0) and np.all(De == 0):
        return m
    else:
        return m, np.sqrt(Me**2 + (5. * De / (D * np.log(10)))**2)

def magperarcsec2_to_mag(mu, a = None, b = None, A = None):
    """
    Converts mag/arcsec^2 to mag
    mu: mag/arcsec^2
    a: semi major axis radius (arcsec)
    b: semi minor axis radius (arcsec)
    A: pre-calculated area (arcsec^2)
    returns: mag
    """
    assert (not A is None) or (not a is None and not b is None)
    if A is None:
        A = np.pi * a * b
    return mu - 2.5*np.log10(A) # https://en.wikipedia.org/wiki/Surface_brightness#Calculating_surface_brightness

def mag_to_magperarcsec2(m, a = None, b = None, R = None, A = None):
    """
    Converts mag to mag/arcsec^2
    m: mag
    a: semi major axis radius (arcsec)
    b: semi minor axis radius (arcsec)
    A: pre-calculated area (arcsec^2)
    returns: mag/arcsec^2
    """
    assert (not A is None) or (not a is None and not b is None) or (not R is None)
    if not R is None:
        A = np.pi * (R**2)
    elif A is None:
        A = np.pi * a * b
    return m + 2.5*np.log10(A) # https://en.wikipedia.org/wiki/Surface_brightness#Calculating_surface_brightness

def surface_brightness(m, A, me = None, Ae = None):
    """
    Calculate a surface brightness given the magnitude within an area and the size of the area.

    m: magnitude in region
    A: area of region
    me: error on magnitude m
    Ae: error on area A
    returns mag/arcsec^2
    """
    if me is None and Ae is None:
        return m + 2.5*np.log10(A)
    else:
        return m + 2.5*np.log10(A), np.sqrt(me**2 + (2.5*Ae/(A*np.log(10)))**2)
    
def magnitude(sb, A, sbe = None, Ae = None):
    """
    Calculate the magnitude within an area given its surface brightness.

    sb: surface brightness in region
    A: area of region
    sbe: error on surface brightness sb
    Ae: error on area A
    returns mag
    """
    
    if sbe is None and Ae is None:
        return sb - 2.5*np.log10(A)
    else:
        return sb - 2.5*np.log10(A), np.sqrt(sbe**2 + (2.5*Ae/(A*np.log(10)))**2)
        
    
def halfmag(mag):
    """
    Computes the magnitude corresponding to half in log space.
    Effectively, converts to luminosity, divides by 2, then
    converts back to magnitude. Distance is not needed as it
    cancels out. Here is a basic walk through:
    m_1 - m_ref = -2.5log10(I_1/I_ref)
    m_2 - m_ref = -2.5log10(I_1/2I_ref)
                = -2.5log10(I_1/I_ref) + 2.5log10(2)
    m_2 = m_1 + 2.5log10(2)
    """

    return mag + 2.5 * np.log10(2)

def flux_to_sb(flux, pixscale, zeropoint):
    """
    Converts a flux value (ie pixel value) to log space surface brightness.

    flux: intensity in a pixel
    pixelscale: pixel area in arcsec^2
    zeropoint: user supplied conversion factor for a given magnitude system

    returns: surface brightness
    """
    return -2.5 * np.log10(flux) + zeropoint + 5 * np.log10(pixscale)

def flux_to_mag(flux, zeropoint, fluxe=None):
    """
    converts a total flux (ie sum of pixel values) to log space magnitude.

    flux: total intensity in an area
    zeropoint: user supplied conversion factor for a given magnitude system
    fluxe: error on flux total

    returns: magnitude
    """
    mag = -2.5 * np.log10(flux) + zeropoint
    if fluxe is None:
        return mag
    return mag, 2.5 * fluxe / (np.log(10) * flux)


def sb_to_flux(sb, pixscale, zeropoint):
    """
    Converts a surface brightness to a flux value (ie pixel value).

    sb: surface brightness of a pixel
    pixelscale: pixel area in arcsec^2
    zeropoint: user supplied conversion factor for a given magnitude system

    returns: pixel flux
    """
    return (pixscale ** 2) * 10 ** (-(sb - zeropoint) / 2.5)


def mag_to_flux(mag, zeropoint, mage=None):
    """
    converts a total magnitude into linear flux units (ie sum of pixel values).

    mag: total magnitude in an area
    zeropoint: user supplied conversion factor for a given magnitude system
    mage: error on total magnitude

    returns: total flux
    """
    if mage is None:
        return 10 ** (-(mag - zeropoint) / 2.5)
    I = 10 ** (-(mag - zeropoint) / 2.5)
    return I, np.log(10) * I * mage / 2.5

def L_to_Flux(L, D):
    """
    Converts luminosity into the observed flux at a given distance.
    L: Luminosity in solar luminosities
    D: Distance to star in parsecs
    returns: Flux at distance D
    """

    return L / (4.0 * np.pi * (D**2))

def Flux_to_L(Flux, D):
    """
    Converts the observed flux into luminosity at a given distance.
    F: Flux at distance D
    D: Distance to star in parsecs
    returns: Luminosity in solar luminosities
    """

    return Flux * 4.0 * np.pi * (D**2)

def mag_to_flux_asinh(m, band):
    """
    Converts an sdss magnitude into fixme
    http://www.sdss.org/dr12/algorithms/fluxcal/
    http://www.sdss.org/dr12/algorithms/magnitudes/
    https://en.wikipedia.org/wiki/AB_magnitude
    """
    assert band in 'ugriz'

    b = asinh_softening[band]

    return 2. * b * np.arcsinh( - m * np.log(10) / 2.5 - np.log(b) )

def flux_to_mag_asinh(f_f0, band):

    assert band in 'ugriz'

    b = asinh_softening[band]

    return 2.5 * (np.arcsinh(f_f0 / (2. * b)) + np.log(b)) / np.log(10)

def pc_to_arcsec(R, D, Re = 0.0, De = 0.0):
    """
    Converts a size in parsec to arcseconds

    R: length in pc
    D: distance in pc
    """

    theta = R / (D * Au_to_Pc)
    if np.all(Re == 0) and np.all(De == 0):
        return theta
    else:
        e = theta * np.sqrt((Re/R)**2 + (De/D)**2)
        return theta, e

def arcsec_to_pc(theta, D, thetae = 0.0, De = 0.0):
    """
    Converts a size in arcseconds to parsec

    theta: angle in arcsec
    D: distance in pc
    """
    r = theta * D * Au_to_Pc
    if np.all(thetae == 0) and np.all(De == 0):
        return r
    else:
        e = r * np.sqrt((thetae / theta)**2 + (De / D)**2)
        return r, e

def DM_to_D(DM, DME = None):
    """
    Convert distance modulus to linear distance (pc)

    DM: distance modulus
    """
    D = 10**(1 + DM/5)
    if DME is None:
        return D
    else:
        return D, 0.2*DME*D*np.log(10)

def D_to_DM(D, DE = None):
    """
    Convert linear distance (pc) to distance modulus.

    D: linear distance in pc
    """
    DM = 5*np.log10(D/10)
    if DE is None:
        return DM
    else:
        return DM, DE*5/(D*np.log(10))

def ISB_to_muSB(I, band, IE = None):
    """
    Converts surface brightness in Lsolar pc^-2 into mag arcsec^-2

    I: surface brightness, (L/Lsun) pc^-2
    band: Photometric band in which measurements were taken
    returns: surface brightness in mag arcsec^-2
    """

    muSB = 21.571 + Abs_Mag_Sun[band] - 2.5 * np.log10(I)
    if IE is None:
        return muSB
    else:
        return muSB, (2.5/np.log(10)) * IE / I

def muSB_to_ISB(mu, band, muE = None):
    """
    Converts surface brightness in mag arcsec^-2 into Lsolar pc^-2

    mu: surface brightness, mag arcsec^-2
    band: Photometric band in which measurements were taken
    returns: surface brightness in (L/Lsun) pc^-2
    """

    ISB = 10**((21.571 + Abs_Mag_Sun[band] - mu)/2.5)
    if muE is None:
        return ISB
    else:
        return ISB, (np.log(10)/2.5) * ISB * muE

def muSBE_to_ISBE(mu, muE, band):
    """
    Converts an uncertainty from surface brightness to intensity

    mu: surface brightness, mag arcsec^-2
    muE: surface brightness uncertainty (unitless)
    band: Photometric band in which measurements were taken
    returns: intensity uncertainty (L/Lsun) pc^-2
    """

    return (np.log(10)/2.5) * muSB_to_ISB(mu, band) * muE

def ISBE_to_muSBE(I, IE, band):
    """
    Converts an uncertainty from intensity to surface brightness

    I: intensity (L/Lsun) pc^-2
    IE: intensity uncertainty (L/Lsun) pc^-2
    band: Photometric band in which measurements were taken
    returns: surface brightness uncertainty (unitless)
    """

    return (2.5/np.log(10)) * IE / I
    

def COG_to_SBprof(R, COG, band):
    """
    Compute a surface brightness profile from a curve of growth.

    R: Radius array values (pc)
    COG: mag array values (mag)
    """

    assert np.all(COG[:-1] - COG[1:] > 0)
    assert len(R) == len(COG)
    ISB = [mag_to_L(COG[0], band = band) / (np.pi * R[0]**2)]
    for i in range(1,len(R)):
        ISB.append((mag_to_L(COG[i], band = band) - mag_to_L(COG[i-1], band = band)) / (np.pi * (R[i]**2 - R[i-1]**2)))
    # SB = [mag_to_magperarcsec2(COG[0],R = R[0])]

    # for i in range(1,len(R)):
    #     SB.append(mag_to_magperarcsec2(L_to_mag(mag_to_L(COG[i], zeropoint = 15) - mag_to_L(COG[i-1], zeropoint = 15), zeropoint = 15), A = np.pi*(R[i]**2 - R[i-1]**2)))

    return ISB_to_muSB(np.array(ISB), band = band)

def v_to_z(v):
    """
    computes the redshift using the cmb velocity which is purely in the radial direction

    v: the velocity of an object along the line of sight (km s^-1)

    returns: z the redshift (unitless)
    """

    # https://en.wikipedia.org/wiki/Redshift#Redshift_formulae
    return np.sqrt((1. + (v / c)) / (1. - (v / c))) - 1.

def z_to_v(z):
    """
    computes the the cmb velocity which is purely in the radial direction from redshift

    z: redshift (unitless)

    returns: v the velocity of an object along the line of sight (km s^-1)
    """

    # https://en.wikipedia.org/wiki/Redshift#Redshift_formulae
    return c * ((1. + z)**2 - 1.) / ((1. + z)**2 + 1.)

def vhel_to_vcmb(vhel, l, b):
    """
    Converts from Heliocentric to 3K Background CMB velocities.
    
    see: https://ned.ipac.caltech.edu/help/velc_help.html#notes
    or see: ApJ 473, 576, 1996

    vhel: Heliocentric velocity in km/s
    l: galactic longitude in radians
    b: galactic latitude in radians
    """
    vapex = 371.0
    lapex = 264.14 *np.pi/180
    bapex = 48.26 *np.pi/180
    return vhel + vapex*(np.sin(b)*np.sin(bapex) + np.cos(b)*np.cos(bapex)*np.cos(l - lapex))

def vcmb_to_vhel(vcmb, l, b):
    """
    Converts from 3K Background CMB to Heliocentric velocities (you probably want the other one).
    
    see: https://ned.ipac.caltech.edu/help/velc_help.html#notes
    or see: ApJ 473, 576, 1996

    vcmb: 3K Background velocity in km/s
    l: galactic longitude in radians
    b: galactic latitude in radians
    """
    vapex = 371.0
    lapex = 264.14 *np.pi/180
    bapex = 48.26 *np.pi/180
    return vcmb - vapex*(np.sin(b)*np.sin(bapex) + np.cos(b)*np.cos(bapex)*np.cos(l - lapex))

# Inclination
######################################################################
def inclination(axisratio, axisratio_min):
    """
    axisratio: the measured axis ratio (b/a) (unitless)
    axisratio_min: the assumed minimum axis ratio (b_min/a) (unitless)

    returns: corrected inclination of galaxy (radians)
    """

    axisratio = np.reshape(np.array(axisratio),-1)
    axisratio[axisratio < 0] = 1e-3
    axisratio[axisratio > 1] = 1 - 1e-3
    assert np.all(np.logical_and(0 <= axisratio, axisratio <= 1))
    assert np.all(np.logical_and(0 <= axisratio_min, axisratio_min < 1))

    if type(axisratio) in [float,np.float64]:
        axisratio = np.array([axisratio])
        
    ret = np.arccos(np.sqrt((axisratio**2 - axisratio_min**2)/(1. - axisratio_min**2)))
    if type(ret) is np.ndarray:
        ret[axisratio < axisratio_min] = np.pi/2.
    elif not np.isfinite(ret):
        ret = np.pi/2.

    return ret if len(ret) > 1 else ret[0]

    
def axisratio(i, axisratio_min):
    """
    i: corrected inclination of galaxy (radians)
    axisratio_min: the assumed minimum axis ratio when the corrected inclination was calculated (b_min/a) (unitless)

    returns: the measured axis ratio (b/a) (unitless)
    """
    if axisratio_min < 0:
        axisratio_min = 1e-3
    elif axisratio_min > 1:
        axisratio_min = 1 - 1e-3
    #assert np.all(np.logical_and(0 <= axisratio_min, axisratio_min < 1))
    assert np.all(np.logical_and(0 <= i,  i <= np.pi/2))

    if i >= np.pi/2. - 1e-3:
        return axisratio_min
    else:
        return np.sqrt((np.cos(i)**2) * (1. - axisratio_min**2) + axisratio_min**2)


# Mass-to-Light
######################################################################
def Get_M2L(colour, colour_type, M2L_colour, table = 'Roediger_BC03', sim = False, colour_err = None):
    """
    colour: the measured colour value (mag, float or numpy array)
    colour_type: the string representation of the colour, eg 'g-i' (ID, string)
    M2L_colour: the colour of the light profile to apply the M2L value to, eg 'i' (ID, string)
    table: which M2L table to use

    returns exp(a + b * colour)
    """

    Cluver_2014 = {'header': ['bw1', 'aw1'],
                   'w1-w2': [-2.54, -0.17]}
    
    Roediger_BC03 = {'header': ['bg', 'ag', 'br', 'ar', 'bi', 'ai', 'bz', 'az', 'bH', 'aH'],
                     'g-r': [2.029, -0.984, 1.629, -0.792, 1.438, -0.771, 1.306, -0.796, 0.980, -0.920], 
                     'g-i': [1.379, -1.067, 1.110, -0.861, 0.979, -0.831, 0.886, -0.848, 0.656, -0.950], 
                     'g-z': [1.116, -1.132, 0.900, -0.916, 0.793, -0.878, 0.716, -0.888, 0.521, -0.968], 
                     'g-H': [0.713, -1.070, 0.577, -0.870, 0.507, -0.834, 0.454, -0.842, 0.313, -0.902], 
                     'r-i': [4.107, -1.170, 3.325, -0.952, 2.925, -0.908, 2.634, -0.912, 1.892, -0.977], 
                     'r-z': [2.322, -1.211, 1.883, -0.987, 1.655, -0.937, 1.483, -0.935, 1.038, -0.975], 
                     'r-H': [1.000, -0.988, 0.814, -0.809, 0.713, -0.778, 0.634, -0.786, 0.414, -0.833], 
                     'i-z': [5.164, -1.212, 4.201, -0.991, 3.683, -0.939, 3.283, -0.931, 2.210, -0.947], 
                     'i-H': [1.257, -0.869, 1.024, -0.713, 0.895, -0.693, 0.792, -0.706, 0.495, -0.761], 
                     'z-H': [1.615, -0.729, 1.316, -0.600, 1.150, -0.593, 1.015, -0.616, 0.615, -0.692]}
    
    Roediger_FSPS = {'header': ['bg', 'ag', 'br', 'ar', 'bi', 'ai', 'bz', 'az', 'bH', 'aH'],
                     'g-r': [1.897, -0.811, 1.497, -0.647, 1.281, -0.602, 1.102, -0.583, 0.672, -0.605], 
                     'g-i': [1.231, -0.805, 0.973, -0.644, 0.831, -0.597, 0.713, -0.576, 0.426, -0.592], 
                     'g-z': [0.942, -0.764, 0.744, -0.612, 0.634, -0.568, 0.542, -0.548, 0.316, -0.565], 
                     'g-H': [0.591, -0.655, 0.468, -0.527, 0.398, -0.494, 0.339, -0.482, 0.191, -0.515], 
                     'r-i': [3.374, -0.745, 2.675, -0.600, 2.275, -0.556, 1.940, -0.537, 1.120, -0.554], 
                     'r-z': [1.795, -0.670, 1.421, -0.539, 1.206, -0.502, 1.021, -0.487, 0.570, -0.513], 
                     'r-H': [0.824, -0.542, 0.654, -0.439, 0.555, -0.418, 0.469, -0.414, 0.254, -0.463], 
                     'i-z': [3.709, -0.550, 2.933, -0.443, 2.484, -0.419, 2.084, -0.411, 1.112, -0.457], 
                     'i-H': [1.073, -0.460, 0.852, -0.375, 0.722, -0.362, 0.608, -0.365, 0.322, -0.430], 
                     'z-H': [1.493, -0.414, 1.188, -0.339, 1.008, -0.333, 0.849, -0.341, 0.449, -0.417]} 
    
    
    Zhang_BC03_40LGDwarf = {'header': ['aB','bB','ag','bg','aV','bV','ar','br','aR','bR','ai','bi','aI','bI','az','bz'],
                            'g-r': [-0.842, 1.786, -0.745, 1.616, -0.617, 1.367, -0.552, 1.216, -0.540, 1.164, -0.511, 1.038, -0.504, 0.988, -0.501, 0.937],
                            'g-i': [-0.837, 1.227, -0.739, 1.109, -0.612, 0.937, -0.547, 0.833, -0.534, 0.797, -0.505, 0.709, -0.497, 0.674, -0.494, 0.638],
                            'g-z': [-0.808, 1.020, -0.712, 0.920, -0.587, 0.776, -0.523, 0.687, -0.511, 0.657, -0.482, 0.582, -0.474, 0.552, -0.470, 0.520],
                            'r-i': [-0.808, 3.844, -0.711, 3.467, -0.586, 2.924, -0.523, 2.592, -0.511, 2.476, -0.481, 2.192, -0.474, 2.076, -0.469, 1.956],
                            'r-z': [-0.725, 2.270, -0.634, 2.039, -0.517, 1.709, -0.459, 1.507, -0.448, 1.435, -0.422, 1.260, -0.415, 1.186, -0.410, 1.107],
                            'B-V': [-0.981, 1.689, -0.869, 1.526, -0.721, 1.289, -0.643, 1.145, -0.627, 1.095, -0.587, 0.975, -0.576, 0.927, -0.568, 0.878],
                            'B-R': [-1.160, 1.140, -1.031, 1.030, -0.858, 0.871, -0.766, 0.773, -0.743, 0.740, -0.691, 0.659, -0.675, 0.626, -0.662, 0.593],
                            'B-I': [-1.288, 0.879, -1.146, 0.794, -0.954, 0.670, -0.850, 0.595, -0.823, 0.569, -0.761, 0.505, -0.740, 0.479, -0.722, 0.453],
                            'V-R': [-1.528, 3.502, -1.365, 3.166, -1.141, 2.678, -1.018, 2.381, -0.985, 2.278, -0.906, 2.029, -0.880, 1.929, -0.856, 1.828],
                            'V-I': [-1.610, 1.822, -1.435, 1.643, -1.197, 1.386, -1.065, 1.229, -1.028, 1.174, -0.940, 1.040, -0.909, 0.986, -0.880, 0.929],
                            'R-I': [-1.662, 3.716, -1.478, 3.343, -1.228, 2.809, -1.088, 2.483, -1.048, 2.368, -0.952, 2.085, -0.916, 1.968, -0.882, 1.845]}

    Zhang_BC03_FullSampleA0p5 = {'header': ['aB','bB','ag','bg','aV','bV','ar','br','aR','bR','ai','bi','aI','bI','az','bz'],
                                 'g-r': [-0.690, 1.566, -0.601, 1.410, -0.475, 1.169, -0.411, 1.020, -0.402, 0.969, -0.382, 0.848, -0.380, 0.793, -0.381, 0.721],
                                 'g-i': [-0.738, 1.128, -0.646, 1.021, -0.517, 0.855, -0.451, 0.751, -0.440, 0.716, -0.416, 0.630, -0.414, 0.590, -0.413, 0.538],
                                 'g-z': [-0.742, 0.914, -0.651, 0.829, -0.523, 0.696, -0.455, 0.611, -0.444, 0.582, -0.420, 0.511, -0.416, 0.478, -0.414, 0.435],
                                 'r-i': [-0.770, 3.381, -0.674, 3.071, -0.546, 2.607, -0.480, 2.314, -0.469, 2.206, -0.446, 1.946, -0.442, 1.819, -0.439, 1.650],
                                 'r-z': [-0.707, 1.722, -0.621, 1.570, -0.503, 1.341, -0.442, 1.192, -0.432, 1.133, -0.409, 0.987, -0.405, 0.913, -0.401, 0.811],
                                 'B-V': [-0.842, 1.527, -0.735, 1.371, -0.585, 1.134, -0.510, 0.992, -0.498, 0.946, -0.468, 0.831, -0.462, 0.779, -0.457, 0.709],
                                 'B-R': [-1.030, 1.057, -0.906, 0.951, -0.731, 0.793, -0.638, 0.696, -0.619, 0.663, -0.576, 0.584, -0.564, 0.548, -0.550, 0.500],
                                 'B-I': [-1.185, 0.821, -1.050, 0.742, -0.854, 0.621, -0.747, 0.545, -0.721, 0.519, -0.663, 0.455, -0.643, 0.426, -0.621, 0.388],
                                 'V-R': [-1.340, 3.131, -1.185, 2.822, -0.965, 2.357, -0.847, 2.075, -0.819, 1.980, -0.756, 1.752, -0.736, 1.648, -0.711, 1.509],
                                 'V-I': [-1.463, 1.621, -1.302, 1.467, -1.068, 1.231, -0.936, 1.082, -0.901, 1.029, -0.821, 0.902, -0.790, 0.842, -0.753, 0.763],
                                 'R-I': [-1.415, 2.888, -1.266, 2.629, -1.052, 2.240, -0.929, 1.988, -0.894, 1.888, -0.810, 1.639, -0.774, 1.513, -0.728, 1.339]}

    Zhang_BC03_FullSampleAvar = {'header': ['aB','bB','ag','bg','aV','bV','ar','br','aR','bR','ai','bi','aI','bI','az','bz'],
                                 'g-r': [-0.701, 1.550, -0.608, 1.394, -0.479, 1.148, -0.417, 0.998, -0.410, 0.948, -0.390, 0.818, -0.389, 0.756, -0.389, 0.667],
                                 'g-i': [-0.732, 1.049, -0.639, 0.948, -0.509, 0.788, -0.444, 0.685, -0.432, 0.647, -0.406, 0.552, -0.401, 0.506, -0.396, 0.439],
                                 'g-z': [-0.695, 0.764, -0.606, 0.691, -0.484, 0.573, -0.421, 0.498, -0.410, 0.468, -0.383, 0.393, -0.376, 0.354, -0.368, 0.296],
                                 'r-i': [-0.664, 2.540, -0.575, 2.298, -0.456, 1.910, -0.397, 1.661, -0.386, 1.555, -0.360, 1.285, -0.353, 1.141, -0.343, 0.924],
                                 'r-z': [-0.597, 1.187, -0.518, 1.084, -0.409, 0.908, -0.357, 0.794, -0.347, 0.741, -0.326, 0.602, -0.319, 0.524, -0.309, 0.403],
                                 'B-V': [-0.811, 1.473, -0.702, 1.317, -0.552, 1.078, -0.478, 0.935, -0.468, 0.889, -0.440, 0.770, -0.435, 0.713, -0.432, 0.634],
                                 'B-R': [-1.016, 1.030, -0.890, 0.926, -0.712, 0.764, -0.621, 0.667, -0.604, 0.635, -0.559, 0.551, -0.546, 0.510, -0.529, 0.453],
                                 'B-I': [-1.147, 0.773, -1.015, 0.699, -0.821, 0.581, -0.715, 0.506, -0.688, 0.479, -0.625, 0.409, -0.601, 0.375, -0.570, 0.326],
                                 'V-R': [-1.259, 2.820, -1.118, 2.554, -0.906, 2.115, -0.787, 1.834, -0.756, 1.729, -0.679, 1.467, -0.650, 1.338, -0.610, 1.153],
                                 'V-I': [-1.285, 1.340, -1.138, 1.209, -0.926, 1.003, -0.805, 0.871, -0.771, 0.818, -0.685, 0.684, -0.646, 0.614, -0.591, 0.511],
                                 'R-I': [-1.147, 2.166, -1.019, 1.974, -0.828, 1.659, -0.722, 1.445, -0.690, 1.352, -0.609, 1.112, -0.572, 0.983, -0.516, 0.789]}
    
    if M2L_colour == '3.6um':
        return np.ones(len(colour))*0.5 
    if colour_type == 'w1':
        return np.ones(len(colour))*0.5 
    if colour_type == 'N-A':
        return colour
    if table == 'Zhang_BC03_40LGDwarf':
        T = Zhang_BC03_40LGDwarf
    elif table == 'Zhang_BC03_FullSampleA0p5':
        T = Zhang_BC03_FullSampleA0p5
    elif table == 'Zhang_BC03_FullSampleAvar':
        T = Zhang_BC03_FullSampleAvar
    elif table == 'Roediger_BC03':
        T = Roediger_BC03
        if np.any(list((c in colour_type) for c in 'BVRI')) or np.any(list((c in M2L_colour) for c in 'BVRI')):
            print('switching colour tables')
            T = Zhang_BC03_FullSampleAvar
    elif table == 'Roediger_FSPS':
        T = Roediger_FSPS
        if np.any(list((c in colour_type) for c in 'BVRI')) or np.any(list((c in M2L_colour) for c in 'BVRI')):
            print('switching colour tables')
            T = Zhang_BC03_FullSampleAvar
    elif table == 'Cluver_2014':
        T = Cluver_2014
        colour = np.clip(colour + 0.65, a_min = -0.1, a_max = 0.2)#fixme convert Vega/AB  #np.clip(colour, a_min = -0.05, a_max = 0.2)
    else:
        raise ValueError('unrecognized table: ', table)
            
    a = None
    b = None
    for i in range(len(T['header'])):
        if T['header'][i] == 'a' + M2L_colour:
            a = T[colour_type][i]
        elif T['header'][i] == 'b' + M2L_colour:
            b = T[colour_type][i]
        if (not a is None) and (not b is None):
            break
    else:
        raise ValueError(f'Colour could not be found in table: {M2L_colour}, {colour_type}')

    if not colour_err is None:
        return 10**(a + b * colour), abs(b * 10**(a + b * colour) * np.log(10) * colour_err)
    else:
        return 10**(a + b * colour)


# SB prof integrator
######################################################################


def fluxdens_to_fluxsum(R, I, axisratio):
    """
    Integrate a flux density profile

    R: semi-major axis length (arcsec)
    I: flux density (flux/arcsec^2)
    axisratio: b/a profile
    """

    S = np.zeros(len(R))
    S[0] = I[0] * np.pi * axisratio[0] * (R[0] ** 2)
    for i in range(1, len(R)):
        S[i] = (
            trapz(2 * np.pi * I[: i + 1] * R[: i + 1] * axisratio[: i + 1], R[: i + 1])
            + S[0]
        )
    return S

