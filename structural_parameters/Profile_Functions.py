import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import iqr
import matplotlib.pyplot as plt
from scipy.special import gamma

# Photometry Profile Evaluating
######################################################################
def fit_two_points(X1, Y1, X2, Y2, X):
    """
    Fit a line to two points and evaluate the fit at a third X value.
    When used with numpy arrays, can simultaneously work on many pairs of points

    X1: the x-axis value for the first point
    Y1: the y-axis value for the first point
    X2: the x-axis value for the second point
    Y2: the y-axis value for the second point
    X: the new x-axis value

    returns: the new y-axis value
    """
    L = (X - X2) / (X1 - X2)
    return L * Y1 + (1. - L) * Y2

def Evaluate_Surface_Brightness(R, SB, atR, E = None):
    """
    Evaluate a smoothed version of a surface brightness profile at
    any radius "atR". For values smaller than min(R) or greater than
    max(R), a linear extrapolation is done with numpy.polyfit on the
    inner 5 points or outer 25% of the profile. Within range(R),
    numpy.interp is used.

    R: Radius of the profile [any units]
    SB: Surface brightness [mag arcsec^-2]
    atR: Radius where SB profile should be evaluated [same units as R]
         if atR is an iterable, this function will call itself recursively
    E: Uncertainty in Surface Brightness [unitless sigma in mag arcsec^-2 space]

    returns: Surface Brightness at "atR"
    """

    if hasattr(atR, '__len__'):
        return list(map(Evaluate_Surface_Brightness, [R]*len(atR), [SB]*len(atR), atR, [E]*len(atR)))

    if len(R) < 5:
        if E is None:
            return '-'
        else:
            return '-','-'

    if atR < R[0]:
        r = np.polyval(np.polyfit(R[:5],SB[:5],1,w = None if E is None else 1./np.clip(E[:5], a_min = 0.001, a_max = None)), max(atR,0.))
    elif atR > R[-1]:
        start = int(len(R) * 3. / 4.)
        r = np.polyval(np.polyfit(R[start:],SB[start:],1,w = None if E is None else 1./np.array(E[start:])), atR)
    else:
        r = np.interp(atR, R, SB)
    if E is None:
        return r
    else:
        if atR < R[0]:
            e = np.mean(E[:5])
        elif atR > R[-1]:
            e = np.mean(E[start:])
        else:
            e = E[np.argmin(np.abs(R - r))]
        if e < 0:
            print('How is this possible??? negative error???', R, SB, atR, E)
        return r, e

def Evaluate_Magnitude(R, m, atR, E = None):
    """
    Evaluate a smoothed version of a curve of growth at
    any radius "atR". For values smaller than min(R), a linear
    extrapolation is done with numpy.polyfit on the
    inner 5 points. For values greater than max(R), the minimum
    m value is returned. Within range(R), numpy.interp is used.

    R: Radius of the profile [any units]
    m: magnitude enclosed within radius R [mag]
    atR: Radius where SB profile should be evaluated [same units as R]
         if atR is an iterable, this function will call itself recursively
    E: Uncertainty in magnitude [unitless sigma in mag space]

    returns: Magnitude at "atR"
    """

    if hasattr(atR, '__len__'):
        return list(map(Evaluate_Magnitude, [R]*len(atR), [m]*len(atR), atR, [E]*len(atR)))
    
    if len(R) < 5:
        if E is None:
            return '-'
        else:
            return '-','-'
    
    if R[0] == 0:
        start = 1
    else:
        start = 0
            
    if atR < R[start]:
        matR = np.polyval(np.polyfit(np.log10(np.abs(R[start:5+start])),m[start:5+start],1,w = None if E is None else 1./np.clip(E[start:5+start], a_min = 0.001, a_max = None)), np.log10(max(atR,1e-3)))
        if E is None:
            return matR
        else:
            return matR, E[start]
    elif atR > R[-1]:
        if E is None:
            return min(m[start:])
        else:
            return min(m[start:]), E[np.argmin(m[start:])]
    else:
        matR = np.interp(atR, R[start:], m[start:])
        if E is None:
            return matR
        else:
            return matR, np.interp(atR, R[start:], E[start:])

def Total_Apparent_Magnitude(m, E = None, return_radius = False):
    """
    Compute the total apparent magnitude from a curve of growth. In principle
    this should be the last point, however in some cases the sky is not
    properly subtracted and the curve of growth does not peak at the last
    point, thus the minimum magnitude is used.

    m: Curve of growth magnitude measurements, assumed to already be
       truncated (ie by Truncate_Surface_Brightness_Profile) [mag]
    E: Error at each point [unitless sigma in mag space]
    return_radius: Also include the index where the total magnitude
                   is measured [index]

    returns: total magnitude, uncertainty at total magnitude if
             E provided, index if requested
    """
   
    if len(m) < 5:
        return tuple(['-']*(1 + int(not E is None) + int(return_radius)))
    
    returnvals = tuple([min(m), None if E is None else E[np.argmin(m)], np.argmin(m)])

    if E is None and not return_radius:
        return returnvals[0]
    elif E is None:
        return returnvals[0], returnvals[2]
    elif return_radius:
        return returnvals
    else:
        return returnvals[0], returnvals[1]
    
def Effective_Radius(R, m, E = None, ratio = 0.5, force_total_mag = None):
    """
    Extract the percentage radius from a curve of growth. Determine the radius
    at which "ratio" fraction of the total light is enclosed.

    R: Radius of the profile [any units]
    m: Curve of growth magnitude measurements, assumed to already be
       truncated (ie by Truncate_Surface_Brightness_Profile) [mag]
    E: Error at each point [unitless sigma in mag space]
    ratio: fraction of total light for radius of interest [unitless fraction]
    force_total_mag: User input value for total magnitude, if None then
                     "Total_Apparent_Magnitude" function is used [None, or float]

    returns: Percentage radius [same units as "R"]
    """
    if len(R) < 5:
        if E is None:
            return '-'
        else:
            return '-','-'

    assert 0 < ratio < 1
    assert len(R) == len(m)
    if not E is None:
        assert len(R) == len(E)

    mag_effective = (Total_Apparent_Magnitude(m) if force_total_mag is None else force_total_mag) + 2.5 * np.log10(1./ratio) 
    Re = '-'
    ReE = '-'
    for i in range(1,len(R)):
        if m[i] < mag_effective < m[i-1]:
            Re = fit_two_points(m[i-1], R[i-1], m[i], R[i], mag_effective)
            if not E is None:
                ReE = abs(np.sqrt(np.mean(E[max(i-2, 0):min(len(R),i+3)]**2)) * np.polyfit(m[max(i-2, 0):min(len(R),i+3)],R[max(i-2, 0):min(len(R),i+3)], 1)[0]) # (R[i] - R[i-1]) / (m[i] - m[i-1])
                
    if E is None:
        return Re
    else:
        return Re, ReE

def Isophotal_Radius(R, SB, mu_iso, E = None):
    """
    Extract radius at which profile reaches a specific surface brightness.
    For cases where the profile crosses the same surface brightness
    multiple times the selection behavior is currently undefined.

    R: Radius of profile [any units]
    SB: Surface brightness [mag arcsec^-2]
    mu_iso: Desired surface brightness value for isophotal radius [mag arcsec^-2]
    E: Uncertainty in Surface Brightness [unitless sigma in mag arcsec^-2 space]

    returns: Isophotal radius [same units as "R"]
    """
    
    if len(R) == 0:
        if E is None:
            return '-','-'
        else:
            return '-', '-', '-'

    # select 5 points closest to the right SB
    N = np.argsort(np.abs(SB - mu_iso))[:5]
    N = N[np.argsort(R[N])]
    # CHOOSE = np.logical_and(SB > (mu_iso - 3.), SB < (mu_iso + 3.))
    # if np.sum(CHOOSE) < 3:
    #     CHOOSE = list(True for c in CHOOSE)
    p = np.polyfit(R[N], SB[N], 1)
    if p[0] < 0:
        print('Odd, negative slope for SB prof linear fit')
        p = np.polyfit(R, SB, 1)
    R0 = min(abs((mu_iso - p[1]) / p[0]), 3*R[-1])
    Riso = np.exp(minimize(lambda r: (Evaluate_Surface_Brightness(R, SB, np.exp(r[0])) - mu_iso)**2, x0 = [np.log(min(abs(R0),R[-1]))], method = 'Nelder-Mead').x[0])
    if E is None:
        return Riso, (0 if Riso < np.max(R) else 1)
    else:
        close = np.argsort(np.abs(R - Riso))[:5]
        p = np.polyfit(R[close], SB[close],1)
        return Riso, min(abs(Evaluate_Surface_Brightness(R, SB, Riso, E = E)[1] / p[0]), 10.), (0 if Riso < np.max(R) else 1)


# Photometry Profile Models
######################################################################
muA = 5*np.log10(1/(10*4.84814e-6))

def I_to_SB(I):
    return 22.5 - 2.5*np.log10(I)
def SB_to_I(SB):
    return 10**((22.5-SB)/2.5)

def disk(R, Rs, I0):
    return I0*np.exp(-R/Rs)

def sersic(R, Re, Ie, n):
    bn = 1.9992*n - 0.3271
    return Ie*np.exp(-bn*((R/Re)**(1/n) - 1))

def sersic_Ie(total_mag, Re, n, ellip = 0):
    bn = 1.9992*n - 0.3271
    # rterm = 5*np.log10(Re)
    # gterm = 2*np.pi*n*(np.exp(bn)/np.real(bn**(2*n)))*(1. - ellip)*np.real(gamma(2*n))
    # lgterm = 2.5*np.log10(gterm)
    # mue = total_mag + rterm + lgterm
    # return SB_to_I(mue)
    return SB_to_I(total_mag) / (2*np.pi*Re**2 *(np.exp(bn)*n/(bn**(2*n))) * np.real(gamma(2*n))*(1. - ellip))
    
def disk_I0(total_mag, Rs, ellip = 0):
    return SB_to_I(total_mag)/(2*np.pi*(1. - ellip)*Rs**2)

def _fitbulgedisk(x, R, SB):
    if np.any(np.array(x) <= 0):
        return np.inf

    I = disk(R,x[0], x[1]) + sersic(R, x[2], x[3], x[4])
    
    return np.sqrt(np.mean((SB - I_to_SB(I))**2))

def _fitbulge(x, R, SB):
    if np.any(np.array(x) <= 0):
        return np.inf
    
    I = sersic(R, x[0], x[1], x[2])
    
    return np.sqrt(np.mean((SB - I_to_SB(I))**2))

def _fitbulgebulge(x, R, SB):
    if np.any(np.array(x) <= 0):
        return np.inf
    
    I1 = sersic(R, x[0], x[1], x[2])
    I2 = sersic(R, x[3], x[4], x[5])
    
    return np.sqrt(np.mean((SB - I_to_SB(I1 + I2))**2))

def fluxintegrate(I, ellip, args, R = np.inf):

    res = quad(lambda r: I(r, *args)*r, 0, R)

    return 2*np.pi*(1-ellip)*res[0]

def bulgediskfit(prof, ellip):

    disksb = np.polyfit(prof['R'][prof['R'] > (prof['R'][-1]/2)],
                        prof['SB'][prof['R'] > (prof['R'][-1]/2)], deg = 1)
    if disksb[0] < 0:
        disksb = np.polyfit(prof['R'], prof['SB'], deg = 1)

    diskf = [2.5/(np.log(10)*disksb[0]), 10**((22.5 - disksb[1])/2.5)]

    bfitloc = max(int(len(prof['R'])/6), 6)
    bulgef = minimize(_fitbulge, x0 = [prof['R'][bfitloc-1], SB_to_I(prof['SB'][bfitloc-1]), 2.], args = (prof['R'][:bfitloc], prof['SB'][:bfitloc]), method = 'Nelder-Mead')

    bulgediskf = minimize(_fitbulgedisk, x0 = [diskf[0], diskf[1], bulgef.x[0], bulgef.x[1], bulgef.x[2]], args = (prof['R'], prof['SB']), method = 'Nelder-Mead')
    
    return {'disk Rs': bulgediskf.x[0], 'disk I0': bulgediskf.x[1], 'disk M': I_to_SB(fluxintegrate(disk, ellip, (bulgediskf.x[0], bulgediskf.x[1]))), 'sersic1 Re': bulgediskf.x[2],
            'sersic1 Ie': bulgediskf.x[3], 'sersic1 M': I_to_SB(fluxintegrate(sersic, ellip, (bulgediskf.x[2], bulgediskf.x[3], bulgediskf.x[4]))), 'sersic1 n': bulgediskf.x[4]}

def diskbulgefit(prof, ellip):
    disksb = np.polyfit(prof['R'][prof['R'] < (prof['R'][-1]/4)],
                        prof['SB'][prof['R'] < (prof['R'][-1]/4)], deg = 1)
    if disksb[0] < 0:
        disksb = np.polyfit(prof['R'], prof['SB'], deg = 1)

    diskf = [2.5/(np.log(10)*disksb[0]), 10**((22.5 - disksb[1])/2.5)]

    bfitloc = max(int(len(prof['R'])/2), 6)
    bulgef = minimize(_fitbulge, x0 = [prof['R'][bfitloc-1], SB_to_I(prof['SB'][bfitloc-1]), 2.], args = (prof['R'][bfitloc:], prof['SB'][bfitloc:]), method = 'Nelder-Mead')

    bulgediskf = minimize(_fitbulgedisk, x0 = [diskf[0], diskf[1], bulgef.x[0], bulgef.x[1], bulgef.x[2]], args = (prof['R'], prof['SB']), method = 'Nelder-Mead')
    
    return {'disk Rs': bulgediskf.x[0], 'disk I0': bulgediskf.x[1], 'disk M': I_to_SB(fluxintegrate(disk, ellip, (bulgediskf.x[0], bulgediskf.x[1]))), 'sersic1 Re': bulgediskf.x[2],
            'sersic1 Ie': bulgediskf.x[3], 'sersic1 M': I_to_SB(fluxintegrate(sersic, ellip, (bulgediskf.x[2], bulgediskf.x[3], bulgediskf.x[4]))), 'sersic1 n': bulgediskf.x[4]}

def bulgebulgefit(prof, ellip):
    init_bulgedisk = bulgediskfit(prof, ellip)
    b1 = 1.9992 - 0.3271
    bulgebulgef = minimize(_fitbulgebulge, x0 = [init_bulgedisk['sersic1 Re'], init_bulgedisk['sersic1 Ie'], init_bulgedisk['sersic1 n'],
                                                 init_bulgedisk['disk Rs']*b1, init_bulgedisk['disk I0']*np.exp(-b1), 1.],
                           args = (prof['R'], prof['SB']), method = 'Nelder-Mead')

    return {'sersic1 Re': bulgebulgef.x[2], 'sersic1 Ie': bulgebulgef.x[3], 'sersic1 M': I_to_SB(fluxintegrate(sersic, ellip, (bulgebulgef.x[2], bulgebulgef.x[3], bulgebulgef.x[4]))), 'sersic1 n': bulgebulgef.x[4],
            'sersic2 Re': bulgebulgef.x[2], 'sersic2 Ie': bulgebulgef.x[3], 'sersic2 M': I_to_SB(fluxintegrate(sersic, ellip, (bulgebulgef.x[2], bulgebulgef.x[3], bulgebulgef.x[4]))), 'sersic2 n': bulgebulgef.x[4]}


def diskfit(prof, ellip):

    disksb = np.polyfit(prof['R'], prof['SB'], deg = 1)
    diskf = [2.5/(np.log(10)*disksb[0]), 10**((22.5 - disksb[1])/2.5)]
    
    return {'disk Rs': diskf[0], 'disk M': I_to_SB(fluxintegrate(disk, ellip, (diskf[0], diskf[1]))), 'disk I0': diskf[1]}

def bulgefit(prof, ellip):
    disksb = np.polyfit(prof['R'], prof['SB'], deg = 1)
    diskf = [2.5/(np.log(10)*disksb[0]), 10**((22.5 - disksb[1])/2.5)]
    b1 = 1.9992 - 0.3271

    bulgef = minimize(_fitbulge, x0 = [diskf[0]*b1, diskf[1]*np.exp(-b1), 1.], args = (prof['R'], prof['SB']), method = 'Nelder-Mead')
    
    return {'sersic1 Re': bulgef.x[0],
            'sersic1 Ie': bulgef.x[1],
            'sersic1 M': I_to_SB(fluxintegrate(sersic, ellip, (bulgef.x[0], bulgef.x[1], bulgef.x[2]))),
            'sersic1 n': bulgef.x[2]}

#Rotation Curves
######################################################################
def Tan_Model_Evaluate(x, R):
    """
    Evaluate an arctan model for a rotation curve. This simple model
    is a good first step when trying to fit a rotation curve, but
    is unable to reflect the rull range of observed rotation curves.
    x: 0 - r_t, the transition radius from rising to flat
       1 - v_c, asymptotic velocity
       2 - v0, the y-axis offset for the zero of the rotation curve
       3 - x0, the x-axis offset for the center of the galaxy
    R: Radius at which we would like the model velocity [any units]

    returns: Velocity for arctan model at R [units specified by x]
    """
    y = (R - x[3]) / x[0]
    return x[2] + (2. * x[1] / np.pi) * np.arctan(y)

def Tan_Model_Loss(x, R, V, E, fixed_origin = False):
    """
    Function that evaluates the difference between a model and
    measured rotation curve. Used only for fitting purposes by
    Tan_Model_Fit
    x: 0 - r_t, the transition radius from rising to flat
       1 - v_c, asymptotic velocity
       2 - v0, the y-axis offset for the zero of the rotation curve
       3 - x0, the x-axis offset for the center of the galaxy
    R: Radius at which we would like the model velocity [any length units]
    V: Measured velocity values [any speed units]
    E: uncertainty on velocity measurements
    fixed_origin: boolean to indicate if model has origin at (0,0)
                  or is allowed to float [boolean]
    
    returns: scalar value to minimize for optimal fit [unitless]
    """
    V_model = Tan_Model_Evaluate((list(x) + [0.,0.]) if fixed_origin else x, R)
    CHOOSE = np.isfinite(V_model)
    if np.sum(CHOOSE) <= 0:
        return 1e9 * len(R)    
    losses = ((V - V_model)/E)[CHOOSE]
    N = np.argsort(losses)
    return np.mean(losses[N][int(0.1*len(losses)):int(0.9*len(losses))]**2) + ((0. if fixed_origin else ((x[3]/20)**2)))

def Tan_Model_Fit(R, V, E = None, fixed_origin = False, n_walkers = 10):
    """
    Fits an arctan model to a rotation curve. This simple model
    is a good first step when trying to fit a rotation curve, but
    is unable to reflect the rull range of observed rotation curves.
    R: Radius at which we would like the model velocity [any length units]
    V: Measured velocity values [any speed units]
    E: uncertainty on velocity measurements
    fixed_origin: boolean to indicate if model has origin at (0,0)
                  or is allowed to float [boolean]

    returns: tuple with parameters for the arctan model as described
             in Tan_Model_Evaluate [various units]
    """
    
    if E is None:
        E = np.ones(len(R))
    else:
        E = np.clip(E, a_min = 3, a_max = None)
    N = np.argsort(V)
    sign = np.sign(np.sum(R*(V-np.median(V))))
    x0s = [[(max(R)-min(R))/15.,
            V[N[-2]] if np.all(R >= 0) else sign*iqr(V, rng = [10, 90])/2.]]
    if not fixed_origin:
        x0s[0] += [(V[N[1]] + V[N[-2]])/2., 0.]
    for i in range(n_walkers - 1):
        x0s.append([x0s[0][0] * 2**(np.random.normal()),
                    x0s[0][1] * 1.2**(np.random.normal())])
        if not fixed_origin:
            x0s[-1] += [x0s[0][2] + np.random.normal(scale = iqr(V, rng = [20,80]) / 10.),
                        x0s[0][3] + np.random.normal(scale = iqr(R, rng = [20,80]) / 10.)]
    res = []
    for i in range(n_walkers):
        res.append(minimize(Tan_Model_Loss,
                            x0 = x0s[i],
                            args = (R, V, E, fixed_origin)))
    return list(min(res, key = lambda x: x.fun if np.isfinite(x.fun) else np.inf).x) + ([0.,0.] if fixed_origin else []) #, x0s[0] + ([0.,0.] if fixed_origin else []))

#---------------------------------------------------------------------
def Tanh_Model_Evaluate(x, R):
    """
    Evaluate a tanh model for a rotation curve. This simple model
    is a good first step when trying to fit a rotation curve, but
    is unable to reflect the rull range of observed rotation curves.
    x: 0 - r_t, the transition radius from rising to flat
       1 - v_c, asymptotic velocity
       2 - v0, the y-axis offset for the zero of the rotation curve
       3 - x0, the x-axis offset for the center of the galaxy
    R: Radius at which we would like the model velocity [any units]

    returns: Velocity for tanh model at R [units specified by x]
    """
    y = (R - x[3]) / x[0]
    return x[2] + x[1] * np.tanh(y)

def Tanh_Model_Loss(x, R, V, E, fixed_origin = False):
    """
    Function that evaluates the difference between a model and
    measured rotation curve. Used only for fitting purposes by
    Tanh_Model_Fit
    x: 0 - r_t, the transition radius from rising to flat
       1 - v_c, asymptotic velocity
       2 - v0, the y-axis offset for the zero of the rotation curve
       3 - x0, the x-axis offset for the center of the galaxy
    R: Radius at which we would like the model velocity [any length units]
    V: Measured velocity values [any speed units]
    E: uncertainty on velocity measurements
    fixed_origin: boolean to indicate if model has origin at (0,0)
                  or is allowed to float [boolean]
    
    returns: scalar value to minimize for optimal fit [unitless]
    """
    V_model = Tanh_Model_Evaluate((list(x) + [0.,0.]) if fixed_origin else x, R)
    CHOOSE = np.isfinite(V_model)
    if np.sum(CHOOSE) <= 0:
        return 1e9 * len(R)    
    losses = ((V - V_model)/E)[CHOOSE]
    N = np.argsort(losses)
    return np.mean(losses[N][int(0.1*len(losses)):int(0.9*len(losses))]**2) + ((0. if fixed_origin else ((x[3]/20)**2)))

def Tanh_Model_Fit(R, V, E = None, fixed_origin = False, n_walkers = 10):
    """
    Fits a tanh model to a rotation curve. This simple model
    is a good first step when trying to fit a rotation curve, but
    is unable to reflect the rull range of observed rotation curves.
    R: Radius at which we would like the model velocity [any length units]
    V: Measured velocity values [any speed units]
    E: uncertainty on velocity measurements
    fixed_origin: boolean to indicate if model has origin at (0,0)
                  or is allowed to float [boolean]

    returns: tuple with parameters for the tanh model as described
             in Tanh_Model_Evaluate [various units]
    """
    
    if E is None:
        E = np.ones(len(R))
    else:
        E = np.clip(E, a_min = 3, a_max = None)
    N = np.argsort(V)
    sign = np.sign(np.sum(R*(V-np.median(V))))
    x0s = [[(max(R)-min(R))/15.,
            V[N[-2]] if np.all(R >= 0) else sign*iqr(V, rng = [10, 90])/2.]]
    if not fixed_origin:
        x0s[0] += [(V[N[1]] + V[N[-2]])/2., 0.]
    for i in range(n_walkers - 1):
        x0s.append([x0s[0][0] * 2**(np.random.normal()),
                    x0s[0][1] * 1.2**(np.random.normal())])
        if not fixed_origin:
            x0s[-1] += [x0s[0][2] + np.random.normal(scale = iqr(V, rng = [20,80]) / 10.),
                        x0s[0][3] + np.random.normal(scale = iqr(R, rng = [20,80]) / 10.)]
    res = []
    for i in range(n_walkers):
        res.append(minimize(Tanh_Model_Loss,
                            x0 = x0s[i],
                            args = (R, V, E, fixed_origin)))
    return list(min(res, key = lambda x: x.fun if np.isfinite(x.fun) else np.inf).x) + ([0.,0.] if fixed_origin else []) #, x0s[0] + ([0.,0.] if fixed_origin else []))

#---------------------------------------------------------------------
def Courteau97_Model_Evaluate(x, R, prof = None):
    """
    See Courteau 1997 Model 2 The Multi-Parameter Function
    x: 0 - r_t, the transition radius from rising to flat
       1 - v_c, asymptotic velocity unless beta is used
       2 - beta, used to model a dropping rotation curve
       3 - gamma, governs the sharpness of the turnover at r_t
       4 - v0, the y-axis offset for the zero of the rotation curve
       5 - x0, the x-axis offset for the center of the galaxy
    R: Radius at which we would like the model velocity [any length units]

    returns: Velocity for Courteau97 model at R [units specified by x]
    """
    sign = np.sign(R-x[5])
    y = np.abs(x[0] / (R - x[5]))
    V = x[4] + sign * x[1] * ((1. + y)**x[2]) / ((1. + y**x[3])**(1./x[3]))
    if prof is None:
        return V
    else:
        return V, prof['E'][np.argmin(np.abs(np.array(prof['R']) - R))], (1 if R > np.max(prof['R']) else 0)

def Courteau97_Model_Loss(x, R, V, E = None, fixed_origin = False):
    """
    Function that evaluates the difference between a model and
    measured rotation curve. Used only for fitting purposes by
    Courteau97_Model_Fit
    x: 0 - r_t, the transition radius from rising to flat
       1 - v_c, asymptotic velocity unless beta is used
       2 - beta, used to model a dropping rotation curve
       3 - gamma, governs the sharpness of the turnover at r_t
       4 - v0, the y-axis offset for the zero of the rotation curve
       5 - x0, the x-axis offset for the center of the galaxy
    R: Radius at which we would like the model velocity [any length units]
    V: Measured velocity values [any speed units]
    E: uncertainty on velocity measurements
    fixed_origin: boolean to indicate if model has origin at (0,0)
                  or is allowed to float [boolean]
    
    returns: scalar value to minimize for optimal fit [unitless]
    """
    if E is None:
        E = np.ones(len(R))
    else:
        E = np.clip(E, a_min = 5, a_max = 20)
    V_model = Courteau97_Model_Evaluate((list(x) + [0.,0.]) if fixed_origin else x, R)
    CHOOSE = np.isfinite(V_model)

    losses = ((V - V_model)/E)[CHOOSE]**2
    
    if np.sum(CHOOSE) < 10:
        return np.mean(losses) + (0. if fixed_origin else (x[5]/5)**2) + x[2]**2 + (x[3]/10.)**2 + 1e8*(len(R) - np.sum(CHOOSE))
    
    # print(np.median(losses)**2, iqr(losses, rng = [10, 90])**2, (0. if fixed_origin else (x[5]/5)**2))
    # return np.median(losses)**2 + iqr(losses, rng = [10, 90])**2 + (0. if fixed_origin else (x[5]/5)**2) + x[2]**2 + (x[3]/20.)**2
    losses.sort()
    return np.mean(losses[:-2]) + (0. if fixed_origin else (x[5]/5)**2) + x[2]**2 + (x[3]/10.)**2 + (x[1]/300)**2 + (x[0]/np.max(np.abs(R)))**2


def Courteau97_Model_Fit(R, V, E = None, fixed_origin = False, x0 = None, n_walkers = 10):
    """
    Fits the Courteau 97 model to a rotation curve. This emperical model
    is able to reflect the rull range of observed rotation curves, however
    some parameters have no direct interpretation.
    R: Radius at which we would like the model velocity [any length units]
    V: Measured velocity values [any speed units]
    E: uncertainty on velocity measurements
    fixed_origin: boolean to indicate if model has origin at (0,0)
                  or is allowed to float [boolean]
    n_walkers: Number of times to attempt fit with random initialization
               parameters, avoids local minima [counts]

    returns: tuple with parameters for the Courteau97 model as described
             in Courteau97_Model_Evaluate [various units]
    """

    if E is None:
        E = np.ones(len(R))
    else:
        E = np.clip(E, a_min = 5, a_max = 20)
    sign = np.sign(np.sum(R*(V - np.median(V)))) if np.any(R < 0) else 1.
    N = np.argsort(V)
    x0s = []
    count = 1
    for i in range(n_walkers):
        if x0 is None:
            x0s.append([[(R[-1] - R[0])/5., sign*iqr(V,rng = (10,90))/2, 0.1, 3.]])
            if not fixed_origin:
                x0s[i][0] += [np.median(V), 0.]
        else:
            x0s.append([x0])
            
        for c in range(4 + (0 if fixed_origin else 2)):
            x0s[i].append([x0s[0][0][0] * 2**(np.random.normal()),
                           x0s[0][0][1] * 1.2**(np.random.normal()),
                           x0s[0][0][2] + np.random.normal(scale = 0.3),
                           x0s[0][0][3] * 2**(np.random.normal())])
            if not fixed_origin:
                x0s[i][-1] += [x0s[0][0][4] + np.random.normal(scale = iqr(V, rng = [20,80]) / 10.),
                               x0s[0][0][5] + np.random.normal(scale = iqr(R, rng = [20,80]) / 10.)]
    res = []
    for i in range(n_walkers):
        res.append(minimize(Courteau97_Model_Loss,
                            x0 = x0s[i][0],
                            args = (R, V, E, fixed_origin),
                            method = 'Nelder-Mead',
                            options = {'initial_simplex': x0s[i]}))

    
    return list(min(res, key = lambda x: x.fun if np.isfinite(x.fun) else np.inf).x) + ([0.,0.] if fixed_origin else [])

#---------------------------------------------------------------------
def Line_Model_Evaluate(x, R):
    """

    x: 0 m_c - central slope [velocity / distance units]
       1 m_o - outer slope [velocity / distance units]
       2 r_t - transition radius [distance]
       3 v0 - x position of origin
       4 x0 - central velocity
    """
    
    res = np.zeros(len(R))

    y = R - x[4]
    
    res[np.abs(y) < x[2]] = x[0] * y[np.abs(y) < x[2]]
    res[y > x[2]] = x[2]*x[0] + x[1]*(y[y > x[2]] - x[2])
    res[y < -x[2]] = -x[2]*x[0] + x[1]*(y[y < -x[2]] + x[2])

    return res + x[3]
    

def Line_Model_Loss(x, R, V, E = None, fixed_origin = False):

    if E is None:
        E = np.ones(len(R))
    else:
        E = np.clip(E, a_min = 3, a_max = None)
    V_model = Line_Model_Evaluate((list(x) + [0.,0.]) if fixed_origin else x, R)
    CHOOSE = np.isfinite(V_model)
    if np.sum(CHOOSE) <= 0:
        return 1e9 * len(R)    

    losses = ((V - V_model)/E)[CHOOSE]
    N = np.argsort(losses)
    return np.mean(losses[N][int(0.1*len(losses)):int(0.9*len(losses))]**2) + 0.1*(0. if fixed_origin else (x[4]/10)**2) + 0.1*(x[2]/10)**2 + 0.1*(x[1]/5)**2

def Line_Model_Fit(R, V, E = None, fixed_origin = False, n_walkers = 10):

    """
    Fits the Kelsey Line model to a rotation curve. This emperical model
    is able to reflect some of observed rotation curves.
    R: Radius at which we would like the model velocity [any length units]
    V: Measured velocity values [any speed units]
    E: uncertainty on velocity measurements
    fixed_origin: boolean to indicate if model has origin at (0,0)
                  or is allowed to float [boolean]
    n_walkers: Number of times to attempt fit with random initialization
               parameters, avoids local minima [counts]

    returns: tuple with parameters for the Line model as described
             in Line_Model_Evaluate [various units]
    """
    
    N = np.argsort(V)
    tan_x = Tan_Model_Fit(R, V, E, fixed_origin)[0]
    x0s = [[tan_x[1]/tan_x[0], 0., tan_x[0]]]
    if not fixed_origin:
        x0s[0] += [tan_x[2], tan_x[3]]
    for i in range(n_walkers - 1):
        x0s.append([x0s[0][0] * 2**(np.random.normal()),
                    x0s[0][1] + np.random.normal(scale = 1./5.),
                    x0s[0][2] * 2**(np.random.normal())])
        if not fixed_origin:
            x0s[-1] += [x0s[0][3] + np.random.normal(scale = iqr(V, rng = [20,80]) / 10.),
                        x0s[0][4] + np.random.normal(scale = iqr(R, rng = [20,80]) / 10.)]
    res = []
    for i in range(n_walkers):
        res.append(minimize(Line_Model_Loss,
                            x0 = x0s[i],
                            args = (R, V, E, fixed_origin)))
    return (list(min(res, key = lambda x: x.fun if np.isfinite(x.fun) else np.inf).x) + ([0.,0.] if fixed_origin else []), x0s[0] + ([0.,0.] if fixed_origin else []))
