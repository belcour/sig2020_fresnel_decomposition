import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

#############################################################################
#                              General utils
#############################################################################

def clamp(theta_t):
    temp = np.where(theta_t < 0.0, 0.0, theta_t)
    return np.where(temp > np.pi/2.0, np.pi/2.0, temp)

def least_squares(reference, func_to_compare):
    return sum(pow(reference - func_to_compare, 2))


#############################################################################
#                              Fresnel
#############################################################################

def polarized_fresnel(CosTheta, Eta):
    """ Code from Seb. Lagarde's blog :
        Compute Fresnel formula for a given direction and ior
        ior : eta_t / eta_i = 1.0 / eta_coat
    """
    SinTheta2 = 1 - CosTheta * CosTheta
    test = (SinTheta2 / (Eta * Eta))
    temp = 1 - (SinTheta2 / (Eta * Eta))
    tempo = np.where(temp < 0, 1.0, temp)
    t0 = np.sqrt(tempo);
    t1 = Eta * t0;
    t2 = Eta * CosTheta;
    rs = (CosTheta - t1) / (CosTheta + t1);
    rp = (t0 - t2) / (t0 + t2);
    return np.where(temp < 0, 1.0, rs*rs), np.where(temp < 0, 1.0, rp*rp)

def fresnel_dielectric_conductor(Eta, Etak, CosTheta, polarized=False):
    CosTheta2 = CosTheta * CosTheta;
    SinTheta2 = 1 - CosTheta2;
    Eta2 = Eta * Eta;
    Etak2 = Etak * Etak;

    t0 = Eta2 - Etak2 - SinTheta2;
    a2plusb2 = np.sqrt(t0 * t0 + 4 * Eta2 * Etak2);
    t1 = a2plusb2 + CosTheta2;
    temp = 0.5 * (a2plusb2 + t0)
    temp = np.where(temp > 0.0, temp, 1.0)
    
    a = np.sqrt(temp);
    t2 = 2 * a * CosTheta;
    Rs = (t1 - t2) / (t1 + t2);

    t3 = CosTheta2 * a2plusb2 + SinTheta2 * SinTheta2;
    t4 = t2 * SinTheta2;   
    Rp = Rs * (t3 - t4) / (t3 + t4);

    if polarized:
        return Rs, Rp
    return 0.5 * (Rs + Rp);

def schlick_approx(theta, eta0, eta1): # for dielectrics only, without TIR
    R0 = R_0(eta0, eta1)
    return R0 + (1- R0)*pow(1 - np.cos(theta), 5)

def fresnel(CosTheta, Eta, kappa=0.0, polarized=False):
    CosTheta = np.maximum(CosTheta, 1.0E-8)
    if kappa == 0.0:
        Rs, Rp = polarized_fresnel(CosTheta, Eta)
        if polarized:
            return Rs, Rp
        return 0.5 * (Rs + Rp);
    return fresnel_dielectric_conductor(Eta, kappa, CosTheta, polarized)

def R_0(eta0, eta1):
    return ((eta1 - eta0)/(eta1 + eta0))**2

def eta_from_R0(R, eta1):
    """ Invert Fresnel at normal incidence :
        Solve (eta1 - x)^2 / (eta1 + x)^2 = R """
    etap = eta1; etam = eta1
    if R != 0.0:
        etap = eta1 * (1.0+np.sqrt(R))/(1-np.sqrt(R))
        etam = eta1 * (1.0-np.sqrt(R))/(1+np.sqrt(R))
    return etam, etap

def solve_eta_from_R(R, eta1):
    return eta_from_R0(R, eta1)

def compute_kappa_from_eta(R0, eta0, eta1):
    temp = (R0*(eta0 +eta1)**2 - (eta0 - eta1)**2)/(1.0 - R0)
    return np.sqrt(temp) if temp > 0.0 else -1.0


#############################################################################
#                              Transmission
#############################################################################

def transmitted(angle, e0, e1):
    temp = e0/e1 * np.sin(angle)
    theta_t = np.where(temp>=0, np.arcsin(temp), -1)
    return theta_t 

def brewster(eta0, eta1):
    return np.arctan(eta1/eta0)


#############################################################################
#                              Multi-layers
#############################################################################

def add(R01, T01, R10, R12):
    #return R01 + T01 * R12 #/ (1.0 - R12 * R10)
    return R01 + T01**2 * R12 / (1.0 - R12 * R10)

def add_with_absorption(R01, R12, tau, transmitted_angle):
    return R01 + (1-R01)**2 * R12 / (np.exp(2*tau/np.cos(transmitted_angle)) - R12 * R01)
    #return R01 + (1-R01)**2 * R12 * np.exp(-2*tau/np.cos(transmitted_angle))

def substract(eta_0, eta_1, R02, angle = 0.0, tau1=0.0):
    """ Return R_12 equivalent at normal incidence with i and j two consecutive layers """
    R01 = fresnel(np.cos(angle), eta_1/eta_0) # 1.0 because normal incidence
    return (R02 - R01)*np.exp(2*tau1/np.cos(angle))/(1.0 + (R02 - 2.0)*R01) # all bounces, cheaper evaluation

def get_base_reflectance(eta_0, eta_1, R02, angle = 0.0, tau1=0.0):
    return substract(eta_0, eta_1, R02, angle, tau1)

def adding_equation(thetas, eta0, eta1, eta2, eta3, kappa3 = 0.0, polarized=False, tau1=0.0, tau2=0.0):
    """ Return the reflectance of a 4 layers material (3 interfaces)
        with all inter-reflections, using adding equation """
    zeros = [np.zeros_like(thetas),np.zeros_like(thetas)] if polarized else np.zeros_like(thetas)
    R01 = fresnel(np.cos(thetas), eta1/eta0, polarized=polarized) if eta1 != eta0 else zeros
    ones = np.ones_like(R01)
    
    T01 = ones - R01
    thetas_t1 = clamp(np.arcsin(eta0 / eta1 * np.sin(thetas)))
    thetas_t1 = np.where(thetas_t1 is not np.nan, thetas_t1, 0.0);
    R10 = fresnel(np.cos(thetas_t1), eta0/eta1, polarized=polarized) if eta1 != eta0 else zeros

    R12 = fresnel(np.cos(thetas_t1), eta2/eta1, polarized=polarized) if eta1 != eta2 else zeros
    T12 = ones - R12
    thetas_t2 = clamp(np.arcsin(eta1/eta2 * np.sin(thetas_t1)))
    thetas_t2 = np.where(thetas_t2 is not np.nan, thetas_t2, 0.0);
    R21 = fresnel(np.cos(thetas_t2), eta1/eta2, polarized=polarized) if eta1 != eta2 else zeros

    k = 0.0 if kappa3 == 0.0 else kappa3/eta2
    R23 = fresnel(np.cos(thetas_t2), eta3/eta2, k, polarized=polarized)  
    if polarized:
        res = []
        for i in range(2):    
            R13 = add_with_absorption(R12[i], R23[i], tau2, thetas_t2)
            R03 = add_with_absorption(R01[i], R13,    tau1, thetas_t1)
            #R13 = add(R12[i], T12[i], R21[i], R23[i])
            #R03 = add(R01[i], T01[i], R10[i], R13)
            res.append(np.where(np.isfinite(R03), R03, ones[0]))
        return res
    
    #R13 = add(R12, T12, R21, R23)
    #R03 = add(R01, T01, R10, R13)
    R13 = add_with_absorption(R12, R23, tau2, thetas_t2)
    R03 = add_with_absorption(R01, R13, tau1, thetas_t1)
    return np.where(np.isfinite(R03), R03, 1.0)

def approx_adding_bounces_3interfaces(thetas, eta0, eta1, eta2, eta3, k1, k2):
    R01 = fresnel(np.cos(thetas), eta1/eta0)
    theta_t1 = clamp(np.arcsin(eta0/eta1 * np.sin(thetas)))
    R10 = fresnel(np.cos(theta_t1), eta0/eta1)
    R12 = fresnel(np.cos(theta_t1), eta2/eta1)
    theta_t2 = clamp(np.arcsin(eta1/eta2 * np.sin(theta_t1)))
    R23 = fresnel(np.cos(theta_t2), eta3/eta2)
    R21 = fresnel(np.cos(theta_t2), eta1/eta2)
    mysum = 0.0
    for k in range(0,int(np.floor(k2))):
        mysum += (R23 * R21)**k
    R = R12 + mysum * (1 - R12)**2 * R23
    mysum = 0.0
    for k in range(0,int(np.floor(k1))):
        mysum += (R * R10)**k
    return R01 + (1 - R01)**2 * R * mysum

def get_curves_normal_incidence(eta0, eta1, R, theta, withSolutions=False):
    # Compute the corresponding value of eta2
    R12 = get_base_reflectance(eta0, eta1, R)
    if R12 <= 0.0:
        if withSolutions:
            return  [-1, -1], [np.full_like(theta, -1), np.full_like(theta, -1)]
        return np.zeros_like(theta), np.zeros_like(theta)
    solutions = solve_eta_from_R(R12, eta1)
    r1 = adding_equation(theta, eta0, eta0, eta1, solutions[0])
    r2 = adding_equation(theta, eta0, eta0, eta1, solutions[1])
    if withSolutions: 
        return [solutions[0], solutions[1]], [r1, r2] 
    return r1, r2 


#############################################################################
#                              2D parametric curve
#############################################################################

def get_eta_min_max(R02, eta0):
    """ Return minimal and maximal values of eta1, i.e. points A and E"""
    return solve_eta_from_R(R02, eta0)
    
def get_eta_minus_plus(R02, eta0):
    """ Return eta1 of BH and DF """
    R01 = R02 / (2.0 - R02)
    return get_eta_min_max(R01, eta0)

def get_eta2_minus_plus(R02, eta0):
    """ Return maximal and minimal value of eta2 - in H and D """
    return pow(get_eta_minus_plus(R02, eta0), 2)/eta0

def get_E(R, eta0):
    e = max(solve_eta_from_R(R, eta0))
    return e, e

def get_A(R, eta0):
    e = min(solve_eta_from_R(R, eta0))
    return e, e

def get_C(R, eta0):
    e = max(solve_eta_from_R(R, eta0))
    return 1.0, e

def get_G(R, eta0):
    e = min(solve_eta_from_R(R, eta0))
    return 1.0, e

def get_B(R, eta0):
    e = min(solve_eta_from_R(R/(2-R), eta0))
    return e, 1.0

def get_F(R, eta0):
    e = max(solve_eta_from_R(R/(2-R), eta0))
    return e, 1.0

def get_D(R, eta0):
    e = max(solve_eta_from_R(R/(2-R), eta0))
    return e, e*e

def get_H(R, eta0):
    e = min(solve_eta_from_R(R/(2-R), eta0))
    return e, e*e

#############################################################################

def get_eta_from_t(t, eC, eE, eF):
    if t < 0.5:
        return eC * (1 - 2*t) + eE*2*t
    return eE * 2 * (1 - t) + eF*(2*t - 1)

def get_solution(t, R12, n1):
    mini, maxi = solve_eta_from_R(R12, n1)
    return maxi if t < 0.5 else mini
    
def t_from_eta_old(eta, eta_min, eta_max):
    tp = (eta - eta_min)/2.0/(eta_max - eta_min)
    tm = (2*eta_max - eta - eta_min)/2.0/(eta_max - eta_min)
    return (tm, tp)

def t_from_eta(eta, etaC, etaE, etaF):
    tp = (eta - etaC)/2.0/(etaE - etaC)
    tm = (2*etaE + eta + etaF)/2.0/(etaF - etaE)
    return (tm, tp)

def find_eta2_from_eta1_parametric(t, eta0, R, theta, eta_min, eta_mean, eta_max):
    # Compute the corresponding value of eta2
    eta1 = get_eta_from_t(t, eta_min, eta_mean, eta_max)
    R12 = get_base_reflectance(eta0, eta1, R)
    if R12 <= 0.0:
        return False# no root because the first interface reflects too much
    solutions = solve_eta_from_R(R12, eta1)
    s = solutions[1] if t < 0.5 else solutions[0]
    return True, eta1, s

def get_parametric_curves_normal_incidence(t, eta0, R, theta, eta_min, eta_mean, eta_max, polarized=False):
    isOK, eta1, eta2 = find_eta2_from_eta1_parametric(t, eta0, R, theta, eta_min, eta_mean, eta_max)
    if isOK:
        if eta2 > 1.0:
            return adding_equation(theta, eta0, eta0, eta1, eta2, polarized=polarized)
   
    # avoid to display the negative quadrant
    return [np.full_like(theta, -1),np.full_like(theta, -1)] if polarized else np.full_like(theta, -1)


#############################################################################
#                              Sliders functions
#############################################################################

# Sliders parameters:
axcolor = 'whitesmoke'
HEIGHT = 0.02
TEXT_SIZE = 0.05
HSPACE = 0.03
VSPACE = 0.1
XPADDING = 0.15
WIDTH = 1.0 - 2* XPADDING

def init(n_hsliders, n_vsliders):
    length = (WIDTH - float(n_vsliders - 1) * VSPACE) /  float(n_vsliders)
    x = []; y = []
    for i in range(n_vsliders):
        x.append(0.02+ XPADDING + i * (VSPACE + length))
    for i in reversed(range(n_hsliders)):
        y.append((i+1) * HSPACE + i * HEIGHT)
    bottom = TEXT_SIZE + float(n_hsliders) * HEIGHT + float(n_hsliders+1) * HSPACE
    return length, x, y, bottom

def add_slider(x_start, y_start, initval, label, col, length, minval, maxval):
    axeta = plt.axes([x_start, y_start, length, HEIGHT], facecolor=axcolor) #position of the slider
    return Slider(axeta, label, minval, maxval, valinit=initval, color=col) #values in the slider
