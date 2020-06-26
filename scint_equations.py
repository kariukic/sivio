import numpy as np

"""
A collection of various ionospheric scintillation equations
Derived in Vendantham et al. 2015 & 2016 
"""
phs_var=5.87
k_0=100
c = 299792458



def rd(phs_var, k_0):
    """The diffractive scale. It is a function of the phase variance and the PS outer scale"""
    return (1/(np.pi*k_0))*(scp.gamma(11/6)/(2*scp.gamma(1/6)*phs_var))**(3/5)

def structure_fn(diff_0r, phs_var):
    return [2*i*(phs_var**2) for i in diff_0r]
   

def structure_fn_approx(r, rdiff=5000):
    return (r/rdiff)**(5/3)


def fresnel_scale(freq, h=300000): #Fresnel scale
    lam = c/freq
    return np.sqrt((lam*h)/(2*np.pi))


def s_eff(nu, pb=30): #effective scintillating flux
    return 5.86*((pb/30)**-1.5)*((nu/150)**-2.025)


def scint_noise_var(s_eff,b, r_d=5):
    return (s_eff**2)*(b/r_d)**(5/3)


def corr_time(b, freq=150000000, v=27.778): #v = turbulent plasma velocity in m/s
    r_f = fresnel_scale(freq)
    for i in b:
        if i<r_f:
            sc = r_f
        else: sc= b
        q = 2*sc/v
    q[:int(r_f)-1] = 2*r_f/v
    return r_f, q


def thermal_noise(t, lamda_nu=0.04*10**6, sefd=1200): #sky noise only
    return sefd/(np.sqrt(2*lamda_nu*t))