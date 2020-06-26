import os
from casacore.tables import table
import matplotlib.pyplot as plt
import scint_equations as sqs
import add_scintillation
import vis_sim
import numpy as np
from powerbox import PowerBox
c = 299792458

def tec_screen(p, resolution=1024):
    pb = PowerBox(resolution, lambda k: 10*k**p, ensure_physical=True)
    return pb.delta_x()/10e16

def iono_phase_shift(channels,p):
    """
    -1/8*pi * e/nm_e * 1/freq**2 * tec
    """
    tec = tec_screen(p)
    print(tec)
    phase_screen = (-1/(8*np.pi)) * (np.e/(8.85418782e-12 * 9.10938356e-31)) * (1/channels**2) * tec
    return phase_screen


def get_bl_lens(mset):
    """Calculate the baseline length for each DATA row in the measurement set"""
    t = table(mset+"/ANTENNA", ack=False)
    pos = t.getcol("POSITION")
    t.close()


    tt = table(mset)
    ant1 = tt.getcol("ANTENNA1")
    ant2 = tt.getcol("ANTENNA2")
    tt.close()

    bls = np.zeros(len(ant1))
    for i in range(len(ant1)):
        p = ant1[i]
        q = ant2[i]
        pos1, pos2 = pos[p], pos[q]
        plt.plot(pos1[0],pos1[1])
        bls[i] = np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)
    plt.savefig("antpos.png")
    return bls

def get_phase_center(tbl):
    ra0, dec0 = tbl.FIELD.getcell('PHASE_DIR', 0)[0]
    print('The phase center is at ra=%s, dec=%s'%(np.degrees(ra0),np.degrees(dec0)))
    return ra0, dec0

def get_lmns(tbl, ras, decs):
    """
    Calculating l, m, n values from ras,decs and phase centre.
    𝑙 = cos 𝛿 * sin Δ𝛼
    𝑚 = sin 𝛿 * cos 𝛿0 − cos 𝛿 * sin 𝛿0 * cos Δ𝛼
    Δ𝛼 = 𝛼 − 𝛼0
    """
    ra0, dec0 = get_phase_center(tbl)

    ra_rad = np.radians(ras)
    dec_rad = np.radians(decs)
    ra_delta = ra_rad - ra0
    ls = np.cos(dec_rad)*np.sin(ra_delta)
    ms = (np.sin(dec_rad)*np.cos(dec0)-np.cos(dec_rad)*np.sin(dec0)*np.cos(ra_delta))
    ns = np.sqrt(1 - ls**2 - ms**2) - 1

    return ls, ms, ns


def antenna_pos(mset):
    t = table(mset+"/ANTENNA", ack=False)
    pos = t.getcol("POSITION")
    t.close()
    return pos

def get_bl_vectors(mset, refant = 0):
    """Calculate the baseline length for each DATA row in the measurement set"""
    pos = antenna_pos(mset) #First get the positions of each antenna recorded in XYZ values
    no_ants = len(pos)
    print('The mset has %s antennas.' % (no_ants))

    bls = np.zeros((no_ants, 3))
    for i in range(no_ants): #calculate and fill bls with distances from the refant
        pos1, pos2 = pos[i], pos[refant]
        bls[i] = np.array([pos2-pos1])
    return bls

def get_antenna_in_uvw(mset, tbl):
    """
    Convert the antennas positions into some sort of uv plane with one antenna as the center of the field.
    """
    ra0, dec0 = get_phase_center(tbl)
    #ra0*=-1 #because we have assumed local LST=0hrs and HA = LST-RA
    xyz = get_bl_vectors(mset)

    us = np.sin(ra0)*xyz[:,0] + np.cos(ra0)*xyz[:,1]
    vs = -np.sin(dec0)*np.cos(ra0)*xyz[:,0] + np.sin(dec0)*np.sin(ra0)*xyz[:,1] + np.cos(dec0)*xyz[:,2]
    ws = np.cos(dec0)*np.cos(ra0)*xyz[:,0] + -np.cos(dec0)*np.sin(ra0)*xyz[:,1] + np.sin(dec0)*xyz[:,2]

    return us, vs, ws



if __name__ == "__main__":
    ras = [0,1] #generate_distribution(0., 4., N, "normal") #np.array([0,2])
    decs = [-27,-20]
    mset = "1000sources_truevissim.ms"
    tbl = table(mset, readonly=False)
    #bls = get_bl_lens(mset)
    #print(bls.max())

    #uvw = vis_sim.get_uvw(tbl)
    #print(uvw.shape)
    #channels  = vis_sim.get_channels(tbl,ls=False)
    #print(channels.shape)

    #phase_scrn = iono_phase_shift(channels,-1.667)
    pos = antenna_pos(mset)
    print(pos[0,:])

    us, vs, ws = get_antenna_in_uvw(mset, tbl)
    print(us[:3], vs[:3], ws[:3])
    plt.scatter(us,vs, marker='x', linestyle='None', label='Antenna positions')
    plt.plot(us[0],vs[0], 'rx', label='Reference antenna')
    plt.ylabel('V distance (m)')
    plt.xlabel('U distance (m)')
    plt.legend()
    plt.title('Antenna positions projected onto the UV plane')
    plt.savefig('antennas_on_uvplane.png')