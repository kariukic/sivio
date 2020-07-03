# Useful ms manipulation snippets
import numpy as np
from casacore.tables import table, maketabdesc, makearrcoldesc
import matplotlib.pyplot as plt

c = 299792458


def get_data(tbl):
    data = tbl.getcol("DATA")
    return data


def get_uvw(tbl):
    uvw = tbl.getcol("UVW")
    return uvw


def get_phase_center(tbl):
    ra0, dec0 = tbl.FIELD.getcell('PHASE_DIR', 0)[0]
    print('The phase center is at ra=%s, dec=%s' % (np.degrees(ra0), np.degrees(dec0)))
    return ra0, dec0


def get_channels(tbl, ls=True):
    if ls:
        chans = c / tbl.SPECTRAL_WINDOW.getcell("CHAN_FREQ", 0)
    else:
        chans = tbl.SPECTRAL_WINDOW.getcell("CHAN_FREQ", 0)
    return chans


def put_col(tbl, col, dat):
    tbl.putcol(col, dat)


def add_col(tbl, colnme):
    col_dmi = tbl.getdminfo('DATA')
    col_dmi['NAME'] = colnme
    shape = tbl.getcell('DATA', 0).shape
    tbl.addcols(maketabdesc(makearrcoldesc(colnme, 0.0+0.0j,
                valuetype='complex', shape=shape)), col_dmi, addtoparent=True)


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
        bls[i] = np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)

    return bls


def plotuv(tbl):
    """Take a look at the uv plane"""
    uvw = get_uvw(tbl)
    plt.figure
    plt.plot(uvw[:, 0], uvw[:, 1], 'k.')
    # plt.plot(-1.*uvw[:,0], -1.*uvw[:,1], 'b.') #unhash to plot the complex conjugates
    plt.xlabel('u (m)')
    plt.ylabel('v (m)')
    plt.savefig('uuvv.png')
