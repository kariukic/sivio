# Useful MS manipulation functions.
import numpy as np
from casacore.tables import table, maketabdesc, makearrcoldesc

c = 299792458


def get_data(tbl, col="DATA"):
    data = tbl.getcol(col)
    return data


def get_uvw(tbl):
    uvw = tbl.getcol("UVW")
    return uvw


def get_phase_center(tbl):
    """
    Grabs the phase centre of the observation in RA and Dec

    Parameters
    ----------
    tbl : casacore table.
        The casacore mset table opened with readonly=False.\n
    Returns
    -------
    float, float.
        RA and Dec in radians.
    """
    ra0, dec0 = tbl.FIELD.getcell("PHASE_DIR", 0)[0]
    return ra0, dec0


def get_channels(tbl, ls=True):
    if ls:
        chans = c / tbl.SPECTRAL_WINDOW.getcell("CHAN_FREQ", 0)
    else:
        chans = tbl.SPECTRAL_WINDOW.getcell("CHAN_FREQ", 0)
    return chans


def get_ant12(mset):
    ms = table(mset, readonly=False, ack=False)
    antenna1 = ms.getcol("ANTENNA1")
    antenna2 = ms.getcol("ANTENNA2")
    return antenna1, antenna2


def put_col(tbl, col, dat):
    """add data 'dat' to the column 'col'"""
    tbl.putcol(col, dat)


def add_col(tbl, colnme):
    """Add a column 'colnme' to the MS"""
    col_dmi = tbl.getdminfo("DATA")
    col_dmi["NAME"] = colnme
    shape = tbl.getcell("DATA", 0).shape
    tbl.addcols(
        maketabdesc(
            makearrcoldesc(colnme, 0.0 + 0.0j, valuetype="complex", shape=shape)
        ),
        col_dmi,
        addtoparent=True,
    )


<<<<<<< HEAD
def get_lmns(tbl, ra_rad, dec_rad):
=======
def get_lmns(tbl, ra_rad, dec_rad, phase_center_shift=0):
>>>>>>> 8a1d7cea8f4573227d5d5d90db629ac9d0763540
    """
    Calculating l, m, n values from ras,decs and phase centre.
    𝑙 = cos 𝛿 * sin Δ𝛼
    𝑚 = sin 𝛿 * cos 𝛿0 − cos 𝛿 * sin 𝛿0 * cos Δ𝛼
    Δ𝛼 = 𝛼 − 𝛼0
    """
    ra0, dec0 = get_phase_center(tbl)

<<<<<<< HEAD
=======
    if phase_center_shift != 0:
        print(
            f"shifting pahse center from {np.rad2deg(ra0)},{np.rad2deg(dec0)} \
            to {np.rad2deg(ra0)+phase_center_shift}, {np.rad2deg(dec0)+phase_center_shift} for testing.\
            Will ruin stuff!!!!!"
        )
        ra0 += np.deg2rad(phase_center_shift)
        dec0 += np.deg2rad(phase_center_shift)

>>>>>>> 8a1d7cea8f4573227d5d5d90db629ac9d0763540
    ra_delta = ra_rad - ra0
    ls = np.cos(dec_rad) * np.sin(ra_delta)
    ms = np.sin(dec_rad) * np.cos(dec0) - np.cos(dec_rad) * np.sin(dec0) * np.cos(
        ra_delta
    )
    ns = np.sqrt(1 - ls ** 2 - ms ** 2) - 1

    return ls, ms, ns


def get_bl_lens(mset):
    """Calculate the baseline length for each DATA row in the measurement set"""
    t = table(mset + "/ANTENNA", ack=False)
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
        bls[i] = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    return bls


def get_bl_vectors(mset, refant=0):
    """
    Gets the antenna XYZ position coordinates and recalculates them with the reference antenna as the origin.

    Parameters
    ----------
    mset : Measurement set. \n
    refant : int, optional
        The reference antenna ID, by default 0. \n

    Returns
    -------
    XYZ coordinates of each antenna with respect to the reference antenna.
    """
    # First get the positions of each antenna recorded in XYZ coordinates from the MS
    t = table(mset + "/ANTENNA", ack=False)
    pos = t.getcol("POSITION")
    t.close()

    no_ants = len(pos)
    print("The mset has %s antennas." % (no_ants))

    bls = np.zeros((no_ants, 3))
    for i in range(no_ants):  # calculate and fill bls with distances from the refant
        pos1, pos2 = pos[i], pos[refant]
        bls[i] = np.array([pos2 - pos1])
    return bls
