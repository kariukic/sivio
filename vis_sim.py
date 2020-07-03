# Simulate visibilities and feed them into an MWA MS
from casacore.tables import table, maketabdesc, makearrcoldesc
import matplotlib.pyplot as plt
import add_scintillation
import numpy as np
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


def plotuv(tbl):
    """Take a look at the uv plane"""
    uvw = get_uvw(tbl)
    plt.figure
    plt.plot(uvw[:, 0], uvw[:, 1], 'k.')
    # plt.plot(-1.*uvw[:,0], -1.*uvw[:,1], 'b.') #unhash to plot the complex conjugates
    plt.xlabel('u (m)')
    plt.ylabel('v (m)')
    plt.savefig('uuvv.png')


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


def simulate_vis(mset, tbl, ras, decs, A, rdiff, clean_vis=False, offset=True, scintillate=True):
    """
    Compute the source visibilities and fix them in the measurement set.
    ras and decs should be lists or 1D arrays.
    A should have the flux for each source.
    """

    # grab uvw values from the ms and divide them with the wavelengths for each channel
    uvw = get_uvw(tbl)
    lmbdas = get_channels(tbl, ls=True)
    uvw_lmbdas = uvw[:, None, :] / lmbdas[None, :, None]

    # Now lets grab the data from the ms table
    data = get_data(tbl)

    # Delete all the grabbed data because we will put the simulated data there
    data[:] = 0

    # grab our lmns
    ls, ms, ns = get_lmns(tbl, ras, decs)
    # assert len(list(A)) == len(list(ls))

    # Now we compute the visibilities and add them up for each source. once for the xx and then yy columns in the MS.
    # for amps, l, m, n in [sources]: data[row, channel, pol] = A * exp(2pi i * (ul + vm + w(n - 1)))
    if clean_vis:
        for amp, l, m, n in zip(A, ls, ms, ns):
            # first compute the clean visibilities.
            d = amp * np.exp(2j * np.pi * (
                uvw_lmbdas[:, :, 0] * l + uvw_lmbdas[:, :, 1] * m + uvw_lmbdas[:, :, 2] * n
            ))
            data[:, :, 0] += d  # feed xx data
            data[:, :, 3] += d  # feed yy data

            put_col(tbl, "DATA", data)

    if scintillate:
        for amp, l, m, n in zip(A, ls, ms, ns):
            d = amp * np.exp(2j * np.pi * (
                uvw_lmbdas[:, :, 0] * l + uvw_lmbdas[:, :, 1] * m + uvw_lmbdas[:, :, 2] * n
            ))
            bls = get_bl_lens(mset)
            scintdat = add_scintillation.add_scint(d, bls, rdiff)
            data[:, :, 0] += scintdat  # feed xx data
            data[:, :, 3] += scintdat  # feed yy data

        if 'SCINT_DATA' not in tbl.colnames():
            print('Adding SCINT_DATA column in MS with simulated visibilities... ...')
            add_col(tbl, "SCINT_DATA")
            put_col(tbl, "SCINT_DATA", data)

    if offset:
        data[:] = 0
        for amp, l, m, n in zip(A, ls, ms, ns):
            phse = np.exp(2j * np.pi * (
                uvw_lmbdas[:, :, 0] * l + uvw_lmbdas[:, :, 1] * m + uvw_lmbdas[:, :, 2] * n
            ))

            # for i in range(data.shape[0]):
            #   phse[i,:] *= np.exp(2*np.pi*phasediff[i])
            #   phse[i,:] = phse[i,:] * np.exp(phasediff[i])
            phasediff = add_scintillation.add_phase_offsets(mset)
            # phasediff = np.ones(uvw_lmbdas.shape[0])*1j*2

            phse = phse * np.exp(2j * np.pi * phasediff)[:, None]
            data[:, :, 0] += amp * phse  # feed xx data
            data[:, :, 3] += amp * phse  # feed yy data

        if 'OFFSET_DATA' not in tbl.colnames():
            print('Adding OFFSET_DATA column in MS with offset visibilities... ...')
            add_col(tbl, "OFFSET_DATA")
            put_col(tbl, "OFFSET_DATA", data)

    print('---Tumemaliza--')


if __name__ == "__main__":
    mset = '1065880248_ionovis_1source_1jy_phasecenter.ms'
    tbl = table(mset, readonly=False)
    print(tbl.getcol("DATA").shape)

    ras = np.array([0])
    decs = np.array([-27.])
    fluxes = np.array([1.])

    simulate_vis(tbl, ras, decs, fluxes)
    tbl.close()
