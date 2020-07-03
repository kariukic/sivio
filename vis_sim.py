# Simulate visibilities and feed them into an MWA MS
import numpy as np

from add_iono_effects import add_scint, add_phase_offsets
import mset_utils as mtls


def simulate_vis(mset, tbl, ras, decs, A, rdiff, clean_vis=False, offset=True, scintillate=True):
    """
    Compute the source visibilities and fix them in the measurement set.
    ras and decs should be lists or 1D arrays.
    A should have the flux for each source.
    """

    # grab uvw values from the ms and divide them with the wavelengths for each channel
    uvw = mtls.get_uvw(tbl)
    lmbdas = mtls.get_channels(tbl, ls=True)
    uvw_lmbdas = uvw[:, None, :] / lmbdas[None, :, None]

    # Now lets grab the data from the ms table
    data = mtls.get_data(tbl)

    # Delete all the grabbed data because we will put the simulated data there
    data[:] = 0

    # grab our lmns
    ls, ms, ns = mtls.get_lmns(tbl, ras, decs)
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

            mtls.put_col(tbl, "DATA", data)

    if scintillate:
        for amp, l, m, n in zip(A, ls, ms, ns):
            d = amp * np.exp(2j * np.pi * (
                uvw_lmbdas[:, :, 0] * l + uvw_lmbdas[:, :, 1] * m + uvw_lmbdas[:, :, 2] * n
            ))
            bls = mtls.get_bl_lens(mset)
            scintdat = add_scint(d, bls, rdiff)
            data[:, :, 0] += scintdat  # feed xx data
            data[:, :, 3] += scintdat  # feed yy data

        if 'SCINT_DATA' not in tbl.colnames():
            print('Adding SCINT_DATA column in MS with simulated visibilities... ...')
            mtls.add_col(tbl, "SCINT_DATA")
            mtls.put_col(tbl, "SCINT_DATA", data)

    if offset:
        data[:] = 0
        for amp, l, m, n in zip(A, ls, ms, ns):
            phse = np.exp(2j * np.pi * (
                uvw_lmbdas[:, :, 0] * l + uvw_lmbdas[:, :, 1] * m + uvw_lmbdas[:, :, 2] * n
            ))

            # for i in range(data.shape[0]):
            #   phse[i,:] *= np.exp(2*np.pi*phasediff[i])
            #   phse[i,:] = phse[i,:] * np.exp(phasediff[i])
            phasediff = add_phase_offsets(mset)
            # phasediff = np.ones(uvw_lmbdas.shape[0])*1j*2

            phse = phse * np.exp(2j * np.pi * phasediff)[:, None]
            data[:, :, 0] += amp * phse  # feed xx data
            data[:, :, 3] += amp * phse  # feed yy data

        if 'OFFSET_DATA' not in tbl.colnames():
            print('Adding OFFSET_DATA column in MS with offset visibilities... ...')
            mtls.add_col(tbl, "OFFSET_DATA")
            mtls.put_col(tbl, "OFFSET_DATA", data)

    print('---Tumemaliza--')
