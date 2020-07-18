# Simulate visibilities and feed them into an MWA MS
import numpy as np
from casacore.tables import table

from phase_screen import get_tec_value
from scint_equations import add_scint
import mset_utils as mtls
from coordinates import radec_to_altaz, MWAPOS


def sim_prep(tbl, ras, decs):
    """Preparing the requiremnts before simulating"""

    # grab uvw values from the ms and divide them with the wavelengths for each channel
    uvw = mtls.get_uvw(tbl)
    lmbdas = mtls.get_channels(tbl, ls=True)
    uvw_lmbdas = uvw[:, None, :] / lmbdas[None, :, None]

    # Now lets grab the data from the ms table
    data = mtls.get_data(tbl)
    print("Shape of the MS data", data.shape)

    # Delete all the grabbed data because we will put the simulated data there
    data[:] = 0

    # grab our lmns
    ls, ms, ns = mtls.get_lmns(tbl, ras, decs)
    # assert len(list(A)) == len(list(ls))

    return data, uvw_lmbdas, ls, ms, ns


def add_phase_offsets(mset, params):
    """Compute the phase differences for each baseline.

    Parameters
    ----------
    mset :  Measurement set.\n
    params : 1D array of len(no. of antennas.)
        The phase offset value for each antenna.

    Returns
    -------
    array.
        The phase difference per baseline.
    """
    mset = table(mset, readonly=False, ack=False)
    antenna1 = mset.getcol("ANTENNA1")
    antenna2 = mset.getcol("ANTENNA2")
    # freqs = mset.SPECTRAL_WINDOW.getcell('CHAN_FREQ', 0)
    # Antennas runs from 1-128
    # antids = np.array(range(0, len(mset.ANTENNA)))

    # from phase_screen import get_antenna_in_uvw
    # us, vs, ws = get_antenna_in_uvw(ms, mset)
    # ds = vs
    # params = 0.0008 * ds

    # params = run_all(ms,)
    # print(params[antenna1] - params[antenna2])
    return params[antenna1] - params[antenna2]

    # ant1_coef = np.zeros((len(antenna1), 768, 3))
    # ant2_coef = np.zeros((len(antenna2), 768, 3))

    # for a in range(len(antids)):
    #    for f in range(768):
    #        ant1_coef[antenna1==a,f] = params[a]
    #       ant2_coef[antenna2==a,f] = -params[a]

    # return ant1_coef + ant2_coef

    # phase1 = ant1_coef[:,:,0] +
    #           ant1_coef[:,:,1] * l * uvw_lmbdas[:, :, 0] +
    #           ant1_coef[:,:,2]*m*uvw_lmbdas[:, :, 1] +
    #           uvw_lmbdas[:, :, 2] * n
    # phase2 = ant2_coef[:,:,0] +
    #           ant2_coef[:,:,1] * l * uvw_lmbdas[:, :, 0] +
    #           ant2_coef[:,:,2] * m *uvw_lmbdas[:, :, 1] +
    #           uvw_lmbdas[:, :, 2] * n
    # phase2 = ant2_coef[:,0] + l*ant2_coef[:,1] + m*ant2_coef[:,2] * n

    # phasediff = 1j*(phase1 - phase2)
    # phasediff = phase1 - phase2
    # print(phasediff[:50], '######phasediff#####')
    # print(phasediff.shape, '-----------------------phasediff')

    # return phasediff


def true_vis(data, uvw_lmbdas, A, ls, ms, ns):
    for amp, l, m, n in zip(A, ls, ms, ns):
        # first compute the clean visibilities.
        d = amp * np.exp(
            2j
            * np.pi
            * (
                uvw_lmbdas[:, :, 0] * l
                + uvw_lmbdas[:, :, 1] * m
                + uvw_lmbdas[:, :, 2] * n
            )
        )
        data[:, :, 0] += d  # feed xx data
        data[:, :, 3] += d  # feed yy data

    return data


def offset_vis(
    mset,
    data,
    uvw_lmbdas,
    A,
    ls,
    ms,
    ns,
    ras,
    decs,
    phs_screen,
    time,
    us,
    vs,
    scale,
    h_pix,
    pp_u_offset,
    pp_v_offset,
):
    """Offset visibilities"""
    data[:] = 0
    source_ppoints = []
    source_params = []

    source_count = 1
    for amp, l, m, n, ra, dec in zip(A, ls, ms, ns, ras, decs):
        print("Source: ", source_count)
        phse = np.exp(
            2j
            * np.pi
            * (
                uvw_lmbdas[:, :, 0] * l
                + uvw_lmbdas[:, :, 1] * m
                + uvw_lmbdas[:, :, 2] * n
            )
        )
        # for i in range(data.shape[0]):
        #   phse[i,:] *= np.exp(2*np.pi*phasediff[i])
        #   phse[i,:] = phse[i,:] * np.exp(phasediff[i])
        # phasediff = add_phase_offsets(mset)
        # phasediff = np.ones(uvw_lmbdas.shape[0])*1j*2

        alt, azimuth = radec_to_altaz(ra, dec, time, MWAPOS)

        # Added -0.1890022463989236 radians factor to center my sky because at ms phase center the
        # zenith angle is off by that fatctor. Investigate!!
        zen_angle = np.pi / 2.0 - alt  # - 0.1890022463989236
        # azimuth -= 1.611163115052922  # applying same correction factor for azimuth
        print("Zenith angle: ", np.rad2deg(zen_angle), "deg  OR", zen_angle, "rad")
        print("Azimuth angle: ", np.rad2deg(azimuth), "deg  OR", azimuth, "rad")

        u_tec_list, v_tec_list, tec_per_ant = get_tec_value(
            phs_screen,
            us,
            vs,
            zen_angle,
            azimuth,
            scale,
            h_pix,
            pp_u_offset,
            pp_v_offset,
        )
        params = tec_per_ant  # * 10 128 phasescreen values one for each pierce point
        phasediff = add_phase_offsets(mset, params)

        source_ppoints.append(np.stack((u_tec_list, v_tec_list)))
        source_params.append(np.stack(params))

        phse = phse * np.exp(2j * np.pi * phasediff)[:, None]
        data[:, :, 0] += amp * phse  # feed xx data
        data[:, :, 3] += amp * phse  # feed yy data

        source_count += 1

    # Lets save the x and y coordinates, the tec params and phasediffs
    npz = mset.split(".")[0] + "_pierce_points.npz"
    np.savez(npz, ppoints=source_ppoints, params=source_params, tecscreen=phs_screen)

    return data


def scint_vis(mset, data, uvw_lmbdas, A, ls, ms, ns, rdiff):
    data[:] = 0
    for amp, l, m, n in zip(A, ls, ms, ns):
        d = amp * np.exp(
            2j
            * np.pi
            * (
                uvw_lmbdas[:, :, 0] * l
                + uvw_lmbdas[:, :, 1] * m
                + uvw_lmbdas[:, :, 2] * n
            )
        )
        bls = mtls.get_bl_lens(mset)
        scintdat = add_scint(d, bls, rdiff)
        data[:, :, 0] += scintdat  # feed xx data
        data[:, :, 3] += scintdat  # feed yy data
    return data


"""
def simulate_vis(
    mset, tbl, ras, decs, A, rdiff, clean_vis=False, offset=True, scintillate=False
):
    '''
    Compute the source visibilities and fix them in the measurement set.
    ras and decs should be lists or 1D arrays.
    A should have the flux for each source.
    '''

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
            d = amp * np.exp(
                2j
                * np.pi
                * (
                    uvw_lmbdas[:, :, 0] * l
                    + uvw_lmbdas[:, :, 1] * m
                    + uvw_lmbdas[:, :, 2] * n
                )
            )
            data[:, :, 0] += d  # feed xx data
            data[:, :, 3] += d  # feed yy data

            mtls.put_col(tbl, "DATA", data)

    if scintillate:
        for amp, l, m, n in zip(A, ls, ms, ns):
            d = amp * np.exp(
                2j
                * np.pi
                * (
                    uvw_lmbdas[:, :, 0] * l
                    + uvw_lmbdas[:, :, 1] * m
                    + uvw_lmbdas[:, :, 2] * n
                )
            )
            bls = mtls.get_bl_lens(mset)
            scintdat = add_scint(d, bls, rdiff)
            data[:, :, 0] += scintdat  # feed xx data
            data[:, :, 3] += scintdat  # feed yy data

        if "SCINT_DATA" not in tbl.colnames():
            print("Adding SCINT_DATA column in MS with simulated visibilities... ...")
            mtls.add_col(tbl, "SCINT_DATA")
            mtls.put_col(tbl, "SCINT_DATA", data)

    if offset:
        data[:] = 0
        for amp, l, m, n in zip(A, ls, ms, ns):
            phse = np.exp(
                2j
                * np.pi
                * (
                    uvw_lmbdas[:, :, 0] * l
                    + uvw_lmbdas[:, :, 1] * m
                    + uvw_lmbdas[:, :, 2] * n
                )
            )

            # for i in range(data.shape[0]):
            #   phse[i,:] *= np.exp(2*np.pi*phasediff[i])
            #   phse[i,:] = phse[i,:] * np.exp(phasediff[i])
            phasediff = add_phase_offsets(mset)
            # phasediff = np.ones(uvw_lmbdas.shape[0])*1j*2

            phse = phse * np.exp(2j * np.pi * phasediff)[:, None]
            data[:, :, 0] += amp * phse  # feed xx data
            data[:, :, 3] += amp * phse  # feed yy data

        if "OFFSET_DATA" not in tbl.colnames():
            print("Adding OFFSET_DATA column in MS with offset visibilities... ...")
            mtls.add_col(tbl, "OFFSET_DATA")
            mtls.put_col(tbl, "OFFSET_DATA", data)

    print("---Tumemaliza--")
"""
