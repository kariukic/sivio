# Simulate visibilities and feed them into an MWA MS
import numpy as np

# from numba import njit, float64, complex64, prange

from scint_equations import add_scint
import mset_utils as mtls


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

    return data, lmbdas, uvw_lmbdas, ls, ms, ns


def add_phase_offsets(antenna1, antenna2, params):
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
    phasediffs = []
    for param in params:
        # freqs = mset.SPECTRAL_WINDOW.getcell('CHAN_FREQ', 0)
        # Antennas runs from 1-128
        # antids = np.array(range(0, len(mset.ANTENNA)))

        # from phase_screen import get_antenna_in_uvw
        # us, vs, ws = get_antenna_in_uvw(ms, mset)
        # ds = vs
        # params = 0.0008 * ds

        # params = run_all(ms,)
        # print(params[antenna1] - params[antenna2])
        diff = param[antenna1] - param[antenna2]

        # ant1_coef = np.zeros((len(antenna1), 768, 3))
        # ant2_coef = np.zeros((len(antenna2), 768, 3))

        # for a in range(len(antids)):
        #    for f in range(768):
        #        ant1_coef[antenna1==a,f] = params[a]
        #        ant2_coef[antenna2==a,f] = -params[a]

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
        phasediffs.append(diff)
    return phasediffs


def add_phase_offsets2(antenna1, antenna2, u_params, v_params):
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
    u_phasediffs, v_phasediffs = [], []
    for uparam in u_params:
        diff_u = uparam[antenna1] - uparam[antenna2]
        u_phasediffs.append(diff_u)

    for vparam in v_params:
        diff_v = vparam[antenna1] - vparam[antenna2]
        v_phasediffs.append(diff_v)

    return np.array(u_phasediffs), np.array(v_phasediffs)


# Before introducing numba.
"""
@njit(
    [
        complex64[:, :, :](
            complex64[:, :, :],
            float64[:, :, :],
            float64[:],
            float64[:],
            float64[:],
            float64[:],
        )
    ],
    parallel=True,
)
"""


def true_vis(data, uvw_lmbdas, A, ls, ms, ns):
    data[:] = 0
    source_count = 1
    for amp, l, m, n in zip(A, ls, ms, ns):
        print("True Source: ", source_count, "...")
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
        source_count += 1
    return data


"""
@njit(
    [
        complex64[:, :, :](
            complex64[:, :, :],
            float64[:, :, :],
            float64[:],
            float64[:],
            float64[:],
            float64[:],
        )
    ],
    parallel=True,
)
def true_vis(data, uvw_lmbdas, A, ls, ms, ns):
    data[:] = 0
    u = uvw_lmbdas[:, :, 0]
    v = uvw_lmbdas[:, :, 1]
    w = uvw_lmbdas[:, :, 2]
    xx = data[:, :, 0]
    yy = data[:, :, 3]
    source_count = 1
    for source in prange(len(A)):
        print("True Source: ", source_count, "...")
        # first compute the clean visibilities.
        d = A[source] * np.exp(
            2j * np.pi * (u * ls[source] + v * ms[source] + w * ns[source])
        )
        xx += d  # feed xx data
        yy += d  # feed yy data
        source_count += 1
    data[:, :, 0] += xx
    data[:, :, 3] += yy
    return data





@njit(
    [
        complex64[:, :, :](
            complex64[:, :, :],
            float64[:],
            float64[:, :, :],
            float64[:],
            float64[:],
            float64[:],
            float64[:],
            float64[:, :, :],
        )
    ],
    parallel=True,
)
"""


def offset_vis(
    data, lmbdas, uvw_lmbdas, A, ls, ms, ns, phasediffs,
):
    """Offset visibilities"""
    data[:] = 0
    source_count = 1
    for amp, l, m, n in zip(A, ls, ms, ns):
        print("Offset Source: ", source_count, "...")
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

        # print("Zenith angle: ", np.rad2deg(zen_angle), "deg  OR", zen_angle, "rad")
        # print("Azimuth angle: ", np.rad2deg(azimuth), "deg  OR", azimuth, "rad")

        # u_tec_list, v_tec_list, tec_per_ant = get_tec_value(
        #    phs_screen, us, vs, zen, az, scale, h_pix, pp_u_offset, pp_v_offset,
        # )
        phasediff = phasediffs[source_count - 1][:, np.newaxis] * lmbdas ** 2

        # print(phasediff.shape, "********phasediff shape********")

        phse = phse * np.exp(2j * np.pi * phasediff)  # [:, None]
        data[:, :, 0] += amp * phse  # feed xx data
        data[:, :, 3] += amp * phse  # feed yy data

        source_count += 1

    return data


def offset_vis2(
    data, lmbdas, uvw_lmbdas, A, ls, ms, ns, u_phasediffs, v_phasediffs,
):
    """Offset visibilities"""
    data[:] = 0
    source_count = 1
    for amp, l, m, n in zip(A, ls, ms, ns):
        print("Offset Source: ", source_count, "...")

        u_phasediff = 100 * u_phasediffs[source_count - 1][:, np.newaxis] * lmbdas ** 2
        v_phasediff = 100 * v_phasediffs[source_count - 1][:, np.newaxis] * lmbdas ** 2
        print(u_phasediff[10000, 0], "u_phasediff shape")
        print(v_phasediff[10000, 0], "v_phasediff shape")
        vvvv = uvw_lmbdas[:, :, 0] * l
        print(vvvv.shape, "uvw_la", vvvv[10000, 0])

        phse = np.exp(
            2j
            * np.pi
            * (
                (uvw_lmbdas[:, :, 0] * l + u_phasediff)
                + (uvw_lmbdas[:, :, 1] * m + v_phasediff)
                + uvw_lmbdas[:, :, 2] * n
            )
        )
        # for i in range(data.shape[0]):
        #   phse[i,:] *= np.exp(2*np.pi*phasediff[i])
        #   phse[i,:] = phse[i,:] * np.exp(phasediff[i])
        # phasediff = add_phase_offsets(mset)
        # phasediff = np.ones(uvw_lmbdas.shape[0])*1j*2

        # print("Zenith angle: ", np.rad2deg(zen_angle), "deg  OR", zen_angle, "rad")
        # print("Azimuth angle: ", np.rad2deg(azimuth), "deg  OR", azimuth, "rad")

        # u_tec_list, v_tec_list, tec_per_ant = get_tec_value(
        #    phs_screen, us, vs, zen, az, scale, h_pix, pp_u_offset, pp_v_offset,
        # )

        # print(phasediff.shape, "********phasediff shape********")

        # phse = phse * np.exp(2j * np.pi * phasediff)  # [:, None]
        # phse = phse * np.exp(
        #    2j
        #    * np.pi
        #   * (uvw_lmbdas[:, :, 0] * u_phasediff + uvw_lmbdas[:, :, 1] * v_phasediff)
        # )

        data[:, :, 0] += amp * phse  # feed xx data
        data[:, :, 3] += amp * phse  # feed yy data

        source_count += 1

    return data


'''
@njit(
    [
        complex64[:, :, :](
            complex64[:, :, :],
            float64[:],
            float64[:, :, :],
            float64[:],
            float64[:],
            float64[:],
            float64[:],
            float64[:, :],
        )
    ],
    parallel=True,
)



def offset_vis2(
    data, lmbdas, uvw_lmbdas, A, ls, ms, ns, phasediffs,
):
    """Offset visibilities"""
    data[:] = 0
    u = uvw_lmbdas[:, :, 0]
    v = uvw_lmbdas[:, :, 1]
    w = uvw_lmbdas[:, :, 2]
    xx = data[:, :, 0]
    yy = data[:, :, 3]
    source_count = 1
    for source in range(len(A)):
        print("Offset Source: ", source_count, "...")
        phse = np.exp(2j * np.pi * (u * ls[source] + v * ms[source] + w * ns[source]))
        # for i in range(data.shape[0]):
        #   phse[i,:] *= np.exp(2*np.pi*phasediff[i])
        #   phse[i,:] = phse[i,:] * np.exp(phasediff[i])
        # phasediff = add_phase_offsets(mset)
        # phasediff = np.ones(uvw_lmbdas.shape[0])*1j*2

        # print("Zenith angle: ", np.rad2deg(zen_angle), "deg  OR", zen_angle, "rad")
        # print("Azimuth angle: ", np.rad2deg(azimuth), "deg  OR", azimuth, "rad")

        # u_tec_list, v_tec_list, tec_per_ant = get_tec_value(
        #    phs_screen, us, vs, zen, az, scale, h_pix, pp_u_offset, pp_v_offset,
        # )
        phasediff = phasediffs[source][:, np.newaxis] * lmbdas ** 2

        # print(phasediff.shape, "********phasediff shape********")

        phse = phse * np.exp(2j * np.pi * phasediff)  # [:, None]
        xx += A[source] * phse  # feed xx data
        yy += A[source] * phse  # feed yy data

        source_count += 1
    data[:, :, 0] += xx
    data[:, :, 3] += yy

    return data



@njit(
    [
        complex64[:, :, :](
            complex64[:, :, :],
            float64[:],
            float64[:, :, :],
            float64[:],
            float64[:],
            float64[:],
            float64[:],
            float64[:],
        )
    ],
    parallel=True,
)
def offset_vis3(
    data, lmbdas, uvw_lmbdas, A, ls, ms, ns, phasediffs,
):
    """Offset visibilities"""
    data[:] = 0
    u = uvw_lmbdas[:, :, 0]
    v = uvw_lmbdas[:, :, 1]
    w = uvw_lmbdas[:, :, 2]
    source_count = 1
    for source in prange(len(A)):
        print("Offset Source: ", source_count, "...")
        phse = np.exp(2j * np.pi * (u * ls[source] + v * ms[source] + w * ns[source]))
        # for i in range(data.shape[0]):
        #   phse[i,:] *= np.exp(2*np.pi*phasediff[i])
        #   phse[i,:] = phse[i,:] * np.exp(phasediff[i])
        # phasediff = add_phase_offsets(mset)
        # phasediff = np.ones(uvw_lmbdas.shape[0])*1j*2

        # print("Zenith angle: ", np.rad2deg(zen_angle), "deg  OR", zen_angle, "rad")
        # print("Azimuth angle: ", np.rad2deg(azimuth), "deg  OR", azimuth, "rad")

        # u_tec_list, v_tec_list, tec_per_ant = get_tec_value(
        #    phs_screen, us, vs, zen, az, scale, h_pix, pp_u_offset, pp_v_offset,
        # )
        # phasediff = phasediffs[source][:, None] * lmbdas ** 2
        p = phasediffs[source]
        phasediff = p.reshape(*p, 1) * lmbdas ** 2

        print(phasediff.shape, "********phasediff shape********")

        phse = phse * np.exp(2j * np.pi * phasediff)  # [:, None]
        data[:, :, 0] += A[source] * phse  # feed xx data
        data[:, :, 3] += A[source] * phse  # feed yy data

        source_count += 1

    return data

'''


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
