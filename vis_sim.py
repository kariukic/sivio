# Simulate visibilities and feed them into an MWA MS
import numpy as np
from scipy import constants

from numba import njit, float64, complex64, prange

from scint_equations import add_scint
import mset_utils as mtls


def sim_prep(tbl, ras, decs):
    """Preparing the requiremnts before simulating"""
    # grab uvw values from the ms and divide them with the wavelengths for each channel
    uvw = mtls.get_uvw(tbl)
    lmbdas = mtls.get_channels(tbl, ls=True)
    frequencies = mtls.get_channels(tbl, ls=False)
    dnu = frequencies[1] - frequencies[0]
    uvw_lmbdas = uvw[:, None, :] / lmbdas[None, :, None]

    # Now lets grab the data from the ms table
    data = mtls.get_data(tbl)
    print("Shape of the MS data", data.shape)

    # Delete all the grabbed data because we will put the simulated data there
    data[:] = 0

    # grab our lmns
    ls, ms, ns = mtls.get_lmns(tbl, ras, decs, phase_center_shift=0)
    # assert len(list(A)) == len(list(ls))
    print("dnu", dnu)
    return data, lmbdas, uvw_lmbdas, dnu, ls, ms, ns


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
def true_vis_numba(data, uvw_lmbdas, fluxes, ls, ms, ns):
    data[:] = 0
    u = uvw_lmbdas[:, :, 0]
    v = uvw_lmbdas[:, :, 1]
    w = uvw_lmbdas[:, :, 2]
    source_count = 1
    for source in prange(len(fluxes)):
        print("True Source: ", source_count, "...")
        source_visibilities = fluxes[source] * np.exp(
            2j * np.pi * (u * ls[source] + v * ms[source] + w * ns[source])
        )
        data[:, :, 0] += source_visibilities  # feed xx data
        data[:, :, 3] += source_visibilities  # feed yy data
        source_count += 1

    return data


def offset_vis_old(
    data, lmbdas, uvw_lmbdas, fluxes, ls, ms, ns, phasediffs,
):
    """Offset visibilities"""
    data[:] = 0
    source_count = 1
    for amp, l, m, n in zip(fluxes, ls, ms, ns):
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


def offset_vis_slow(
    data, lmbdas, uvw_lmbdas, A, ls, ms, ns, u_phasediffs, v_phasediffs, spar
):
    """Offset visibilities"""
    data[:] = 0
    source_count = 1
    for amp, l, m, n in zip(A, ls, ms, ns):
        print("Offset Source: ", source_count, "...")

        # u_phasediff = 100 * u_phasediffs[source_count - 1][:, np.newaxis] * lmbdas ** 2
        # v_phasediff = 100 * v_phasediffs[source_count - 1][:, np.newaxis] * lmbdas ** 2
        u_phasediff = float(spar) * u_phasediffs[source_count -
                                        1][:, np.newaxis] * lmbdas ** 2
        v_phasediff = float(spar) * v_phasediffs[source_count -
                                        1][:, np.newaxis] * lmbdas ** 2

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


@njit(parallel=True,)
def offset_vis_numba(
    data, lmbdas, uvw_lmbdas, A, ls, ms, ns, u_phasediffs, v_phasediffs,
):
    """Offset visibilities"""
    data[:] = 0
    u = uvw_lmbdas[:, :, 0]
    v = uvw_lmbdas[:, :, 1]
    w = uvw_lmbdas[:, :, 2]
    source_count = 1
    lmbdas = np.expand_dims(lmbdas, axis=-1)
    print(lmbdas.shape)
    for source in prange(len(A)):
        print("Offset Source: ", source_count, "...")
        print(np.expand_dims(u_phasediffs[source], axis=-1).shape)
        u_phasediff = 100 * \
            np.expand_dims(u_phasediffs[source], axis=-1) * lmbdas ** 2
        v_phasediff = 100 * \
            np.expand_dims(v_phasediffs[source], axis=-1) * lmbdas ** 2

        # u_phasediff = 100 * u_phasediffs[source][:, np.newaxis] * lmbdas ** 2
        # v_phasediff = 100 * v_phasediffs[source][:, np.newaxis] * lmbdas ** 2

        phse = np.exp(
            -2j
            * np.pi
            * (
                (u * ls[source] + u_phasediff)
                + (v * ms[source] + v_phasediff)
                + w * ns[source]
            )
        )
        xx = A[source] * phse  # feed xx data
        yy = A[source] * phse  # feed yy data

        data[:, :, 0] += xx
        data[:, :, 3] += yy
        source_count += 1

    return data


"""
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
    '''Offset visibilities'''
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
"""


# Adopted from Bella's code
def thermal_variance_baseline(
    dnu=40000.0000000298, Tsys=240, timestamps=1, effective_collecting_area=21
):
    """
        The thermal variance of each baseline (assumed constant across baselines/times/frequencies.
        Equation comes from Trott 2016 (from Morales 2005)
        """
    # dnu = frequencies[1] - frequencies[0]
    integration_time = timestamps * 8

    sigma = (
        2
        * 1e26
        * constants.k
        * Tsys
        / effective_collecting_area
        / np.sqrt(dnu * integration_time)
    )
    return sigma ** 2


def add_thermal_noise(visibilities, dnu):
    """
    Add thermal noise to each visibility.
    Parameters
    ----------
    visibilities : (n_baseline, n_freq)-array
        The visibilities at each baseline and frequency.
    frequencies : (n_freq)-array
        The frequencies of the observation.
    beam_area : float
        The area of the beam (in sr).
    delta_t : float, optional
        The integration time.
    Returns
    -------
    visibilities : array
        The visibilities at each baseline and frequency with the thermal noise from the sky.
    """

    print("Adding thermal noise...")
    rl_im = np.random.normal(0, 1, (2,) + visibilities.shape)

    # NOTE: we divide the variance by two here, because the variance of the absolute value of the
    #       visibility should be equal to thermal_variance_baseline, which is true if the variance of both
    #       the real and imaginary components are divided by two.
    return visibilities + np.sqrt(thermal_variance_baseline(dnu) / 2) * (
        rl_im[0, :] + rl_im[1, :] * 1j
    )


def sigma_b(frequencies, tile_diameter=4.0):
    "The Gaussian beam width at each frequency"
    epsilon = 0.42  # scaling from airy disk to Gaussian
    return (epsilon * constants.c) / (frequencies / tile_diameter)


def beam_area(frequencies):
    """
    The integrated beam area. Assumes a frequency-dependent Gaussian beam (in lm-space, as this class implements).
    Parameters
    ----------
    frequencies : array-like
        Frequencies at which to compute the area.
    Returns
    -------
    beam_area: (nfreq)-array
        The beam area of the sky, in lm.
    """
    sig = sigma_b(frequencies)
    return np.pi * sig ** 2


def beam(frequencies, sky_coords, n_cells=1028, min_attenuation=1e-7):
    """
    Generate a frequency-dependent Gaussian beam attenuation across the sky per frequency.
    Parameters
    ----------
    ncells : int
        Number of cells in the sky grid.
    sky_size : float
        The extent of the sky in lm.
    Returns
    -------
    attenuation : (ncells, ncells, nfrequencies)-array
        The beam attenuation (maximum unity) over the sky.

    beam_area : (nfrequencies)-array
        The beam area of the sky (in sr).
    """

    # Create a meshgrid for the beam attenuation on sky array
    L, M = np.meshgrid(np.sin(sky_coords), np.sin(sky_coords), indexing="ij")

    attenuation = np.exp(
        np.outer(-(L ** 2 + M ** 2), 1.0 / (2 * sigma_b(frequencies) ** 2)).reshape(
            (n_cells, n_cells, len(frequencies))
        )
    )

    attenuation[attenuation < min_attenuation] = 0

    return attenuation


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
