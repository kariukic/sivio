import numpy as np
from numba import complex64, float64, njit, prange
from scipy import constants

import mset_utils as mtls
from coordinates import radec_to_altaz, MWAPOS


def sim_prep(tbl, ras, decs):
    """Preparing the requiremnts before simulating"""
    # grab uvw values from the ms and divide them with the wavelengths for each channel
    uvw = mtls.get_uvw(tbl)
    lmbdas = mtls.get_channels(tbl, ls=True)
    uvw_lmbdas = uvw[:, None, :] / lmbdas[None, :, None]
    # Frequency resolution
    frequencies = mtls.get_channels(tbl, ls=False)
    dnu = frequencies[1] - frequencies[0]

    # grab the data from the ms table
    data = mtls.get_data(tbl)
    # Delete all the data because we will put the simulated data there
    data[:] = 0

    # Get the sources lmns
    ls, ms, ns = mtls.get_lmns(tbl, ras, decs, phase_center_shift=0)
    return data, lmbdas, uvw_lmbdas, dnu, ls, ms, ns


# @njit(
#     [
#         complex64[:, :, :](
#             complex64[:, :, :],
#             float64[:, :, :],
#             float64[:, :, :],
#             float64[:],
#             float64[:],
#             float64[:],
#         )
#     ],
#     parallel=True,
# )
@njit(parallel=True)
def true_vis_numba(data, uvw_lmbdas, fluxes, ls, ms, ns):
    data[:] = 0
    u = uvw_lmbdas[:, :, 0]
    v = uvw_lmbdas[:, :, 1]
    w = uvw_lmbdas[:, :, 2]
    source_count = 1
    for source in prange(len(fluxes)):
        print("True Source: ", source_count, "...")
        xx_source_visibilities = fluxes[source, :, 0] * np.exp(
            2j * np.pi * (u * ls[source] + v * ms[source] + w * ns[source])
        )
        yy_source_visibilities = fluxes[source, :, 3] * np.exp(
            2j * np.pi * (u * ls[source] + v * ms[source] + w * ns[source])
        )
        data[:, :, 0] += xx_source_visibilities  # feed xx data
        data[:, :, 3] += yy_source_visibilities  # feed yy data
        source_count += 1

    return data


def get_antenna_in_uvw(mset, tbl, lst):
    """
    Convert the antennas positions into some sort of uv plane with one antenna position as the center of the field.
    [Old idea, now I just use the xyz values and they seem mixed up on MWA MS]

    Parameters
    ----------
    mset :
        Measurement set.\n
    tbl : casacore table.
        The casacore mset table opened with readonly=False.\n
    lst : Format: 00h00m00.0000s
        Local sideral time.\n

    Returns
    -------
    3 lists/arrays.
        u, v and w positions of the projected antennas in metres.
    """
    """
    from astropy.coordinates import Angle
    from astropy import units as unit
    ra0, dec0 = mtls.get_phase_center(tbl)
    # ra0 = np.deg2rad(0)
    print("lst", lst)
    ra0 = Angle(ra0, unit.radian)
    print("ra0", ra0)
    ha = Angle(lst) - ra0  # HA = LST-RA
    print("Hour angle", ra0)

    xyz = mtls.get_bl_vectors(mset)

    us = np.sin(ha) * xyz[:, 0] + np.cos(ha) * xyz[:, 1]

    vs = (
        -np.sin(dec0) * np.cos(ha) * xyz[:, 0]
        + np.sin(dec0) * np.sin(ha) * xyz[:, 1]
        + np.cos(dec0) * xyz[:, 2]
    )
    ws = (
        np.cos(dec0) * np.cos(ha) * xyz[:, 0]
        + -np.cos(dec0) * np.sin(ha) * xyz[:, 1]
        + np.sin(dec0) * xyz[:, 2]
    )

    return us, vs, ws
    """
    xyz = mtls.get_bl_vectors(mset)
    # return xyz[:, 0], xyz[:, 2], xyz[:, 1]
    return xyz[:, 0], xyz[:, 1], xyz[:, 2]


def scale_to_pixel_range(us, scale=10):
    """
    Scale antenna positions into the axis range of the tec field.

    Parameters
    ----------
    us : list/array.
        list of distances to be scaled. \n
    scale : int, optional.
        the pixel to distance scaling in metres, by default 10 \n

    Returns
    -------
    array.
        scaled distances.
    """
    pixel_max = max(us) / scale
    pixel_min = 0

    scaled = ((us - np.min(us)) / (np.max(us) - np.min(us))) * (
        pixel_max - pixel_min
    ) + pixel_min

    return np.array(scaled)


def phase_center_offset(ra0, dec0, h_pix, time):
    """
    Calculate the pixel coordinates for the phase center position.
    This is the offset used to aligh the phase center to the center of the phase screen

    Parameters
    ----------
    tbl : casacore MS table.
        The MS table to get the phase center from.\n
    h_pix : int/float
        The scaled height of the phase screen.\n
    time : astropy time object.
        The time of observation.

    Returns
    -------
    float.
        The pixel coordinates of the phase center on the phase screen.
    """
    new_u0 = 0
    new_v0 = 0
    altt, azz = radec_to_altaz(ra0, dec0, time, MWAPOS)
    zen_angle = np.pi / 2.0 - altt
    zen_angle_radius = h_pix * np.tan(zen_angle)

    pp_u_off = zen_angle_radius * np.sin(azz) + new_u0
    pp_v_off = zen_angle_radius * np.cos(azz) + new_v0
    return pp_u_off, pp_v_off


def compute_initial_setup_params(tec, us, vs, height, scale, ra0, dec0, time):
    # Lets first get the gradient of all pixels in the tecscreen
    du, dv = np.gradient(tec)
    # Apply scaling to the array field and height
    scaled_u_axis_ants_pos = scale_to_pixel_range(us, scale=scale)
    scaled_v_axis_ants_pos = scale_to_pixel_range(vs, scale=scale)

    u_tec_center = tec.shape[0] // 2  # + us_scaled[refant]
    v_tec_center = tec.shape[1] // 2

    h_pix = height / scale
    ra0_pp_u_coord, dec0_pp_v_coord = phase_center_offset(ra0, dec0, h_pix, time)

    u_coord_shift_factor = tec.shape[0] + ra0_pp_u_coord - u_tec_center
    v_coord_shift_factor = v_tec_center - dec0_pp_v_coord

    # And now at this point we have the set up parameters:
    initial_setup_params = (
        du,
        dv,
        scaled_u_axis_ants_pos,
        scaled_v_axis_ants_pos,
        u_coord_shift_factor,
        v_coord_shift_factor,
        h_pix,
    )

    return initial_setup_params


@njit
def single_source_pierce_point(
    du,
    dv,
    source_zenith_angle,
    source_azimuth_angle,
    scaled_u_axis_ants_pos,
    scaled_v_axis_ants_pos,
    h_pix,
    u_shift_factor,
    v_shift_factor,
    refant=0,
    get_gradients=True,
):
    ants_u_pierce_points, ants_v_pierce_points, ants_u_gradients, ants_v_gradients = (
        [np.float64(x) for x in range(0)],
        [np.float64(x) for x in range(0)],
        [np.float64(x) for x in range(0)],
        [np.float64(x) for x in range(0)],
    )
    for scaled_u_ant_pos, scaled_v_ant_pos in zip(
        scaled_u_axis_ants_pos, scaled_v_axis_ants_pos
    ):
        # For each antenna, the antenna position becomes a new origin
        # This antenna position first has to be in reference to the refant.
        new_u0 = scaled_u_ant_pos - scaled_u_axis_ants_pos[refant]
        new_v0 = scaled_v_ant_pos - scaled_v_axis_ants_pos[refant]

        # For each antenna, the zenith angle projects a circle onto the tec screen.
        # the radius of the circle is given by:
        # zen_angle should be in radians
        zen_angle_radius = h_pix * np.tan(source_zenith_angle)
        # The azimuth angle gives us the arc on this circle from some starting point
        # We can then obtain the u and v coordinates for the pierce point.
        pp_u_coord = zen_angle_radius * np.sin(source_azimuth_angle) + new_u0
        pp_v_coord = zen_angle_radius * np.cos(source_azimuth_angle) + new_v0
        # Shift the u and v piercepoint values in reference to the phase screen center
        u_axis_pierce_point = u_shift_factor - pp_u_coord
        v_axis_pierce_point = v_shift_factor + pp_v_coord
        # Collect pierce points for each antenna for the source.
        ants_u_pierce_points.append(u_axis_pierce_point)
        ants_v_pierce_points.append(v_axis_pierce_point)

        if get_gradients:
            # get the u and v direction gradients at the antenna pierce point.
            u_gradient = du[
                int(round(u_axis_pierce_point)), int(round(v_axis_pierce_point))
            ]
            v_gradient = dv[
                int(round(u_axis_pierce_point)), int(round(v_axis_pierce_point))
            ]

            ants_u_gradients.append(u_gradient)
            ants_v_gradients.append(v_gradient)
    if get_gradients:
        return (
            np.array(ants_u_gradients),
            np.array(ants_v_gradients),
        )
    else:
        return (
            np.array(ants_u_pierce_points),
            np.array(ants_v_pierce_points),
        )


def screen_size(height, radius, ra0, dec0, us, vs, MWAPOS, time):
    max_ra = ra0 + np.deg2rad(radius)
    max_dec = dec0 + np.deg2rad(radius)

    max_uv = max(max(us), max(vs))
    new_u0 = max_uv - us[0]
    new_v0 = max_uv - vs[0]

    alt, max_az = radec_to_altaz(max_ra, max_dec, time, MWAPOS)
    max_zen = np.pi / 2.0 - alt

    zen_angle_radius = height * np.tan(max_zen)

    pp_u_coord = zen_angle_radius * np.sin(max_az) + new_u0
    pp_v_coord = zen_angle_radius * np.cos(max_az) + new_v0

    # the 1000 is a 1km padding just to make sure no pierce point falls outside the phase screen
    return 2 * max(pp_u_coord, pp_v_coord) + 1000


def collective_pierce_points(zenith_angles, azimuths, initial_setup_params):
    (
        du,
        dv,
        scaled_u_axis_ants_pos,
        scaled_v_axis_ants_pos,
        u_coord_shift_factor,
        v_coord_shift_factor,
        h_pix,
        _,
        _,
        _,
    ) = initial_setup_params
    u_per_source, v_per_source = [], []
    for zen_a, az in zip(zenith_angles, azimuths):
        ants_u_pierce_points, ants_v_pierce_points = single_source_pierce_point(
            du,
            dv,
            zen_a,
            az,
            scaled_u_axis_ants_pos,
            scaled_v_axis_ants_pos,
            h_pix,
            u_coord_shift_factor,
            v_coord_shift_factor,
            get_gradients=False,
        )

        u_per_source.append(ants_u_pierce_points)
        v_per_source.append(ants_v_pierce_points)
    return np.stack((u_per_source, v_per_source))


@njit
def add_phase_offsets(lmbdas, antenna1, antenna2, uparams, vparams, spar):
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

    u_phasediffs = uparams[antenna1] - uparams[antenna2]
    v_phasediffs = vparams[antenna1] - vparams[antenna2]

    u_phasediffs = float(spar) * np.expand_dims(u_phasediffs, axis=1) * lmbdas ** 2
    v_phasediffs = float(spar) * np.expand_dims(v_phasediffs, axis=1) * lmbdas ** 2

    return u_phasediffs, v_phasediffs


@njit
def get_uv_phase_offsets(
    lmbdas, source_zenith_angle, source_azimuth_angle, initial_setup_params
):
    (
        du,
        dv,
        scaled_u_axis_ants_pos,
        scaled_v_axis_ants_pos,
        u_coord_shift_factor,
        v_coord_shift_factor,
        h_pix,
        ant1,
        ant2,
        spar,
    ) = initial_setup_params

    ants_u_gradients, ants_v_gradients = single_source_pierce_point(
        du,
        dv,
        source_zenith_angle,
        source_azimuth_angle,
        scaled_u_axis_ants_pos,
        scaled_v_axis_ants_pos,
        h_pix,
        u_coord_shift_factor,
        v_coord_shift_factor,
        refant=0,
        get_gradients=True,
    )

    # calculate the phase diffs for the source
    u_phasediffs, v_phasediffs = add_phase_offsets(
        lmbdas, ant1, ant2, ants_u_gradients, ants_v_gradients, spar
    )

    return u_phasediffs, v_phasediffs


@njit
def single_source_offset_vis(args):
    """Compute single source offset visibilities

    Parameters
    ----------
    args : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    (lmbdas, uvw_lmbdas, zen_a, az, flux, ll, mm, nn, initial_setup_params) = args

    u_phasediffs, v_phasediffs = get_uv_phase_offsets(
        lmbdas, zen_a, az, initial_setup_params
    )

    offset_phase = np.exp(
        2j
        * np.pi
        * (
            (uvw_lmbdas[:, :, 0] * ll + u_phasediffs)
            + (uvw_lmbdas[:, :, 1] * mm + v_phasediffs)
            + uvw_lmbdas[:, :, 2] * nn
        )
    )
    return flux[:, 0] * offset_phase, flux[:, 1] * offset_phase


@njit(parallel=True)
def compute_offset_vis_parallel(
    data,
    initial_setup_params,
    zenith_angles,
    azimuths,
    lmbdas,
    uvw_lmbdas,
    fluxes,
    ls,
    ms,
    ns,
):
    data[:] = 0
    source_num = 0
    for source in prange(len(fluxes)):
        print("source: ", source_num)
        params = (
            lmbdas,
            uvw_lmbdas,
            zenith_angles[source],
            azimuths[source],
            fluxes[source],
            ls[source],
            ms[source],
            ns[source],
            initial_setup_params,
        )
        xx_offset_data, yy_offset_data = single_source_offset_vis(params)
        source_num += 1
        data[:, :, 0] += xx_offset_data  # feed xx data
        data[:, :, 3] += yy_offset_data  # feed yy data

    return data


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

    # print("Adding thermal noise...")
    rl_im = np.random.normal(0, 1, (2,) + visibilities.shape)

    # NOTE: we divide the variance by two here, because the variance of the absolute value of the
    #       visibility should be equal to thermal_variance_baseline, which is true if the variance of both
    #       the real and imaginary components are divided by two.
    return visibilities + np.sqrt(thermal_variance_baseline(dnu) / 2) * (
        rl_im[0, :] + rl_im[1, :] * 1j
    )
