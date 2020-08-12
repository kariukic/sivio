#! /usr/bin/env python
__author__ = [
    "Kariuki Chege",
]

import numpy as np
from powerbox import PowerBox
import scipy.ndimage.filters as sp

# from numba import njit


import mset_utils as mtls
from coordinates import radec_to_altaz, MWAPOS


c = 299792458
# 1/(8*np.pi**2)*k.elementary_charge**2/(k.epsilon_0*k.electron_mass)
# k=scipy.constants
constant = 40.30819300005713
tecu = 1e16
kolmogorov_index = -11 / 3  # -1.66667


def convert_to_tecu(frequency, phscreen):
    frequency *= 1e6
    stec = 1 / constant * phscreen * frequency ** 2
    stec /= 1e16
    return stec  # slant TEC


def scale_to_pi_range(phscreen):
    """Scaling our produced phase screen to between [-180,180] degrees

    Parameters
    ----------
    phscreen : array.
        The phase screen to be scaled.

    Returns
    -------
    array.
        Scaled array.
    """
    return (
        (phscreen - np.min(phscreen)) / (np.max(phscreen) - np.min(phscreen))
    ) * 2 * np.pi - np.pi


def linear_tec(npix, sine=False):
    """Generate linear TEC plane increasing uniformly in one direction.

    Parameters
    ----------
    npix : int
        Number of pixels per side

    Returns
    -------
    array.
        2D npix by npix tec array.
    """
    if sine:
        xs = np.linspace(0, np.pi, npix)
        ys = np.linspace(0, np.pi, npix)
        tau, phi = np.meshgrid(xs, ys)
        tec = np.sin(3 * (0.9 * tau + 0.8 * phi))
        np.fliplr(tec)[np.triu_indices(npix, npix / 3.5)] = np.pi * 1e-6
        np.rot90(tec)[np.triu_indices(npix, npix / 3.5)] = np.pi * 1e-6
        tec += np.random.normal(loc=1e-6, scale=0.1, size=(npix, npix))  # noise
        # for once lets change ridge orientation to along minor diagonal.
        # tec = np.fliplr(tec)
    else:
        tec = np.tile(np.linspace(0, np.pi, npix), (npix, 1))
    return tec


def iono_phase_shift(scale=3, size=60000, tec_type="l"):
    """
    produces a phase offset screen.

    Parameters
    ----------
    scale : int, optional.
        Scaling factor. the number of metres represented by the side of a single pixel], by default 10. \n
    size : int, optional.
        Total length of each side, by default 150000.\n
    tec_type : str, optional.
        If k, the tec used follows kolmogorov turbulence, by default l for linear.

    Returns
    -------
    2darray.
        Phase offset values.
    """

    # \Delta STEC [1/m^3] = \Delta\theta [rad] * \nu^2 [(MHz)^2] / k
    # k = 1/(8\pi^2) e^2/(\epsilon_0 m_e) [m^3 / s^2]
    # 1 TECU = 10^16 [1/m^2]

    # Hence: $\Delta\theta = 40.30819/1e16 * 1/\nu^2 * STEC

    resolution = size // scale  # the number of pixel per side of tec plane

    if tec_type == "k":
        apply_filter = True
        pb = PowerBox(
            resolution, lambda k: 10 * k ** kolmogorov_index, ensure_physical=True
        )
        phs_screen = pb.delta_x()

    elif tec_type == "s":
        apply_filter = True
        phs_screen = linear_tec(resolution, sine=True)

    else:
        apply_filter = True
        phs_screen = linear_tec(resolution, sine=False)

    phs_screen = scale_to_pi_range(phs_screen)
    if apply_filter:
        sigma = [80, 80]
        phs_screen = sp.gaussian_filter(phs_screen, sigma, mode="constant")

    return phs_screen


def get_antenna_in_uvw(mset, tbl, lst):
    """
    Convert the antennas positions into some sort of uv plane with one antenna position as the center of the field.

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
    return xyz[:, 0], xyz[:, 1], xyz[:, 2]


def scale_to_pixel_range(us, scale=10):
    """
    Scale antenna positions into the axis range of the tec field.

    Parameters
    ----------
    us : list/array.
        list of distances to be scaled. \n
    scale : int, optional.
        the pixel to distance scaling inmetres, by default 10 \n

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


def get_tec_value(
    tec, us, vs, zen_angles, azimuths, scale, h_pix, pp_u_offset, pp_v_offset, refant=0
):
    """
    Given a source position (zenith and azimuth angles) with reference to the reference antenna, this function obtains
    the tec phase offset value at the pierce point corresponding to each antenna. \n \n

    Function inspired by: https://stackoverflow.com/a/22778207/7905494 \n
    Plotting a point(s) of a circle laying on a plane given the origin, radius and angle(s).

    Parameters
    ----------
    tec : 2Darray.
        The TEC/phase screen. \n
    us : arraylike.
        x coordinates for each antenna. Named 'us' bcause the array is projected into the uv space.\n
    vs : arraylike.
        y coordinates for each antenna. \n
    zen_angle : float.
        The zenith angle of the source as calculated from the array earth location and observation time. \n
    az : float.
        The azimuth angle.
    scale : int, optional
        Scaling factor. the number of metres represented by the side of a single pixel], by default 10. \n
    h : int, optional
        Height of the tec screen in metres, by default 200000. \n
    refant : int, optional
        Reference antenna ID, by default 0

    Returns
    -------
    1: array.
        x coordinates for the pierce points of each antenna.\n
    2: array.
        y coordinates for the pierce points of each antenna.\n
    3: array.
        The TEC/phase offset value at the piercepoint of each antenna.
    """
    # print(
    #    "tecscreen size:  %s by %s km. Height:  %s km."
    #    % (
    #        tec.shape[0] * scale / 1000,
    #        tec.shape[1] * scale / 1000,
    #        h_pix * scale / 1000,
    #    )
    # )
    # Apply scaling to the array field and tec height.
    us_scaled = scale_to_pixel_range(us, scale=scale)
    vs_scaled = scale_to_pixel_range(vs, scale=scale)

    u_tec_center = tec.shape[0] // 2  # + us_scaled[refant]
    v_tec_center = tec.shape[1] // 2  # + us_scaled[refant]

    u_per_source, v_per_source, tec_per_source = [], [], []
    for zen_angle, az in zip(zen_angles, azimuths):
        u_tec_list, v_tec_list, tec_per_ant = [], [], []
        for u, v in zip(us_scaled, vs_scaled):
            # For each antenna, the antenna position becomes a new origin
            # This antenna position first has to be in reference to the refant.
            new_u0 = u - us_scaled[refant]
            new_v0 = v - vs_scaled[refant]
            # For each antenna, the zenith angle projects a circle onto the tec screen.
            # the radius of the circle is given by:
            # zen_angle should be in radians
            zen_angle_radius = h_pix * np.tan(zen_angle)
            # The azimuth angle gives us the arc on this circle from some starting point
            # We can then obtain the u and v coordinates for the pierce point.
            pp_u = zen_angle_radius * np.sin(az) + new_u0
            pp_v = zen_angle_radius * np.cos(az) + new_v0

            # Collect pierce points for each antenna.
            uu = tec.shape[0] - (pp_u - pp_u_offset + u_tec_center)
            vv = pp_v - pp_v_offset + v_tec_center
            u_tec_list.append(uu)
            v_tec_list.append(vv)
            # phase offset value per pierce point.
            tec_per_ant.append(tec[int(round(uu)), int(round(vv))])
        u_per_source.append(u_tec_list)
        v_per_source.append(v_tec_list)
        tec_per_source.append(tec_per_ant)

    return np.array(u_per_source), np.array(v_per_source), np.array(tec_per_source)
