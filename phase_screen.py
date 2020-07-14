#! /usr/bin/env python
__author__ = [
    "Kariuki Chege",
]

import numpy as np
from powerbox import PowerBox
import scipy.ndimage.filters as sp


import mset_utils as mtls


c = 299792458
constant = 40.30819
tecu = 1e16
kolmogorov_index = -1.66667


def linear_tec(npix, sine=False):
    """Generate linear TEC plane increasing uniformly in one direction.

    Parameters
    ----------
    npix : int
        Number of pixels per side

    Returns
    -------
    [numpy array]
        [2D npix by npix tec array]
    """
    if sine:
        xs = np.linspace(0, 50, npix)
        ys = np.linspace(0, 50, npix)
        tau, phi = np.meshgrid(xs, ys)
        tec = np.sin(5 * (tau + phi))  # + np.random.normal(scale=1 / 60, size=npix)
        tec = abs(tec)
    else:
        tec = np.tile(np.linspace(0, 100, npix), (npix, 1))
    return tec


def iono_phase_shift(frequency, scale=10, size=150000, tec_type="l"):
    """Should convert a tec screen to a phase offset screen using the commented formula.
    Not yet doing that.

    Parameters
    ----------
    frequency : float.\n
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
        pb = PowerBox(
            resolution, lambda k: 10 * k ** kolmogorov_index, ensure_physical=True
        )
        tec = pb.delta_x()
        # Apply gaussian filter
        sigma = [40, 40]
        tec = sp.gaussian_filter(tec, sigma, mode="constant")

    elif tec_type == "s":
        tec = linear_tec(resolution, sine=True)

    else:
        tec = linear_tec(resolution, sine=False)

    # ammend formula here.
    # phase_screen = constant/tecu * 1/(frequency**1e6) * tec -->
    # shouts "OverflowError: (34, 'Numerical result out of range')"
    phase_screen = tec

    return phase_screen


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
    ra, dec0 = mtls.get_phase_center(tbl)
    ra0 = lst - ra  # RA0 here is the hour angle since HA = LST-RA
    print("hour_angle: ", ra0, type(ra0))
    xyz = mtls.get_bl_vectors(mset)

    us = np.sin(ra0) * xyz[:, 0] + np.cos(ra0) * xyz[:, 1]
    vs = (
        -np.sin(dec0) * np.cos(ra0) * xyz[:, 0]
        + np.sin(dec0) * np.sin(ra0) * xyz[:, 1]
        + np.cos(dec0) * xyz[:, 2]
    )
    ws = (
        np.cos(dec0) * np.cos(ra0) * xyz[:, 0]
        + -np.cos(dec0) * np.sin(ra0) * xyz[:, 1]
        + np.sin(dec0) * xyz[:, 2]
    )

    return us, vs, ws


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
    min_u = min(us)
    max_u = max(us)
    scaled = [
        ((u - min_u) / (max_u - min_u)) * (pixel_max - pixel_min) + pixel_min
        for u in us
    ]
    return np.array(scaled)


def get_tec_value(tec, us, vs, zen_angle, az, scale=10, h=200000, refant=0):
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
    print(
        "tecscreen is size %s by %s km and at height %s km."
        % (tec.shape[0] * scale / 1000, tec.shape[1] * scale / 1000, h / 1000)
    )
    # Apply scaling to the array field and tec height.
    us_scaled = scale_to_pixel_range(us, scale=scale)
    vs_scaled = scale_to_pixel_range(vs, scale=scale)
    h_pix = h / scale
    print("refant scaled u and v position", us_scaled[refant], vs_scaled[refant])

    u_tec_center = tec.shape[0] // 2  # + us_scaled[refant]
    v_tec_center = tec.shape[1] // 2  # + us_scaled[refant]

    u_tec_list, v_tec_list, tec_per_ant = [], [], []
    for u, v in zip(us_scaled, vs_scaled):
        # For each antenna, the antenna position becomes a new origin
        # This antenna position first has to be in reference to the refant.
        new_u0 = u - us_scaled[refant]
        new_v0 = v - vs_scaled[refant]
        # For each antenna, the zenith angle projects a circle onto the tec screen.
        # the radius of the circle is given by:
        zen_angle_radius = h_pix * np.tan(zen_angle)  # zen_angle should be in radians
        # The azimuth angle gives us the arc on this circle from some starting point
        # We can then obtain the u and v coordinates for the pierce point.
        # The pi rad added to azimuth angle  is to rotate the start point of the arc
        #   to the topside of the tec field.
        pp_u = zen_angle_radius * np.sin(az + np.pi / 2.0) + new_u0
        pp_v = zen_angle_radius * np.cos(az + np.pi / 2.0) + new_v0
        # Collect pierce points for each antenna.
        uu = pp_u + u_tec_center  # + us_scaled[refant]
        vv = pp_v + v_tec_center  # + vs_scaled[refant]

        u_tec_list.append(uu)
        v_tec_list.append(vv)

        tec_per_ant.append(tec[int(round(uu)), int(round(vv))])
    print(tec_per_ant[0], "First/middle value")
    return np.array(u_tec_list), np.array(v_tec_list), np.array(tec_per_ant)
