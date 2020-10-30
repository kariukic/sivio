#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.fftpack

# from powerbox import PowerBox
import scipy.ndimage.filters as sp

# from scipy.constants import c

# 1/(8*np.pi**2)*k.elementary_charge**2/(k.epsilon_0*k.electron_mass)
# k=scipy.constants
constant = 40.30819300005713
tecu = 1e16
kolmogorov_index = -11 / 3  # -1.66667

# Fron Bruno Sciolla github


def fftind(size):
    """Returns a numpy array of shifted Fourier coordinates k_x k_y.

    Input args:
        size (integer): The size of the coordinate array to create
    Returns:
        k_ind, numpy array of shape (2, size, size) with:
            k_ind[0,:,:]:  k_x components
            k_ind[1,:,:]:  k_y components

    Example:

        print(fftind(5))

        [[[ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]]

        [[ 0  0  0  0  0]
        [ 1  1  1  1  1]
        [-3 -3 -3 -3 -3]
        [-2 -2 -2 -2 -2]
        [-1 -1 -1 -1 -1]]]

    """
    k_ind = np.mgrid[:size, :size] - int((size + 1) / 2)
    k_ind = scipy.fftpack.fftshift(k_ind)
    return k_ind


def gaussian_random_field(alpha=11, size=128, flag_normalize=False):
    """Returns a numpy array of shifted Fourier coordinates k_x k_y.

    Input args:
        alpha (double, default = 3.0):
            The power of the power-law momentum distribution
        size (integer, default = 128):
            The size of the square output Gaussian Random Fields
        flag_normalize (boolean, default = True):
            Normalizes the Gaussian Field:
                - to have an average of 0.0
                - to have a standard deviation of 1.0

    Returns:
        gfield (numpy array of shape (size, size)):

    Example:
    import matplotlib
    import matplotlib.pyplot as plt
    example = gaussian_random_field()
    plt.imshow(example)
    """

    # Defines momentum indices
    k_idx = fftind(size)

    # Defines the amplitude as a power law 1/|k|^(alpha/2)
    # amplitude = np.power(k_idx[0] ** 2 + k_idx[1] ** 2 + 1e-10, -alpha / 4.0)
    amplitude = np.power(k_idx[0] ** 2 + k_idx[1] ** 2 + 1e-10, -alpha / 3.0)
    amplitude[0, 0] = 0

    # Draws a complex gaussian random noise with normal
    # (circular) distribution
    noise = np.random.normal(size=(size, size)) + 1j * np.random.normal(
        size=(size, size)
    )

    # To real space
    gfield = np.fft.ifft2(noise * amplitude).real

    # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield / np.std(gfield)

    return gfield


def convert_to_tecu(frequency, phscreen):
    frequency *= 1e6
    stec = 1 / constant * phscreen * frequency ** 2
    stec /= 1e16
    return stec  # slant TEC


def scale_to_pi_range(phscreen):
    """Scaling our produced phase screen to between [0, 180] degrees

    Parameters
    ----------
    phscreen : array.
        The phase screen to be scaled.

    Returns
    -------
    array.
        Scaled array.
    """
    return np.pi * (
        (phscreen - np.min(phscreen)) / (np.max(phscreen) - np.min(phscreen))
    )


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

        tec = np.sin(7 * (-0.1 * tau + 0.8 * phi))
        # np.fliplr(tec)[np.triu_indices(npix, npix / 3.5)] = np.pi * 1e-6
        tec[np.triu_indices(npix, npix / 4)] = np.pi * 1e-6
        # np.rot90(tec)[np.triu_indices(npix, npix / 2)] = np.pi * 1e-6
        tec[np.tril_indices(0, npix / 4)] = np.pi * 1e-6
        tec += np.random.normal(loc=1e-6, scale=0.1, size=(npix, npix))  # noise
        # for once lets change ridge orientation to along minor diagonal.
        # tec = np.fliplr(tec)
        """

        tec = np.random.normal(loc=1e-6, scale=0.1, size=(npix, npix)) + np.sin(
            10 * (0.9 * tau + 0.001 * phi)
        )
        pix = tec.shape[0]
        lim = 2 * pix // 5
        xlim = 4 * pix // 5
        tec[:, 0:lim] = np.pi * 1e-6
        tec[:, xlim:pix] = np.pi * 1e-6
        """
    else:
        tec = np.tile(np.linspace(0, 10 * np.pi, npix), (npix, 1))
    return tec


def make_phase_screen(scale=100, size=170000, tec_type="l"):
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

    # the number of pixel per side of tec plane
    resolution = int(size // scale)

    if tec_type == "k":
        apply_filter = True
        # pb = PowerBox(
        #     resolution, lambda k: 10 * k ** kolmogorov_index, ensure_physical=True
        # )
        # phs_screen = pb.delta_x()

        phs_screen = gaussian_random_field(alpha=1.6667, size=resolution)

    elif tec_type == "s":
        apply_filter = True
        phs_screen = linear_tec(resolution, sine=True)

    else:
        apply_filter = True
        phs_screen = linear_tec(resolution, sine=False)

    phs_screen = scale_to_pi_range(phs_screen)
    if apply_filter:
        sigma = [30, 30]
        phs_screen = sp.gaussian_filter(phs_screen, sigma, mode="constant")

    return phs_screen


# def get_tec_value_old(
#     tec, us, vs, zen_angles, azimuths, scale, h_pix, pp_u_offset, pp_v_offset, refant=0
# ):
#     """
#     OLD/DEPRECATED. HAS NO GRADIENT!
#     Given a source position (zenith and azimuth angles) with reference to the reference antenna, this function obtains
#     the tec phase offset value at the pierce point corresponding to each antenna. \n \n

#     Function inspired by: https://stackoverflow.com/a/22778207/7905494 \n
#     Plotting a point(s) of a circle laying on a plane given the origin, radius and angle(s).

#     Parameters
#     ----------
#     tec : 2Darray.
#         The TEC/phase screen. \n
#     us : arraylike.
#         x coordinates for each antenna. Named 'us' bcause the array is projected into the uv space.\n
#     vs : arraylike.
#         y coordinates for each antenna. \n
#     zen_angle : float.
#         The zenith angle of the source as calculated from the array earth location and observation time. \n
#     az : float.
#         The azimuth angle.
#     scale : int, optional
#         Scaling factor. the number of metres represented by the side of a single pixel], by default 10. \n
#     h : int, optional
#         Height of the tec screen in metres, by default 200000. \n
#     refant : int, optional
#         Reference antenna ID, by default 0

#     Returns
#     -------
#     1: array.
#         x coordinates for the pierce points of each antenna.\n
#     2: array.
#         y coordinates for the pierce points of each antenna.\n
#     3: array.
#         The TEC/phase offset value at the piercepoint of each antenna.
#     """
#     # print(
#     #    "tecscreen size:  %s by %s km. Height:  %s km."
#     #    % (
#     #        tec.shape[0] * scale / 1000,
#     #        tec.shape[1] * scale / 1000,
#     #        h_pix * scale / 1000,
#     #    )
#     # )
#     # Apply scaling to the array field and tec height.
#     us_scaled = scale_to_pixel_range(us, scale=scale)
#     vs_scaled = scale_to_pixel_range(vs, scale=scale)

#     u_tec_center = tec.shape[0] // 2  # + us_scaled[refant]
#     v_tec_center = tec.shape[1] // 2  # + us_scaled[refant]

#     u_per_source, v_per_source, tec_per_source = [], [], []
#     for zen_angle, az in zip(zen_angles, azimuths):
#         u_tec_list, v_tec_list, tec_per_ant = [], [], []
#         for u, v in zip(us_scaled, vs_scaled):
#             # For each antenna, the antenna position becomes a new origin
#             # This antenna position first has to be in reference to the refant.
#             new_u0 = u - us_scaled[refant]
#             new_v0 = v - vs_scaled[refant]
#             # For each antenna, the zenith angle projects a circle onto the tec screen.
#             # the radius of the circle is given by:
#             # zen_angle should be in radians
#             zen_angle_radius = h_pix * np.tan(zen_angle)
#             # The azimuth angle gives us the arc on this circle from some starting point
#             # We can then obtain the u and v coordinates for the pierce point.
#             pp_u = zen_angle_radius * np.sin(az) + new_u0
#             pp_v = zen_angle_radius * np.cos(az) + new_v0

#             # Collect pierce points for each antenna.
#             uu = tec.shape[0] - (pp_u - pp_u_offset + u_tec_center)
#             vv = pp_v - pp_v_offset + v_tec_center
#             u_tec_list.append(uu)
#             v_tec_list.append(vv)
#             # phase offset value per pierce point.
#             tec_per_ant.append(tec[int(round(uu)), int(round(vv))])
#         u_per_source.append(u_tec_list)
#         v_per_source.append(v_tec_list)
#         tec_per_source.append(tec_per_ant)

#     return np.array(u_per_source), np.array(v_per_source), np.array(tec_per_source)


# def get_tec_value(
#     tec, us, vs, zen_angles, azimuths, scale, h_pix, pp_u_offset, pp_v_offset, refant=0
# ):
#     """
#     Given a source position (zenith and azimuth angles) with reference to the reference antenna, this function obtains
#     the tec phase offset value at the pierce point corresponding to each antenna. \n \n

#     Function inspired by: https://stackoverflow.com/a/22778207/7905494 \n
#     Plotting a point(s) of a circle laying on a plane given the origin, radius and angle(s).

#     Parameters
#     ----------
#     tec : 2Darray.
#         The TEC/phase screen. \n
#     us : arraylike.
#         x coordinates for each antenna. Named 'us' because the array is projected into the "uv" space.\n
#     vs : arraylike.
#         y coordinates for each antenna. \n
#     zen_angle : float.
#         The zenith angle of the source as calculated from the array earth location and observation time. \n
#     az : float.
#         The azimuth angle.
#     scale : int, optional
#         Scaling factor. the number of metres represented by the side of a single pixel], by default 10. \n
#     h : int, optional
#         Height of the tec screen in metres, by default 200000. \n
#     refant : int, optional
#         Reference antenna ID, by default 0

#     Returns
#     -------
#     1: array.
#         x coordinates for the pierce points of each antenna.\n
#     2: array.
#         y coordinates for the pierce points of each antenna.\n
#     3: array.
#         The TEC/phase offset value at the piercepoint of each antenna.
#     """
#     # print(
#     #    "tecscreen size:  %s by %s km. Height:  %s km."
#     #    % (
#     #        tec.shape[0] * scale / 1000,
#     #        tec.shape[1] * scale / 1000,
#     #        h_pix * scale / 1000,
#     #    )
#     # )
#     # Lets first get the gradient of all pixwls in the tec
#     du, dv = np.gradient(tec)
#     # Apply scaling to the array field and tec height.
#     us_scaled = scale_to_pixel_range(us, scale=scale)
#     vs_scaled = scale_to_pixel_range(vs, scale=scale)

#     u_tec_center = tec.shape[0] // 2  # + us_scaled[refant]
#     v_tec_center = tec.shape[1] // 2  # + us_scaled[refant]

#     u_per_source, v_per_source, u_vec_per_source, v_vec_per_source = [], [], [], []
#     for zen_angle, az in zip(zen_angles, azimuths):
#         u_tec_list, v_tec_list, u_vec_list, v_vec_list = [], [], [], []
#         for u, v in zip(us_scaled, vs_scaled):
#             # For each antenna, the antenna position becomes a new origin
#             # This antenna position first has to be in reference to the refant.
#             new_u0 = u - us_scaled[refant]
#             new_v0 = v - vs_scaled[refant]
#             # For each antenna, the zenith angle projects a circle onto the tec screen.
#             # the radius of the circle is given by:
#             # zen_angle should be in radians
#             zen_angle_radius = h_pix * np.tan(zen_angle)
#             # The azimuth angle gives us the arc on this circle from some starting point
#             # We can then obtain the u and v coordinates for the pierce point.
#             pp_u = zen_angle_radius * np.sin(az) + new_u0
#             pp_v = zen_angle_radius * np.cos(az) + new_v0

#             # Collect pierce points for each antenna.
#             uu = tec.shape[0] - (pp_u - pp_u_offset + u_tec_center)
#             vv = pp_v - pp_v_offset + v_tec_center
#             u_tec_list.append(uu)
#             v_tec_list.append(vv)

#             # Get the gradient

#             u_vec = du[int(round(uu)), int(round(vv))]
#             v_vec = dv[int(round(uu)), int(round(vv))]
#             u_vec_list.append(u_vec)
#             v_vec_list.append(v_vec)

#             # phase offset value per pierce point.
#             # tec_per_ant.append(tec[int(round(uu)), int(round(vv))])
#         u_per_source.append(u_tec_list)
#         v_per_source.append(v_tec_list)

#         # tec_per_source.append(tec_per_ant)
#         u_vec_per_source.append(u_vec_list)
#         v_vec_per_source.append(v_vec_list)

#     return (
#         np.array(u_per_source),
#         np.array(v_per_source),
#         np.array(u_vec_per_source),
#         np.array(v_vec_per_source),
#     )  # np.array(tec_per_source)


# def get_xy_phase_vectors(tec, u_per_source, v_per_source):
#     dx, dy = np.gradient(tec)
#     vec_u, vec_v = [], []
#     for u, v in zip(u_per_source, v_per_source):
#         vec_u.append(dx[u, v])
#         vec_v.append(dy[u, v])
#     return vec_u, vec_v
