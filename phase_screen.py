#! /usr/bin/env python
__author__ = [
    "Kariuki Chege",
]

from casacore.tables import table
import numpy as np
from powerbox import PowerBox

from plotting import plot_antennas_on_tec_field
from coordinates import radec_to_altaz, get_time, MWAPOS

c = 299792458
constant = 40.30819
tecu = 1e16
frequency = 150


def linear_tec(npix):
    """Simple linear TEC plane increasing uniformly in one direction"""
    return np.tile(np.linspace(0, 1, npix), (npix, 1))


def iono_phase_shift(p, frequency, scale=5, size=50000, kolmogorov=False):
    """
    Should convert a tec screen to a phase screen using the commented formula.
    Not yet doing that.
    """
    # \Delta STEC [1/m^3] = \Delta\theta [rad] * \nu^2 [(MHz)^2] / k
    # k = 1/(8\pi^2) e^2/(\epsilon_0 m_e) [m^3 / s^2]
    # 1 TECU = 10^16 [1/m^2]

    # Hence: $\Delta\theta = 40.30819/1e16 * 1/\nu^2 * STEC

    resolution = size // scale  # the number of pixel per side of tec plane

    if kolmogorov:
        pb = PowerBox(resolution, lambda k: 10 * k ** p, ensure_physical=True)
        tec = pb.delta_x()
    else:
        tec = linear_tec(resolution)

    # ammend formula here.
    # phase_screen = constant/tecu * 1/(frequency**1e6) * tec -->
    # shouts "OverflowError: (34, 'Numerical result out of range')"
    phase_screen = tec

    return phase_screen


def get_phase_center(tbl):
    """Grabs the phase centre of the observation in RA and Dec"""
    ra0, dec0 = tbl.FIELD.getcell("PHASE_DIR", 0)[0]
    print("The phase center is at ra=%s, dec=%s" % (np.degrees(ra0), np.degrees(dec0)))
    return ra0, dec0


def antenna_pos(mset):
    """Extracts the antenna positions in XYZ coordinates from the MS"""
    t = table(mset + "/ANTENNA", ack=False)
    pos = t.getcol("POSITION")
    t.close()
    return pos


def get_bl_vectors(mset, refant=0):
    """Uses the antenna XYZ position values and recalculates them with the reference antenna as the origin."""
    pos = antenna_pos(
        mset
    )  # First get the positions of each antenna recorded in XYZ values
    no_ants = len(pos)
    print("The mset has %s antennas." % (no_ants))

    bls = np.zeros((no_ants, 3))
    for i in range(no_ants):  # calculate and fill bls with distances from the refant
        pos1, pos2 = pos[i], pos[refant]
        bls[i] = np.array([pos2 - pos1])
    return bls


def get_antenna_in_uvw(mset, tbl):
    """
    Convert the antennas positions into some sort of uv plane with one antenna position as the center of the field.
    """
    ra0, dec0 = get_phase_center(tbl)
    ra0 *= -1  # because we have assumed local LST=0hrs and HA = LST-RA
    xyz = get_bl_vectors(mset)

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


def scale_to_pixel_range(us, scale=5):
    """
    Scale antenna positions into the axis range of the tec field.
    Set the scale in meters - length represented by side of a pixel.
    """
    # refxx = int(tec.shape[1]//2.)
    # refyy = int(tec.shape[0]//2.)
    pixel_max = max(us) / scale
    pixel_min = 0
    min_u = min(us)
    max_u = max(us)
    scaled = [
        ((u - min_u) / (max_u - min_u)) * (pixel_max - pixel_min) + pixel_min
        for u in us
    ]
    return np.array(scaled)


# **Obsolete**
def get_tec_value_old(tec, us, vs, zen_x, zen_y, scale=5, h=200000, refpos=0):
    """
    Obtain the tec value at the positon corresponding to each antenna
    Each pixel represents a distance of tecscreen_height/no_of_tec_axis _pixels e.g 200000/2024 = 98.8m
    So, if tec =1024*1024 pixels and h=2024, the tec screen is 200km high and,
    the tec field is 100km in each direction.
    """
    print(
        "tecscreen is size %s by %s km and at height %s km."
        % (tec.shape[0] * scale / 1000, tec.shape[1] * scale / 1000, h / 1000)
    )

    us_scaled = scale_to_pixel_range(us)
    vs_scaled = scale_to_pixel_range(vs)

    u_tec_list, v_tec_list, tec_per_ant = [], [], []

    h_pix = h / scale  # appling scaling to phase screen height

    for u, v in zip(us_scaled, vs_scaled):
        u_to_origin = u - us_scaled[0]
        v_to_origin = v - vs_scaled[0]
        u_tec = int(u_to_origin + h_pix * np.tan(np.deg2rad(zen_x)))
        v_tec = int(v_to_origin + h_pix * np.tan(np.deg2rad(zen_y)))

        u_tec_list.append(u_tec)
        v_tec_list.append(v_tec)
        tec_per_ant.append(tec[u_tec + int(us_scaled[0]), v_tec + int(vs_scaled[0])])

    return np.array(u_tec_list), np.array(v_tec_list), np.array(tec_per_ant)


# Rewrote function below from old one above that needs arbitrary zen angles
#       in both u and v directions
# Function inspired by: 'https://stackoverflow.com/a/22778207/7905494'
#       Plotting a point(s) of a circle laying on a plane given origin, radius
#       and angle(s).
# This one now can take zenith and azimuth angles calculated from position of
#       a source in the sky.
def get_tec_value(tec, us, vs, zen_angle, az, scale=5, h=200000, refant=0):
    """
    Obtain the tec value at the positon corresponding to each antenna
    Each pixel represents a distance of tecscreen_height/no_of_tec_axis _pixels e.g 200000/2024 = 98.8m
    So, if tec =1024*1024 pixels and h=2024, the tec screen is 200km high and,
    the tec field is 100km in each direction.
    """
    print(
        "tecscreen is size %s by %s km and at height %s km."
        % (tec.shape[0] * scale / 1000, tec.shape[1] * scale / 1000, h / 1000)
    )
    # Apply scaling to the array field and tec height.
    # We are using the overall chosen scale of say, 1pixel_side = 5m
    us_scaled = scale_to_pixel_range(us)
    vs_scaled = scale_to_pixel_range(vs)
    h_pix = h / scale

    u_tec_list, v_tec_list, tec_per_ant = [], [], []
    for u, v in zip(us_scaled, vs_scaled):
        # For each antenna, the antenna position becomes a new origin
        # This antenna position first has to be in reference to the refant.
        new_u0 = u - us_scaled[refant]
        new_v0 = v - vs_scaled[refant]
        # For each antenna, the zenith angle projects a circle onto the tec screen.
        # the radius of the circle is given by:
        zen_angle_radius = h_pix * np.tan(np.deg2rad(zen_angle))
        # The azimuth angle gives us the arc on this circle from some starting point
        # We can then obtain the u and v coordinates for the pierce point.
        # The pi rad added to azimuth angle  is to rotate the start point of the arc
        #   to the topside of the tec field.
        pp_u = zen_angle_radius * np.sin(az + np.pi) + new_u0
        pp_v = zen_angle_radius * np.cos(az + np.pi) + new_v0
        # Collect pierce points for each antenna.
        u_tec_list.append(pp_u)
        v_tec_list.append(pp_v)
        tec_per_ant.append(
            tec[
                int(round(pp_u)) + int(round(us_scaled[refant])),
                int(round(pp_v)) + int(round(vs_scaled[refant])),
            ]
        )

    return np.array(u_tec_list), np.array(v_tec_list), np.array(tec_per_ant)


#  ---------------------------------------------------------------------------------------


def run_all(ms, metafits, ra, dec, plot=False):
    """
    A function that runs all the functions listed
    above to output the phase offsets per antenna.
    """
    scale = 5  # number of metres represented by one pixel size
    field_size = 50000  # 5km by 5km. Thus our tec array will have shape (10000,10000)
    h = 200000  # 200km height of tec field in the sky above array.
    kolmogorov_index = -1.66667

    tbl = table(ms, readonly=False)
    us, vs, ws = get_antenna_in_uvw(ms, tbl)

    phase_screen = iono_phase_shift(
        kolmogorov_index, frequency, scale=scale, size=field_size, kolmogorov=False
    )

    time = get_time(metafits, MWAPOS)
    alt, azimuth = radec_to_altaz(ra, dec, time, MWAPOS)
    zen_angle = np.pi / 2.0 - alt
    u_tec_list, v_tec_list, tec_per_ant = get_tec_value(
        phase_screen, us, vs, zen_angle, azimuth, scale=scale, h=h
    )

    if plot:
        plot_antennas_on_tec_field(phase_screen, u_tec_list, v_tec_list)
        # pl.plot_antennas_on_uvplane(us,vs, shift=False, alpha=0.1,  name='antennas_on_uvplane.png')

    return tec_per_ant


if __name__ == "__main__":
    ms = "2sources_truesimvis.ms"
    run_all(ms, plot=True)
