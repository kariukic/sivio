from __future__ import print_function, division

from astropy.coordinates import AltAz, SkyCoord, EarthLocation
from astropy.io.fits import getheader
from astropy.time import Time
import astropy.units as u
import numpy as np


MWAPOS = EarthLocation.from_geodetic(
    lon="116:40:14.93", lat="-26:42:11.95", height=377.8
)


def get_time(metafits, pos):
    metafits = getheader(metafits)
    time = Time(metafits["DATE-OBS"], location=pos)
    lst = time.sidereal_time("mean")
    return time, lst


def radec_to_altaz(ra, dec, time, pos):
    """
    Convert RA and Dec to altitude and azimuth (Equatorial to Horizontal coordinates).
    ra and dec should be in radians.
    """
    # print("RA: ", ra, "Dec: ", dec)
    # ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    coord = SkyCoord(ra, dec, unit=(u.radian, u.radian))
    coord.time = time + pos.lon.hourangle
    coord = coord.transform_to(AltAz(obstime=time, location=pos))
    return coord.alt.rad, coord.az.rad


if __name__ == "__main__":
    metafits = "/home/kariuki/mset_data/1098108248.metafits"
    # metafits = "/home/kariuki/scint_sims/1065880128.metafits"
    x = get_time(metafits, MWAPOS)
    print(x[0].value)
    time, lst = x
    alt, az = radec_to_altaz(np.radians(60), np.radians(-30), time, MWAPOS)
    zen_angle = np.pi / 2.0 - alt

    c = SkyCoord(ra=60 * u.degree, dec=-30 * u.degree)
    print(c.to_string("hmsdms"))

    print("Time: ", time, "LST: ", lst)
    print("Alt: ", np.rad2deg(alt), "Az: ", np.rad2deg(az))
    print("zenith angle: ", np.rad2deg(zen_angle))
