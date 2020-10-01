import os
import random
import numpy as np
import yaml
import pandas as pd
from yaml import SafeLoader as SafeLoader
from astropy.io import fits


from mwa_pb import primary_beam
from coordinates import radec_to_altaz, get_time, MWAPOS

print("Done importing")


def precess_to_j2000(ra, deg):
    from astropy.time import Time
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.coordinates import FK5

    fk5c = SkyCoord(ra * u.degree, deg * u.degree, frame=FK5(equinox=Time("J2013")))
    fk5_2000 = FK5(equinox=Time(2000, format="jyear"))
    ra, dec = (
        fk5c.transform_to(fk5_2000).ra.value,
        fk5c.transform_to(fk5_2000).dec.value,
    )
    return ra, dec


def loadfile(data_file, n_sources, ra0, dec0, filename="sky_model.csv"):
    with open(data_file, "r") as f:
        unpacked = yaml.load(f, Loader=SafeLoader)

    flux = []
    ra = []
    dec = []
    names = []
    ampscales = []
    stds = []
    sourcelist = unpacked["sources"]
    for source in sourcelist:
        # print(unpacked['sources'][source]['name'])
        names.append(unpacked["sources"][source]["name"])
        dec.append(unpacked["sources"][source]["dec"])
        ra.append(unpacked["sources"][source]["ra"])
        flux.append(unpacked["sources"][source]["flux_density"])
        ampscales.append(np.nanmedian(unpacked["sources"][source]["amp_scales"]))
        stds.append(np.nanstd(unpacked["sources"][source]["amp_scales"]))
    df = pd.DataFrame(
        list(zip(ra, dec, ampscales, stds, flux, names)),
        columns=["ra", "dec", "ampscales", "stds", "flux", "names"],
    )
    df2 = df.dropna(axis=0)

    ra0 = np.rad2deg(ra0)
    dec0 = np.rad2deg(dec0)
    df3 = df2[
        (df2.ra < ra0 + 9)
        & (df2.ra > ra0 - 9)
        & (df2.dec > dec0 - 8.1)
        & (df2.dec < dec0 + 8.9)
    ]
    print(df3.shape, df3.ra.max(), df3.ra.min(), df3.dec.max(), df3.dec.min())
    df3 = df3.nlargest(n_sources, "flux", keep="all")

    if filename not in os.listdir(os.path.abspath(".")):
        print('saving model"s RTS data products to csv file..')
        df3.to_csv("%s" % (filename))

        print(df3.shape)
        ras = np.array(df3.ra)
        decs = np.array(df3.dec)
        fluxes = np.array(df3.flux)

        print("RAs range", ras.min(), ras.max())
        print("Decs range", decs.min(), decs.max())
        print("Fluxes range", fluxes.min(), fluxes.max())
        return ras, decs, fluxes


def generate_distribution(mean, sigma, size, dist, type="ra"):
    """
    #Other distribution types
    if dist == "constant":
        return np.ones(size) * mean
    elif dist == "lognormal":
        return np.random.lognormal(loc=mean,
                                   sigma=sigma,
                                   size=size)
    """
    if dist == "normal":
        d = np.random.normal(loc=mean, scale=sigma, size=size)
        if type == "ra":
            print("RAs range", d.min(), d.max())
            d = np.where(d < 0, d + 360, d)
        elif type == "dec":
            print("Decs range", d.min(), d.max())
        else:
            d = np.random.normal(loc=mean, scale=sigma, size=size)
        return d

    else:
        raise ValueError("Unrecognised distribution ({}).".format(dist))


def sample_floats(low, high, size=1):
    """ Return a k-length list of unique random floats
        in the range of low <= x <= high
    """
    result = []
    seen = set()
    for i in range(size):
        x = random.uniform(low, high)
        while x in seen:
            x = random.uniform(low, high)
        seen.add(x)
        result.append(x)
    return result


def random_model(N, ra0, dec0, filename="sky_model.csv"):
    ras = sample_floats(
        ra0 - 9, ra0 + 9, size=N
    )  # generate_distribution(0.0, 4.0, N, "normal", type="ra")
    decs = sample_floats(
        dec0 - 8.1, dec0 + 8.1, size=N
    )  # -1 * generate_distribution(27.0, 4.0, N, "normal", type="dec")
    fluxes = sample_floats(
        0.5, 15, size=N
    )  # np.abs(generate_distribution(1.0, 4.0, N, "normal", type="f"))

    ras = np.array(ras)
    ras = np.where(ras < 0, ras + 360, ras)
    decs = np.array(decs)
    fluxes = np.array(fluxes)

    print("RAs range", ras.min(), ras.max())
    print("Decs range", decs.min(), decs.max())
    print("Fluxes range", fluxes.min(), fluxes.max())

    df = pd.DataFrame(
        list(zip(list(ras), list(decs), list(fluxes))), columns=["ra", "dec", "flux"],
    )
    df.to_csv("%s" % (filename))
    return ras, decs, fluxes


def compute_mwa_beam_attenuation(
    ras, decs, metafits, pos, freq=150e6, zenith_pointing=True
):
    """Compute the beam attenuation

    Parameters
    ----------
    ras : float/array
        source RA
    decs : float/array
        source dec
    metafits : str
        path to observation
    pos : str
        Array longitude and latitude
    freq : float/int, optional
        frequency, by default 150e6
    zenith_pointing : bool, optional
        True if observation is zenith pointing, by default True

    Returns
    -------
    float/array, float/array
        XX, YY beam attenuation for the input direction and frequency.
    """
    # for a zenith pointing all delays are zero, and,
    # you need delays for both XX and YY so need to give it 2 sets of 16 delays
    if zenith_pointing:
        delays = np.zeros((2, 16))
    else:
        with fits.open(metafits) as hdu:
            print("getting delays from metafits")
            delays = list(map(int, hdu[0].header["DELAYS"].split(",")))
            delays = [delays, delays]
    print(f"delays: {delays}")

    time, _ = get_time(metafits, pos)
    alt, az = radec_to_altaz(np.deg2rad(ras), np.deg2rad(decs), time, pos)
    za = np.pi / 2.0 - alt

    print(f"zenith angle: {za} azimuth: {az}")
    XX, YY = primary_beam.MWA_Tile_full_EE(
        za, az, freq=freq, delays=delays, zenithnorm=True, power=True, interp=False,
    )
    return XX, YY


if __name__ == "__main__":
    metafits = "/home/kariuki/scint_sims/mset_data/1098108248.metafits"
    print(
        compute_mwa_beam_attenuation(
            0.0, -27.0, metafits, MWAPOS, zenith_pointing=False
        )
    )
