import os
import random
from math import sin, cos, acos

import numpy as np
import pandas as pd
import yaml
from yaml import SafeLoader as SafeLoader


def great_circle_dist1(r1, d1, r2, d2):
    return acos(sin(d1) * sin(d2) + cos(d1) * cos(d2) * cos(r1 - r2))


# def deg2rad(x):
#     return x * np.pi / 180.0


def precise_dist(ra1, dec1, ra2, dec2):
    d1 = np.deg2rad(dec1)
    d2 = np.deg2rad(dec2)
    r1 = np.deg2rad(ra1)
    r2 = np.deg2rad(ra2)
    dist = great_circle_dist1(r1, d1, r2, d2)
    dist = dist * 180.0 / np.pi
    return dist


def random_sky_model(
    N, ra0, dec0, ra_radius=24, dec_radius=18, save=True, filename="sky_model.csv"
):
    ras = sample_floats(ra0 - ra_radius, ra0 + ra_radius, size=N)
    decs = sample_floats(dec0 - dec_radius, dec0 + dec_radius, size=N)

    # fluxes = sample_floats(0.5, 15, size=N)
    # make proper source  count fluxes and pick N fluxes randomly
    sky_fluxes = make_fluxes(sky_type="random")
    fluxes = sky_fluxes[np.random.choice(len(sky_fluxes), size=N, replace=False)]

    ras = np.array(ras)
    ras = np.where(ras < 0, ras + 360, ras)
    decs = np.array(decs)
    fluxes = np.array(fluxes)

    # print("RAs range", ras.min(), ras.max())
    # print("Decs range", decs.min(), decs.max())
    # print("Fluxes range", fluxes.min(), fluxes.max())

    df = pd.DataFrame(
        list(zip(list(ras), list(decs), list(fluxes))),
        columns=["ra", "dec", "flux"],
    )
    if save:
        df.to_csv("%s" % (filename))
    return ras, decs, fluxes


def make_fluxes(
    sky_type,
    fluxes=0,
    seed=0,
    k1=4100,
    gamma1=1.59,
    k2=4100,
    gamma2=2.5,
    flux_low=1,  # 40e-3,
    flux_mid=5.0,  # 1,
    flux_high=15.0,  # 5.0,
):
    if sky_type == "random":
        np.random.seed(seed)
        fluxes = stochastic_sky(
            seed, k1, gamma1, k2, gamma2, flux_low, flux_mid, flux_high
        )
    elif sky_type == "point":
        fluxes = np.array(fluxes)
    return fluxes


# def generate_distribution(mean, sigma, size, dist, type="ra"):
#     """
#     #Other distribution types
#     if dist == "constant":
#         return np.ones(size) * mean
#     elif dist == "lognormal":
#         return np.random.lognormal(loc=mean,
#                                    sigma=sigma,
#                                    size=size)
#     """
#     if dist == "normal":
#         d = np.random.normal(loc=mean, scale=sigma, size=size)
#         if type == "ra":
#             print("RAs range", d.min(), d.max())
#             d = np.where(d < 0, d + 360, d)
#         elif type == "dec":
#             print("Decs range", d.min(), d.max())
#         else:
#             d = np.random.normal(loc=mean, scale=sigma, size=size)
#         return d

#     else:
#         raise ValueError("Unrecognised distribution ({}).".format(dist))


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


def sample_floats(low, high, size=1):
    """Return a k-length list of unique random floats
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


def stochastic_sky(
    seed=0, k1=4100, gamma1=1.59, k2=4100, gamma2=2.5, S_low=400e-3, S_mid=1, S_high=5.0
):
    np.random.seed(seed)

    # Franzen et al. 2016
    # k1 = 6998, gamma1 = 1.54, k2=6998, gamma2=1.54
    # S_low = 0.1e-3, S_mid = 6.0e-3, S_high= 400e-3 Jy

    # Cath's parameters
    # k1=4100, gamma1 =1.59, k2=4100, gamma2 =2.5
    # S_low = 0.400e-3, S_mid = 1, S_high= 5 Jy

    if S_low > S_mid:
        norm = (
            k2 * (S_high ** (1.0 - gamma2) - S_low ** (1.0 - gamma2)) / (1.0 - gamma2)
        )
        n_sources = np.random.poisson(norm * 2.0 * np.pi)
        # generate uniform distribution
        uniform_distr = np.random.uniform(size=n_sources)
        # initialize empty array for source fluxes
        source_fluxes = np.zeros(n_sources)
        source_fluxes = (
            uniform_distr * norm * (1.0 - gamma2) / k2 + S_low ** (1.0 - gamma2)
        ) ** (1.0 / (1.0 - gamma2))
    else:
        # normalisation
        norm = k1 * (S_mid ** (1.0 - gamma1) - S_low ** (1.0 - gamma1)) / (
            1.0 - gamma1
        ) + k2 * (S_high ** (1.0 - gamma2) - S_mid ** (1.0 - gamma2)) / (1.0 - gamma2)
        # transition between the one power law to the other
        mid_fraction = (
            k1
            / (1.0 - gamma1)
            * (S_mid ** (1.0 - gamma1) - S_low ** (1.0 - gamma1))
            / norm
        )
        n_sources = np.random.poisson(norm * 2.0 * np.pi)

        #########################
        # n_sources = 1e5
        #########################

        # generate uniform distribution
        uniform_distr = np.random.uniform(size=n_sources)
        # initialize empty array for source fluxes
        source_fluxes = np.zeros(n_sources)

        source_fluxes[uniform_distr < mid_fraction] = (
            uniform_distr[uniform_distr < mid_fraction] * norm * (1.0 - gamma1) / k1
            + S_low ** (1.0 - gamma1)
        ) ** (1.0 / (1.0 - gamma1))

        source_fluxes[uniform_distr >= mid_fraction] = (
            (uniform_distr[uniform_distr >= mid_fraction] - mid_fraction)
            * norm
            * (1.0 - gamma2)
            / k2
            + S_mid ** (1.0 - gamma2)
        ) ** (1.0 / (1.0 - gamma2))
    return source_fluxes


def gleam_model(
    csvfile="Gleam_low_band_catalogue_si.csv", filename="gleam_model_20_deg_radius.csv"
):

    df = pd.read_csv(csvfile, sep=";")
    df2 = df[df["Fp151"] > 0.4]
    df2 = df.dropna(axis=0)

    ra0 = np.rad2deg(ra0)
    dec0 = np.rad2deg(dec0)

    if filename not in os.listdir(os.path.abspath(".")):
        print('saving model"s RTS data products to csv file..')
        df2.to_csv("%s" % (filename))

        print(df2.shape)
        ras = np.array(df2.RAJ2000)
        decs = np.array(df2.DEJ2000)
        fluxes = np.array(df2.Fp151)

        print("RAs range", ras.min(), ras.max())
        print("Decs range", decs.min(), decs.max())
        print("Fluxes range", fluxes.min(), fluxes.max())
        return ras, decs, fluxes


if __name__ == "__main__":
    ras, decs, fluxes = random_sky_model(50, 0.0, -27.0, save=False)
    print(np.max(fluxes))
