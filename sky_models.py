import os
import random
import numpy as np
import yaml
import pandas as pd
from yaml import SafeLoader as SafeLoader


def loadfile(data_file, n_sources, model_only=True):
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

    df3 = df2[(df2.ra < 9) & (df2.ra > -9) & (df2.dec > -35.1) & (df2.dec < -18.1)]
    print(df3.shape, df3.ra.max(), df3.ra.min(), df3.dec.max(), df3.dec.min())
    df3 = df3.nlargest(n_sources, "flux", keep="all")
    print("min/maximum flux", df3.flux.min(), df3.flux.max())

    filename = data_file.split(".")[0] + "_rts_data_products.csv"
    if filename not in os.listdir(os.path.abspath(".")):
        print('saving model"s RTS data products to csv file..')
        df3.to_csv("%s" % (filename))

    if model_only:
        print(df3.shape)
        ras = np.array(df3.ra)
        decs = np.array(df3.dec)
        fluxes = np.array(df3.flux)
        print(ras.max(), ras.min(), decs.max(), decs.min())
        return ras, decs, fluxes
    else:
        return df3


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
            d[0] = 0.0
            print("RAs range", d.min(), d.max())
            d = np.where(d < 0, d + 360, d)
        elif type == "dec":
            print("Decs range", d.min(), d.max())
            d[0] = -27.0
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


def random_model(N, simple=False, filename="model.txt"):
    if simple:
        return (
            [0.0],
            [-27.0],
            [1],
        )  # [0.0, 7.5, 349.5], [-27.0, -19.6, -35.1], [1, 1, 1]
    ras = sample_floats(
        -9, 9, size=N
    )  # generate_distribution(0.0, 4.0, N, "normal", type="ra")
    decs = sample_floats(
        -35.1, -18.1, size=N
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
    with open("%s" % (filename), "w") as f:
        for (ra, dec, flux) in zip(ras, decs, fluxes):
            f.write("{0},{1},{2}\n".format(ra, dec, flux))
        return ras, decs, fluxes
