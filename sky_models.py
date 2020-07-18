import os
import numpy as np
import yaml
import pandas as pd
from yaml import SafeLoader as SafeLoader


# path = "/home/chege/Desktop/curtin_work/jack_yamls/"
# print("done importing")


def loadfile(data_file, model_only=True):
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

    df3 = df2[(df2.ra < 7.57) & (df2.ra > -7.3) & (df2.dec > -33.8) & (df2.dec < -20)]
    print(df3.ra.max(), df3.ra.min())

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
            d = np.where(d < 0, d + 360, d)
        elif type == "dec":
            d[0] = -27.0
        return d

    else:
        raise ValueError("Unrecognised distribution ({}).".format(dist))


def random_model(N, simple=False, filename="model.txt"):
    if simple:
        return (
            [0.0],
            [-27.0],
            [1],
        )  # [0.0, 7.5, 349.5], [-27.0, -19.6, -35.1], [1, 1, 1]
    ras = generate_distribution(0.0, 4.0, N, "normal", type="ra")
    decs = generate_distribution(-27.0, 4.0, N, "normal", type="dec")
    fluxes = list(np.ones(N))  # np.abs(generate_distribution(1., 3., N, "normal"))

    with open("%s" % (filename), "w") as f:
        for (ra, dec, flux) in zip(ras, decs, fluxes):
            f.write("{0},{1},{2}\n".format(ra, dec, flux))
        return ras, decs, fluxes
