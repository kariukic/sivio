from cthulhu.reconstruct import Obsid
import pandas as pd
import numpy as np


def cthulhu_analyse(sorted_true_cat, sorted_offset_cat):
    """Runs the cthulhu package (`https://pypi.org/project/cthulhu/`) on matched catalogues

    Parameters
    ----------
    sorted_true_cat : string
        path to true sky catalog produced by match catalogs module
    sorted_offset_cat : string
        path to true sky catalog produced by match catalogs module

    Returns
    -------
    cthulhu obect
        The object carries statistics of the position offsets
    """
    true_sky = pd.read_csv(sorted_true_cat)
    offset_sky = pd.read_csv(sorted_offset_cat)
    ra = true_sky.ra
    dec = true_sky.dec
    assert len(ra) == len(dec) > 0
    ra = np.where(ra > 300, ra - 360, ra)
    ra_shifts = offset_sky.ra - true_sky.ra
    dec_shifts = offset_sky.dec - true_sky.dec
    o = Obsid((ra, dec, ra_shifts, dec_shifts), frequency=138.875, radius=20)
    o.reconstruct_tec(filtering=False)
    o.obsid_metric()
    print("Metrics: ", o.metrics)
    print(o.metric)
    return o
