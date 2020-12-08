import pandas as pd
from argparse import ArgumentParser
import numpy as np
import os
import sys


def extract_sources(true_image, offset_image):
    """Run source-finder on both a true and offset sky images

    Parameters
    ----------
    true_image : string
        Path to the true sky fits format image
    offset_image : string
        path to the offset sky fits format image

    Returns
    -------
    tuple(string, string)
        path to a true and an offset sky catalog in csv format
    """
    os.system("BANE %s" % (true_image))
    os.system("BANE %s" % (offset_image))
    pref = true_image.split("_")[0] + "_" + true_image.split("_")[1]
    true_catalog = "%s_true_sources.csv" % (pref)
    off_catalog = "%s_offset_sources.csv" % (pref)

    os.system(
        "aegean --autoload %s --seedclip 6 --table %s" % (true_image, true_catalog)
    )
    os.system(
        "aegean --autoload %s --seedclip 6 --table %s" % (offset_image, off_catalog)
    )
    return true_catalog, off_catalog


def load_data(true_catalog, off_catalog):
    """Reads the catalogues produced by the "extract_sources" function.

    Parameters
    ----------
    true_catalog : string
        path to a truesky catalog in csv format
    off_catalog : [type]
        path to an offset sky catalog in csv format

    Returns
    -------
    tuple(array, pandas dataframe, pandas dataframe)
        columns in the catlogues and dataframes for both catalogues
    """
    # Read both csv files and grab the column names
    true_sky = pd.read_csv(true_catalog)
    offset_sky = pd.read_csv(off_catalog)

    print(
        len(true_sky.loc[true_sky["source"] > 0]),
        " 'components' in true catalog skipped!",
    )
    print(
        len(offset_sky.loc[offset_sky["source"] > 0]),
        " 'components' in offset catalog skipped!",
    )
    true_sky = true_sky.loc[true_sky["source"] == 0]
    offset_sky = offset_sky.loc[offset_sky["source"] == 0]
    cols = true_sky.columns

    # sort the sources by island from first to last in both dataframes
    df_true_sky = true_sky.sort_values(by=["island"])
    df_offset_sky = offset_sky.sort_values(by=["island"])
    df_true_sky = df_true_sky.reset_index(drop=True)
    df_offset_sky = df_offset_sky.reset_index(drop=True)

    # add 360 deg to all RAs above 300 to make them -ve (depends on the FoV)
    df_true_sky.ra = np.where(
        df_true_sky.ra > 300, df_true_sky.ra - 360, df_true_sky.ra
    )
    df_offset_sky.ra = np.where(
        df_offset_sky.ra > 300, df_offset_sky.ra - 360, df_offset_sky.ra
    )
    return cols, df_true_sky, df_offset_sky


def write_csv(true_data, offset_data, true_cat, offset_cat, cols=""):
    """Save matched catalogues

    Parameters
    ----------
    true_data : array
        Sorted true sky data
    offset_data : array
        Sorted offset sky data
    true_cat : string
        true sky catalog name
    offset_cat : string
        offset sky catalog name
    cols : str, optional
        Names of the columns, by default ""

    Returns
    -------
    tuple(string, string)
        Paths to the sorted catalogues
    """
    s_df_true_sky = pd.DataFrame(true_data, columns=cols)
    s_df_offset_sky = pd.DataFrame(offset_data, columns=cols)

    sorted_df_true_sky = "sorted_" + true_cat.split(".")[0] + ".csv"
    sorted_df_offset_sky = "sorted_" + offset_cat.split(".")[0] + ".csv"

    print("saving sorted catalogues in csv files..")
    s_df_true_sky.to_csv("%s" % (sorted_df_true_sky), index=False)
    s_df_offset_sky.to_csv("%s" % (sorted_df_offset_sky), index=False)

    return sorted_df_true_sky, sorted_df_offset_sky


def match_dem(df_true_sky, df_offset_sky, dra, ddec, dflux):
    """Trys to match sources from two catalogues for correct calculation of position offsets and flux scintillation.

    Parameters
    ----------
    df_true_sky : pandas dataframe
        The reference dataframe
    df_offset_sky : pandas dataframe
        The second dataframe for matching \n
    dra : float
        The maximun RA offset allowed in degrees.\n
    ddec : float
        The maximun Dec offset allowed in degrees.\n
    dflux : float
        The maximun flux difference allowed in degrees.

    Returns
    -------
    3 lists.
        1: list of all matched islands.
        2: list of the first matched catalogue.
        3: list of the second matched catalogue.
    """
    matched_islands_t = []
    matched_islands_o = []
    data_true = []
    data_offset = []

    for i in df_true_sky.island.values:
        print("-----------------------------------------------------")
        print("True sky island no: ", i)
        for j in df_offset_sky.island.values:
            if j not in matched_islands_o:
                # try:
                # print("island", j)
                rx = float(df_true_sky.loc[df_true_sky["island"] == i]["ra"].values)
                ry = float(df_offset_sky.loc[df_offset_sky["island"] == j]["ra"].values)
                d_ra = rx - ry
                # print("type d_ra", type(d_ra), d_ra)

                dx = float(df_true_sky.loc[df_true_sky["island"] == i]["dec"])
                dy = float(df_offset_sky.loc[df_offset_sky["island"] == j]["dec"])
                d_dec = dx - dy
                # print("type d_dec", type(d_dec), d_dec)

                fx = float(df_true_sky.loc[df_true_sky["island"] == i]["int_flux"])
                fy = float(df_offset_sky.loc[df_offset_sky["island"] == j]["int_flux"])

                d_flux = abs(1 - (fy / fx))
                # print("type d_flux", type(d_flux), d_flux)
                # print("d_ra, d_dec, d_flux")
                # print(d_ra)
                # print(d_dec)
                # print(d_flux)
                # except TypeError:
                #    print("Exception, probably multiple islands with same value")
                if (
                    float(abs(d_ra)) < dra
                    and float(abs(d_dec)) < ddec
                    and float(abs(d_flux)) < dflux
                ):
                    print("Offset sky island no: ", j)
                    data_true.append(
                        df_true_sky.loc[df_true_sky["island"] == i]
                        .values.flatten()
                        .tolist()
                    )
                    data_offset.append(
                        df_offset_sky.loc[df_offset_sky["island"] == j]
                        .values.flatten()
                        .tolist()
                    )
                    # make 2 lists of matched islands in each catalogue
                    matched_islands_t.append(i)
                    matched_islands_o.append(j)
                    break

        else:
            continue

    print("Matched sources: ", len(data_true), len(data_offset))

    unmatched_islands_t = []
    unmatched_islands_o = []
    for isl in df_true_sky.island.values:
        if isl not in matched_islands_t:
            unmatched_islands_t.append(isl)
    for isl in df_offset_sky.island.values:
        if isl not in matched_islands_o:
            unmatched_islands_o.append(isl)

    print("true cat unmatched sources: ", len(unmatched_islands_t), unmatched_islands_t)
    print("off cat unmatched sources: ", len(unmatched_islands_o), unmatched_islands_o)

    return unmatched_islands_t, unmatched_islands_o, data_true, data_offset


def repeat_match(
    unmatched_islands_t, unmatched_islands_o, df_true_sky, df_offset_sky, dr, dd, df
):
    """Repeats the catalogue matchin with a different set of parameters

    Parameters
    ----------
    unmatched_islands_t : array
        Previously unmatched islands in true sky catalog
    unmatched_islands_o : array
        Previously unmatched islands in offset sky catalog
    df_true_sky : pandas dataframe
        The reference dataframe
    df_offset_sky : pandas dataframe
        The second dataframe for matching \n
    dr : float
        The maximun RA offset allowed in degrees.\n
    dd : float
        The maximun Dec offset allowed in degrees.\n
    df : float
        The maximun flux difference allowed in degrees.

    Returns
    -------
    4 lists
        1: list of all unmatched islands in true catalogue.
        2: list of all unmatched islands in offset catalogue.
        3: true catalogue.
        3: offset catalogue.
    """
    print(df_true_sky.shape)
    print(df_offset_sky.shape)
    df_true_sky = df_true_sky.loc[df_true_sky["island"].isin(unmatched_islands_t)]
    df_offset_sky = df_offset_sky.loc[df_offset_sky["island"].isin(unmatched_islands_o)]
    (
        r_unmatched_islands_t,
        r_unmatched_islands_o,
        r_data_true,
        r_data_offset,
    ) = match_dem(df_true_sky, df_offset_sky, dr, dd, df)
    return r_unmatched_islands_t, r_unmatched_islands_o, r_data_true, r_data_offset


def final_match(
    r_unmatched_islands_t, r_unmatched_islands_o, df_true_sky, df_offset_sky
):
    """One final match sorry. Might look for a better way to do this in future.

    Parameters
    ----------

    r_unmatched_islands_t : array
        Previously unmatched islands in true sky catalog
    r_unmatched_islands_o : array
        Previously unmatched islands in offset sky catalog
    df_true_sky : pandas dataframe
        The reference dataframe
    df_offset_sky : pandas dataframe
        The second dataframe for matching

    Returns
    -------
    [type]
        [description]
    """
    print(df_true_sky.shape)
    print(df_offset_sky.shape)
    f_data_true, f_data_offset = [], []
    df_true_sky = df_true_sky.loc[df_true_sky["island"].isin(r_unmatched_islands_t)]
    df_offset_sky = df_offset_sky.loc[
        df_offset_sky["island"].isin(r_unmatched_islands_o)
    ]
    for index, row in df_true_sky.iterrows():
        print(index)
        f_data_true.append(row)
    for index, row in df_offset_sky.iterrows():
        print(index)
        f_data_offset.append(row)

    return f_data_true, f_data_offset


def main_match(true_file, offset_file):
    """Performs the full source catalog matching

    Parameters
    ----------
    true_file : string
        path to a true sky catalog/image in csv/fits format
    offset_file : string
        path to an offset sky catalog/image in csv/fits format

    Returns
    -------
    tuple(string, string)
        Paths to the sorted catalogues
    """
    if true_file.split(".")[-1] == "csv":
        true_catalog, off_catalog = true_file, offset_file
    else:
        true_catalog, off_catalog = extract_sources(true_file, offset_file)
        try:
            true_catalog = true_catalog.split(".")[0] + "_comp.csv"
            off_catalog = off_catalog.split(".")[0] + "_comp.csv"
        except FileNotFoundError:
            print("Aegean found 0 sources in the images. Exiting.")
            sys.exit()
    cols, df_true_sky, df_offset_sky = load_data(true_catalog, off_catalog)
    unmatched_islands_t, unmatched_islands_o, data_true, data_offset = match_dem(
        df_true_sky, df_offset_sky, 0.05, 0.05, 0.1
    )
    print("REPEATING MATCH")
    (
        r_unmatched_islands_t,
        r_unmatched_islands_o,
        r_data_true,
        r_data_offset,
    ) = repeat_match(
        unmatched_islands_t,
        unmatched_islands_o,
        df_true_sky,
        df_offset_sky,
        0.07,
        0.07,
        0.2,
    )
    data_true += r_data_true
    data_offset += r_data_offset

    print(len(data_true))
    print(len(data_offset))

    print("REPEATING MATCH AGAIN")
    (
        r2_unmatched_islands_t,
        r2_unmatched_islands_o,
        r2_data_true,
        r2_data_offset,
    ) = repeat_match(
        r_unmatched_islands_t,
        r_unmatched_islands_o,
        df_true_sky,
        df_offset_sky,
        0.08,
        0.08,
        0.5,
    )
    data_true += r2_data_true
    data_offset += r2_data_offset

    """
        print("FINAL MATCH")
        f_data_true, f_data_offset = final_match(
            r_unmatched_islands_t, r_unmatched_islands_o, df_true_sky, df_offset_sky
        )

        data_true += f_data_true
        data_offset += f_data_offset

        print(data_true[0])
        # print(data_offset[0])

        print("WRITING")
    """

    sorted_df_true_sky, sorted_df_offset_sky = write_csv(
        data_true, data_offset, true_file, offset_file, cols=cols
    )
    return sorted_df_true_sky, sorted_df_offset_sky


if __name__ == "__main__":
    """This module takes two images/catalogues and returns two csv file catalogues containing the matching sources from each
    """
    parser = ArgumentParser(
        "python match_catalogs.py", description="Ionospheric effects simulations"
    )
    parser.add_argument(
        "--true_file",
        "-t",
        required=True,
        help="The image/catalog with the true source positions",
    )
    parser.add_argument(
        "--offset_file",
        "-o",
        required=True,
        help="The image/catalog with the offset source positions",
    )
    args = parser.parse_args()

    main_match(args.true_file, args.offset_file)
