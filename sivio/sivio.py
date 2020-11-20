import csv
import logging
import multiprocessing
import os
import sys
import time as tm
from argparse import ArgumentParser

import numpy as np
import psutil
from casacore.tables import table

from numba import set_num_threads

import mset_utils as mtls
import plotting
import sky_models
import vis_sim as numba_dance
from beam import hyperbeam
from coordinates import MWAPOS, get_time, radec_to_altaz
from cthulhu_analysis import cthulhu_analyse
from match_catalogs import main_match
from phase_screen import make_phase_screen

__author__ = "Kariuki Chege"
__version__ = "0.1.0"
__date__ = "2020-10-30"


def main():
    parser = ArgumentParser(
        "python run_vis_sim.py", description="Ionospheric effects simulations"
    )
    parser.add_argument("--ms_template", required=True, help="Template measurement set")
    parser.add_argument("--metafits", required=True, help="Path to the metafits file")
    parser.add_argument(
        "--yfile", required=False, help="Path to yaml file to obtain the sky model from"
    )
    parser.add_argument(
        "--tecpath",
        required=False,
        help="path to a .npz saved ready made tecscreen array",
    )
    parser.add_argument(
        "--modelpath",
        required=False,
        help="path to a ready made csv file sky model. \
            Just ra, dec and flux rows for now. Flat spectrum point sources.",
    )
    parser.add_argument(
        "--sim",
        required=False,
        action="store_true",
        help="Run in simulation mode. Otherwise, \
            you can just opt to image or plot already simulated data",
    )
    parser.add_argument(
        "--radius",
        "-r",
        type=float,
        default=20.0,
        help="Radius of the GLEAM sky area to obtain sources from, centered at phase center",
    )
    parser.add_argument(
        "--n_sources",
        "-n",
        type=int,
        default=10,
        help="Number of point sources to simulate",
    )
    parser.add_argument(
        "--flux_cutoff",
        "-c",
        type=float,
        default=0.0,
        help="The sky model minimum flux density",
    )
    parser.add_argument(
        "--spar", type=int, default=20, help="Number of point sources to simulate",
    )
    parser.add_argument(
        "--offset_vis",
        "-o",
        action="store_true",
        help="corrupt the visibilities with ionospheric activity",
    )
    # parser.add_argument("--scint_vis", action="store_true")
    # parser.add_argument("--rdiff", type=float, default=5, help="Diffractive scale [m]")
    parser.add_argument(
        "--true_vis",
        "-t",
        action="store_true",
        help="Simulate the true (un-corrupted) visibilities",
    )
    # parser.add_argument(
    #     "--size", type=int, default=140000, help="TEC field size per side [m]"
    # )
    parser.add_argument(
        "--scale", type=int, default=100, help="pixel to distance scaling [m]"
    )
    parser.add_argument(
        "--height", type=int, default=200000, help="TEC plane height from ground [m]"
    )
    parser.add_argument(
        "--tec_type",
        type=str,
        default="l",
        help="l = linear TEC, s = TEC modulated with sine ducts,\
              k = TEC with kolmogorov turbulence.",
    )
    parser.add_argument(
        "--match",
        "-m",
        action="store_true",
        help="match sources from both clean and corrupt images",
    )
    parser.add_argument(
        "--match_done",
        action="store_true",
        help="You already have matched catalogues, go straight to cthulhu plotting",
    )
    parser.add_argument(
        "--image",
        "-i",
        action="store_true",
        help="Run wsclean. Default settings are: \
            '-abs-mem 40 -size 4096 4096 -scale 30asec -niter 1000000 -auto-threshold 3'",
    )
    parser.add_argument("--plot", "-p", action="store_true", help="Make plots")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    # logging configuration
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(module)s:%(levelname)s %(message)s",
    )
    log = logging.getLogger("sivio")
    logging_level = logging.DEBUG if args.debug else logging.INFO
    log.setLevel(logging_level)
    log.info("This is SIVIO {0}-({1})".format(__version__, __date__))

    log.info(
        f"No. of CPUS: {multiprocessing.cpu_count()}: Total memory: ~{psutil.virtual_memory().total/1e9} GB"
    )

    if args.modelpath is not None:
        modelpath = os.path.abspath(args.modelpath)
    if args.tecpath is not None:
        tecpath = os.path.abspath(args.tecpath)
    ms_template_path = os.path.abspath(args.ms_template)
    metafitspath = os.path.abspath(args.metafits)

    if "/" in args.ms_template:
        obsid = args.ms_template.split("/")[-1].split(".")[0]
    else:
        obsid = args.ms_template.split(".")[0]

    mset = "%s_sources_%s_%stec.ms" % (args.n_sources, obsid, args.tec_type,)
    prefix = mset.split(".")[0]
    output_dir = os.path.abspath(".") + "/" + prefix + "_spar" + str(args.spar)
    log.info(f"Output path is {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    if args.sim:
        if mset not in os.listdir(os.path.abspath(".")):
            log.info("Making the simulation measurement set..")
            os.system("mkdir %s" % (mset))
            os.system("cp -r %s/* %s" % (ms_template_path, mset))

        tbl = table(mset, readonly=False)
        ra0, dec0 = mtls.get_phase_center(tbl)
        frequencies = mtls.get_channels(tbl, ls=False)
        log.info(f"The phase center is at ra={np.degrees(ra0)}, dec={np.degrees(dec0)}")

        if args.modelpath is not None:
            ras, decs, fluxes = [], [], []
            with open(modelpath, "r") as file:
                reader = csv.reader(file, delimiter=",")
                next(reader)
                for row in reader:
                    ras.append(float(row[1]))
                    decs.append(float(row[2]))
                    fluxes.append(float(row[3]))
            ras, decs, fluxes = np.array(ras), np.array(decs), np.array(fluxes)
        else:
            if args.yfile is not None:
                model_textfile = prefix + "yaml_sky_model.csv"
                ras, decs, fluxes = sky_models.loadfile(
                    args.yfile,
                    args.n_sources,
                    np.rad2deg(ra0),
                    np.rad2deg(dec0),
                    filename=model_textfile,
                )
            else:
                model_textfile = prefix + "_sky_model.csv"
                # ras, decs, fluxes = sky_models.random_sky_model(
                #     args.n_sources,
                #     np.rad2deg(ra0),
                #     np.rad2deg(dec0),
                #     filename=model_textfile,
                # )
                # ras, decs, fluxes = sky_models.gleam_model(
                #     frequencies,
                #     csvfile="/home/kariuki/databases/Gleam_low_band_catalogue_si.csv",
                #     filename=model_textfile,
                #     metafits=metafitspath,
                # )
                names, ras, decs, alphas, ref_fluxes = sky_models.read_gleam(
                    np.degrees(ra0),
                    np.degrees(dec0),
                    radius=args.radius,
                    n_sources=args.n_sources,
                    min_flux_cutoff=args.flux_cutoff,
                )
        fluxes = sky_models.gleam_model(
            ras, decs, alphas, ref_fluxes, frequencies
        ).astype(complex)

        ras = np.radians(ras)
        decs = np.radians(decs)

        log.info(f"The sky model has {len(ras)} sources")
        # get beam stuff
        time, lst = get_time(metafitspath, MWAPOS)
        alts, azimuths = radec_to_altaz(ras, decs, time, MWAPOS)
        zen_angles = np.pi / 2.0 - alts

        # compute apparent fluxes for all sources and each frequency channel
        start = tm.time()
        fluxes = hyperbeam(
            zen_angles, azimuths, frequencies, fluxes, metafits=metafitspath
        )
        print(fluxes.shape, fluxes[:, 0, :])
        log.info("hyperbeam elapsed: %g", tm.time() - start)

        # if model_textfile not in os.listdir(os.path.abspath(".")):
        #     df3 = pd.DataFrame(
        #         list(zip(names, ras, decs, fluxes)),
        #         columns=["name", "ra", "dec", "flux"],
        #     )
        #     log.info('saving model"s RTS data products to csv file..')
        #     df3.to_csv(
        #         f"{model_textfile}", index=False,
        #     )

        # assert len(ras) == len(decs) == len(fluxes)
        log.info(f"The sky model has {len(ras)} sources")
        data, lmbdas, uvw_lmbdas, dnu, ls, ms, ns = numba_dance.sim_prep(tbl, ras, decs)

        if args.true_vis:
            # logger.info("Simulating the true visibilities...")
            # set_num_threads(5)
            start = tm.time()
            true_data = numba_dance.true_vis_numba(data, uvw_lmbdas, fluxes, ls, ms, ns)
            log.info("Adding thermal noise to model visibilities...")
            true_data[:, :, 0] = numba_dance.add_thermal_noise(true_data[:, :, 0], dnu)
            true_data[:, :, 3] = true_data[:, :, 0]
            log.info(
                "Time elapsed simulating model visibilities: %g", tm.time() - start
            )

            mtls.put_col(tbl, "DATA", true_data)

        if args.offset_vis:
            us, vs, _ = numba_dance.get_antenna_in_uvw(mset, tbl, lst)
            log.info("Calculating optimum phase screen size...")
            screensize = numba_dance.screen_size(
                args.height, args.radius, ra0, dec0, us, vs, MWAPOS, time,
            )

            log.info(
                f"The phase screen is a square, {round(screensize/1000, 1)} km per side."
            )
            set_num_threads(5)
            log.info("Simulating offset visibilities...")
            # logger.info("Simulating offset visibilities...")
            # "Get the phase screen"
            if args.tecpath is not None:
                loaded_tecpath = np.load(tecpath)
                phs_screen = loaded_tecpath["tecscreen"]
            else:
                phs_screen = make_phase_screen(
                    scale=args.scale, size=screensize, tec_type=args.tec_type
                )
                tecnpz = prefix + "_phase_screen.npz"
                np.savez(tecnpz, tecscreen=phs_screen)

            initial_setup_params = numba_dance.compute_initial_setup_params(
                phs_screen, us, vs, args.height, args.scale, ra0, dec0, time
            )
            ant1, ant2 = mtls.get_ant12(mset)
            initial_setup_params = list(initial_setup_params) + list(
                (ant1, ant2, args.spar)
            )
            initial_setup_params = tuple(initial_setup_params)
            # start = tm.time()
            # source_ppoints = numba_dance.collective_pierce_points(
            #     zen_angles, azimuths, initial_setup_params
            # )
            # log.info("collective_pierce_points elapsed: %g", tm.time() - start)
            # npz = mset.split(".")[0] + "_pierce_points.npz"
            # np.savez(npz, ppoints=source_ppoints)
            start = tm.time()
            offset_data = numba_dance.compute_offset_vis_parallel(
                data,
                initial_setup_params,
                zen_angles,
                azimuths,
                lmbdas,
                uvw_lmbdas,
                fluxes,
                ls,
                ms,
                ns,
            )
            log.info("Adding thermal noise to offset visibilities...")
            offset_data[:, :, 0] = numba_dance.add_thermal_noise(
                offset_data[:, :, 0], dnu
            )
            offset_data[:, :, 3] = offset_data[:, :, 0]
            log.info(
                "Time elapsed simulating offset visibilities: %g", tm.time() - start
            )

            if "OFFSET_DATA" not in tbl.colnames():
                log.info("Adding OFFSET_DATA column in MS with offset visibilities...")
                mtls.add_col(tbl, "OFFSET_DATA")
            mtls.put_col(tbl, "OFFSET_DATA", offset_data)
        # """
        # if args.scint_vis:
        #     scint_data = scint_vis(
        #         mset, data, uvw_lmbdas, fluxes, ls, ms, ns, args.rdiff
        #     )
        #     if "SCINT_DATA" not in tbl.colnames():
        #         log.info(
        #             "Adding SCINT_DATA column in MS with simulated visibilities... ..."
        #         )
        #         mtls.add_col(tbl, "SCINT_DATA")
        #     mtls.put_col(tbl, "SCINT_DATA", scint_data)
        # """
        tbl.close()

    if args.image:
        imagename = prefix + "_truevis"
        command = (
            "wsclean -name %s -abs-mem 40 -size 4096 4096 -scale 30asec -niter 1000000 -auto-threshold 3 \
                    -data-column %s %s"
            % (imagename, "DATA", mset)
        )
        try:
            os.system(command)
            wsclean_imager = True
        except Exception:
            wsclean_imager = False
            log.info("Unexpected wsclean error.")

        if args.offset_vis:
            if wsclean_imager:
                imagename = prefix + "_offsetvis"
                command2 = (
                    "wsclean -name %s -abs-mem 40 -size 4096 4096 -scale 30asec -niter 1000000 -auto-threshold 3 \
                            -data-column %s %s"
                    % (imagename, "OFFSET_DATA", mset)
                )
                os.system(command2)
        """
        if args.scint_vis:
            imagename = prefix + "_%srd_scint" % (int(args.rdiff / 1000))
            command3 = (
                "wsclean -name %s -abs-mem 40 -size 4096 4096 -scale 30asec -niter 1000000 -auto-threshold 3 \
                            -data-column %s %s"
                % (imagename, "SCINT_DATA", mset)
            )
            os.system(command3)
        """
        os.system("rm -r *dirty* *psf* *-residual* *-model*")

    if args.match:
        true_image = prefix + "_truevis-image.fits"
        offset_image = prefix + "_offsetvis-image.fits"
        sorted_df_true_sky, sorted_df_offset_sky = main_match(true_image, offset_image)

    if args.plot:
        if args.offset_vis:
            phs_screen = np.rad2deg(phs_screen)
        elif args.tecpath:
            phscrn = np.load(tecpath)
            phs_screen = np.rad2deg(phscrn["tecscreen"])
        else:
            try:
                phscrn_path = prefix + "_phase_screen.npz"
                phs_screen = np.rad2deg(np.load(phscrn_path)["tecscreen"])
            except FileNotFoundError:
                sys.exit(f"Phase screen not found at {phscrn_path}. Exiting.")
        npz = prefix + "_pierce_points.npz"
        tecdata = np.load(npz)
        ppoints = tecdata["ppoints"]
        # log.info(phs_screen.shape[0])
        # fieldcenter = (phs_screen.shape[0]) // 2
        # log.info(fieldcenter, phs_screen.shape)

        # params = np.rad2deg(tecdata["params"])
        # max_bl = mtls.get_bl_lens(mset).max()
        # plotting.ppoints_on_tec_field(
        #     phs_screen, ppoints, params, fieldcenter, prefix, max_bl, args.scale
        # )

        if args.match:
            try:
                pname = prefix + "_cthulhu_plots.png"
                obj = cthulhu_analyse(sorted_df_true_sky, sorted_df_offset_sky)
                plotting.cthulhu_plots(
                    obj, phs_screen, ppoints, args.scale, plotname=pname
                )
            except AssertionError:
                log.info("No matching sources in both catalogs.")

        elif args.match_done:
            try:
                pname = prefix + "_cthulhu_plots.png"
                sorted_true_sky_cat = "sorted_" + prefix + "_truevis-image.csv"
                sorted_offset_sky_cat = "sorted_" + prefix + "_offsetvis-image.csv"
                obj = cthulhu_analyse(sorted_true_sky_cat, sorted_offset_sky_cat)
                plotting.cthulhu_plots(
                    obj, phs_screen, ppoints, args.scale, plotname=pname
                )
            except FileNotFoundError:
                log.info("No catalog files found to match.")

    log.info("Wrapping up")


if __name__ == "__main__":
    main()
