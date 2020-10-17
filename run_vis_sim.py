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

import mset_utils as mtls
import plotting
import sky_models
from coordinates import MWAPOS, get_time, radec_to_altaz
from cthulhu_analysis import cthulhu_analyse
from match_catalogs import main_match
from phase_screen import make_phase_screen
from vis_sim_numba import (
    add_thermal_noise,
    collective_pierce_points,
    compute_initial_setup_params,
    compute_offset_vis_parallel,
    get_antenna_in_uvw,
    sim_prep,
    true_vis_numba,
)

print(
    f"No. of CPUS: {multiprocessing.cpu_count()}: Total memory: ~{psutil.virtual_memory().total/1e9} GB"
)


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
        help="States you want to run the simulation. If not, \
            you can just opt to image or plot already simulated data",
    )
    parser.add_argument(
        "--n_sources",
        "-n",
        type=int,
        default=10,
        help="Number of point sources to simulate",
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
        help="Simulate the true (un-corrupted) visibilities too",
    )
    parser.add_argument(
        "--size", type=int, default=80000, help="TEC field size per side [m]"
    )
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
        help="States you already have matched catalogues, go straight to cthulhu plotting",
    )
    parser.add_argument(
        "--image",
        "-i",
        action="store_true",
        help="Run wsclean. Default settings are: \
            '-abs-mem 40 -size 2048 2048 -scale 30asec -niter 1000000 -auto-threshold 3'",
    )
    parser.add_argument("--plot", "-p", action="store_true", help="Make plots")
    parser.add_argument(
        "--mp",
        action="store_true",
        help="run multiprocessing. not yet fully implemented",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    if args.debug:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    if "/" in args.ms_template:
        obsid = args.ms_template.split("/")[-1].split(".")[0]
    else:
        obsid = args.ms_template.split(".")[0]
    print("obsid:", obsid)
    mset = "%s_sources_%s_%stec.ms" % (args.n_sources, obsid, args.tec_type,)
    prefix = mset.split(".")[0]
    output_dir = "simulation_output/" + prefix + "_spar" + str(args.spar)
    print(output_dir)
    if os.path.exists(output_dir):
        output_dir += "_run2"
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    if args.sim:
        if mset not in os.listdir(os.path.abspath(".")):
            print("Making the simulation measurement set..")
            os.system("mkdir %s" % (mset))
            os.system("cp -r %s/* %s" % (args.ms_template, mset))

        tbl = table(mset, readonly=False)
        ra0, dec0 = mtls.get_phase_center(tbl)
        print(
            "The phase center is at ra=%s, dec=%s" % (np.degrees(ra0), np.degrees(dec0))
        )
        if args.modelpath is not None:
            ras, decs, fluxes = [], [], []
            with open(args.modelpath, "r") as file:
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
                ras, decs, fluxes = sky_models.random_model(
                    args.n_sources,
                    np.rad2deg(ra0),
                    np.rad2deg(dec0),
                    filename=model_textfile,
                )

        ras = np.radians(ras)
        decs = np.radians(decs)
        assert len(ras) == len(decs) == len(fluxes)
        print("The sky model has %s sources" % (len(ras)))

        data, lmbdas, uvw_lmbdas, dnu, ls, ms, ns = sim_prep(tbl, ras, decs)

        if args.true_vis:
            # logger.info("Simulating the true visibilities...")
            # set_num_threads(5)
            start = tm.time()
            true_data = true_vis_numba(data, uvw_lmbdas, fluxes, ls, ms, ns)
            print("Adding thermal noise to model visibilities...")
            true_data[:, :, 0] = add_thermal_noise(true_data[:, :, 0], dnu)
            true_data[:, :, 3] = true_data[:, :, 0]
            print("Time elapsed simulating model visibilities: %g", tm.time() - start)

            mtls.put_col(tbl, "DATA", true_data)

        if args.offset_vis:
            print("Simulating offset visibilities...")
            # logger.info("Simulating offset visibilities...")
            # "Get the phase screen"
            if args.tecpath is not None:
                loaded_tecpath = np.load(args.tecpath)
                phs_screen = loaded_tecpath["tecscreen"]
            else:
                phs_screen = make_phase_screen(
                    scale=args.scale, size=args.size, tec_type=args.tec_type
                )
                tecnpz = prefix + "_phase_screen.npz"
                np.savez(tecnpz, tecscreen=phs_screen)

            time, lst = get_time(args.metafits, MWAPOS)
            alts, azimuths = radec_to_altaz(ras, decs, time, MWAPOS)
            zen_angles = np.pi / 2.0 - alts

            us, vs, _ = get_antenna_in_uvw(mset, tbl, lst)
            ant1, ant2 = mtls.get_ant12(mset)

            initial_setup_params = compute_initial_setup_params(
                phs_screen, us, vs, args.height, args.scale, ra0, dec0, time
            )
            initial_setup_params = list(initial_setup_params) + list(
                (ant1, ant2, args.spar)
            )
            initial_setup_params = tuple(initial_setup_params)
            start = tm.time()
            source_ppoints = collective_pierce_points(
                zen_angles, azimuths, initial_setup_params
            )
            print("collective_pierce_points elapsed: %g", tm.time() - start)
            npz = mset.split(".")[0] + "_pierce_points.npz"
            np.savez(npz, ppoints=source_ppoints)
            start = tm.time()
            offset_data = compute_offset_vis_parallel(
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
            print("Adding thermal noise to offset visibilities...")
            offset_data[:, :, 0] = add_thermal_noise(offset_data[:, :, 0], dnu)
            offset_data[:, :, 3] = offset_data[:, :, 0]
            print("Time elapsed simulating offset visibilities: %g", tm.time() - start)

            if "OFFSET_DATA" not in tbl.colnames():
                print("Adding OFFSET_DATA column in MS with offset visibilities...")
                mtls.add_col(tbl, "OFFSET_DATA")
            mtls.put_col(tbl, "OFFSET_DATA", offset_data)
        # """
        # if args.scint_vis:
        #     scint_data = scint_vis(
        #         mset, data, uvw_lmbdas, fluxes, ls, ms, ns, args.rdiff
        #     )
        #     if "SCINT_DATA" not in tbl.colnames():
        #         print(
        #             "Adding SCINT_DATA column in MS with simulated visibilities... ..."
        #         )
        #         mtls.add_col(tbl, "SCINT_DATA")
        #     mtls.put_col(tbl, "SCINT_DATA", scint_data)
        # """
        tbl.close()

    if args.image:
        imagename = prefix + "_truevis"
        command = (
            "wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 30asec -niter 1000000 -auto-threshold 3 \
                    -data-column %s %s"
            % (imagename, "DATA", mset)
        )
        try:
            os.system(command)
            wsclean_imager = True
        except Exception:
            wsclean_imager = False
            print("Unexpected wsclean error.")

        if args.offset_vis:
            if wsclean_imager:
                imagename = prefix + "_offsetvis"
                command2 = (
                    "wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 30asec -niter 1000000 -auto-threshold 3 \
                            -data-column %s %s"
                    % (imagename, "OFFSET_DATA", mset)
                )
                os.system(command2)
        """
        if args.scint_vis:
            imagename = prefix + "_%srd_scint" % (int(args.rdiff / 1000))
            command3 = (
                "wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 30asec -niter 1000000 -auto-threshold 3 \
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
            phscrn = np.load(args.tecpath)
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
        fieldcenter = (args.size / args.scale) // 2
        print(fieldcenter, phs_screen.shape)

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
                    obj, phs_screen, ppoints, fieldcenter, args.scale, plotname=pname
                )
            except AssertionError:
                print("No matching sources in both catalogs.")

        elif args.match_done:
            try:
                pname = prefix + "_cthulhu_plots.png"
                sorted_true_sky_cat = "sorted_" + prefix + "_truevis-image.csv"
                sorted_offset_sky_cat = "sorted_" + prefix + "_offsetvis-image.csv"
                obj = cthulhu_analyse(sorted_true_sky_cat, sorted_offset_sky_cat)
                plotting.cthulhu_plots(
                    obj, phs_screen, ppoints, fieldcenter, args.scale, plotname=pname
                )
            except FileNotFoundError:
                print("No catalog files found to match.")

    print("Wrapping up")

    # os.system("mv %s* sorted_%s* %s" % (args.n_sources, args.n_sources, output_dir))


if __name__ == "__main__":
    main()
