import os
import logging
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from casacore.tables import table

import mset_utils as mtls
from vis_sim import sim_prep, true_vis, offset_vis  # , scint_vis
import sky_models
from phase_screen import get_antenna_in_uvw, iono_phase_shift, phase_center_offset
from coordinates import get_time, MWAPOS
import plotting
from match_catalogs import main_match
from cthulhu_analysis import cthulhu_analyse


def main():
    parser = ArgumentParser(
        "python run_vis_sim.py", description="Ionospheric effects simulations"
    )
    parser.add_argument("--ms_template", required=True, help="Template measurement set")
    parser.add_argument("--metafits", required=True, help="Path to the metafits file")
    parser.add_argument(
        "--yfile", required=False, help="Path to yaml file to pluck the sky model from"
    )
    parser.add_argument(
        "--tecpath", required=False, help="path to ready made tecscreen"
    )
    parser.add_argument(
        "--modelpath", required=False, help="path to ready made csv file sky model",
    )
    parser.add_argument(
        "--sim",
        required=False,
        action="store_true",
        help="Says you want to do the simulation. If not, you can choose to image or plot already simulated data",
    )
    parser.add_argument(
        "--n_sources", "-n", type=int, default=10, help="Number of sources to simulate"
    )
    parser.add_argument("--offset_vis", "-o", action="store_true")
    # parser.add_argument("--scint_vis", action="store_true")
    # parser.add_argument("--rdiff", type=float, default=5, help="Diffractive scale [m]")
    parser.add_argument(
        "--true_vis",
        "-t",
        action="store_true",
        help="Simulate the true visibilities too",
    )
    parser.add_argument(
        "--size", type=int, default=90000, help="TEC field size per side [m]"
    )
    parser.add_argument(
        "--scale", type=int, default=5, help="pixel to distance scaling [m]"
    )
    parser.add_argument(
        "--height", type=int, default=200000, help="TEC plane height from ground [m]"
    )
    parser.add_argument(
        "--tec_type",
        type=str,
        default="l",
        help="l = linear tec, s = TEC modulated with sine ducts,  k = TEC with kolmogorov turbulence.",
    )
    parser.add_argument("--match", "-m", action="store_true", help="match sources")
    parser.add_argument("--image", "-i", action="store_true", help="Run wsclean")
    parser.add_argument("--plot", "-p", action="store_true", help="Make plots")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    # print(args, "args")

    logger = logging.getLogger(__name__)
    if args.debug:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    obsid = args.ms_template.split(".")[0]
    mset = "sim%s_%s_%s_tec.ms" % (args.n_sources, obsid, args.tec_type,)
    prefix = mset.split(".")[0]
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
            sky_mod = pd.read_csv(args.modelpath)
            ras = sky_mod["ra"]
            decs = sky_mod["dec"]
            fluxes = sky_mod["flux"]
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

        data, lmbdas, uvw_lmbdas, ls, ms, ns = sim_prep(tbl, ras, decs)

        if args.true_vis:
            logger.info("Simulating the true visibilities...")
            true_data = true_vis(data, uvw_lmbdas, fluxes, ls, ms, ns)
            mtls.put_col(tbl, "DATA", true_data)

        if args.offset_vis:
            logger.info("Simulating offset visibilities...")
            "Get the phase screen"
            if args.tecpath is not None:
                loaded_tecpath = np.load(args.tecpath)
                phs_screen = loaded_tecpath["tecscreen"]
            else:
                phs_screen = iono_phase_shift(
                    scale=args.scale, size=args.size, tec_type=args.tec_type
                )
                tecnpz = prefix + "_phase_screen.npz"
                np.savez(tecnpz, tecscreen=phs_screen)
            time, lst = get_time(args.metafits, MWAPOS)

            us, vs, ws = get_antenna_in_uvw(mset, tbl, lst)
            h_pix = args.height / args.scale
            # first lets calculate the offset of the ref antenna from phase screen center.
            pp_u_offset, pp_v_offset = phase_center_offset(ra0, dec0, h_pix, time)
            # print("pp_u_offset,pp_v_offset", pp_u_offset, pp_v_offset)

            offset_data = offset_vis(
                mset,
                data,
                lmbdas,
                uvw_lmbdas,
                fluxes,
                ls,
                ms,
                ns,
                ras,
                decs,
                phs_screen,
                time,
                us,
                vs,
                args.scale,
                h_pix,
                pp_u_offset,
                pp_v_offset,
            )

            if "OFFSET_DATA" not in tbl.colnames():
                print("Adding OFFSET_DATA column in MS with offset visibilities...")
                mtls.add_col(tbl, "OFFSET_DATA")
            mtls.put_col(tbl, "OFFSET_DATA", offset_data)
        """
        if args.scint_vis:
            scint_data = scint_vis(
                mset, data, uvw_lmbdas, fluxes, ls, ms, ns, args.rdiff
            )
            if "SCINT_DATA" not in tbl.colnames():
                print(
                    "Adding SCINT_DATA column in MS with simulated visibilities... ..."
                )
                mtls.add_col(tbl, "SCINT_DATA")
            mtls.put_col(tbl, "SCINT_DATA", scint_data)
        """
        tbl.close()

    if args.image:
        imagename = prefix + "_truevis"
        command = (
            "wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 30asec -niter 1000000 -auto-threshold 3 \
                    -data-column %s %s"
            % (imagename, "DATA", mset)
        )
        os.system(command)

        if args.offset_vis:
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
        npz = prefix + "_pierce_points.npz"
        tecdata = np.load(npz)
        ppoints = tecdata["ppoints"]
        params = np.rad2deg(tecdata["params"])
        # tecscreen = np.rad2deg(tecdata["tecscreen"])
        fieldcenter = (args.size / args.scale) // 2
        print(fieldcenter, phs_screen.shape)

        max_bl = mtls.get_bl_lens(mset).max()

        plotting.ppoints_on_tec_field(
            phs_screen, ppoints, params, fieldcenter, prefix, max_bl, args.scale
        )
        if args.match:
            pname = prefix + "_cthulhu_plots.png"
            obj = cthulhu_analyse(sorted_df_true_sky, sorted_df_offset_sky)
            plotting.cthulhu_plots(
                obj, phs_screen, ppoints, fieldcenter, plotname=pname
            )

    sim_dir = "output_" + prefix
    os.system("mkdir %s" % (sim_dir))
    os.system("mv sim%s* sorted_sim%s* %s" % (args.n_sources, args.n_sources, sim_dir))


if __name__ == "__main__":
    main()
