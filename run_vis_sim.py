import os
import logging
import numpy as np
from argparse import ArgumentParser
from casacore.tables import table

import mset_utils as mtls
from vis_sim import sim_prep, true_vis, offset_vis, scint_vis
import sky_models
from phase_screen import get_antenna_in_uvw, iono_phase_shift, phase_center_offset
from coordinates import get_time, MWAPOS
import plotting


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
        "--sim",
        required=False,
        action="store_true",
        help="Says you want to do the simulation. If not, you can choose to image or plot already simulated data",
    )
    parser.add_argument("--offset_vis", "-o", action="store_true")
    parser.add_argument("--scint_vis", action="store_true")
    parser.add_argument("--rdiff", type=float, default=5, help="Diffractive scale [m]")
    parser.add_argument(
        "--size", type=int, default=60000, help="TEC field size per side [m]"
    )
    parser.add_argument(
        "--scale", type=int, default=3, help="pixel to distance scaling [m]"
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
    parser.add_argument(
        "--true_vis",
        "-t",
        action="store_true",
        help="Simulate the true visibilities too",
    )
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

    mset = "sim_1065880128_klmgv6060_tec.ms"
    if args.sim:
        if args.yfile is not None:
            ras, decs, fluxes = sky_models.loadfile(args.yfile, model_only=True)
        else:
            ras, decs, fluxes = sky_models.random_model(
                1, simple=True, filename="simple_sky.txt"
            )
        assert len(ras) == len(decs) == len(fluxes)
        ras = np.radians(ras)
        decs = np.radians(decs)
        print("The sky model has %s sources" % (len(ras)))

        if mset not in os.listdir(os.path.abspath(".")):
            print("Making the simulation measurement set..")
            os.system("mkdir %s" % (mset))
            os.system("cp -r %s/* %s" % (args.ms_template, mset))

        tbl = table(mset, readonly=False)
        ra0, dec0 = mtls.get_phase_center(tbl)
        data, uvw_lmbdas, ls, ms, ns = sim_prep(tbl, ras, decs)

        if args.true_vis:
            logger.info("Simulating the true visibilities...")
            true_data = true_vis(data, uvw_lmbdas, fluxes, ls, ms, ns)
            mtls.put_col(tbl, "DATA", true_data)

        if args.offset_vis:
            logger.info("Simulating offset visibilities...")
            "Get the phase screen"
            # TODO incorporate phase offset frequency dependence per channel
            phs_screen = iono_phase_shift(
                scale=args.scale, size=args.size, tec_type=args.tec_type
            )
            time, lst = get_time(args.metafits, MWAPOS)

            us, vs, ws = get_antenna_in_uvw(mset, tbl, lst)
            h_pix = args.height / args.scale
            # first lets calculate the offset of the ref antenna from phase screen center.
            pp_u_offset, pp_v_offset = phase_center_offset(ra0, dec0, h_pix, time)
            print("pp_u_offset,pp_v_offset", pp_u_offset, pp_v_offset)

            offset_data = offset_vis(
                mset,
                data,
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
        tbl.close()

    if args.plot:
        npz = mset.split(".")[0] + "_pierce_points.npz"
        tecdata = np.load(npz)
        ppoints = tecdata["ppoints"]
        params = np.rad2deg(tecdata["params"])
        tecscreen = np.rad2deg(tecdata["tecscreen"])
        fieldcenter = (args.size / args.scale) // 2

        max_bl = mtls.get_bl_lens(mset).max()
        prefix = mset.split(".")[0]
        plotting.ppoints_on_tec_field(
            tecscreen, ppoints, params, fieldcenter, prefix, max_bl, args.scale
        )

    if args.image:
        imagename = mset.split(".")[0] + "_truevis"
        command = (
            "wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 25asec -niter 1000000 -auto-threshold 3 \
                    -data-column %s %s"
            % (imagename, "DATA", mset)
        )
        os.system(command)

        if args.offset_vis:
            imagename = mset.split(".")[0] + "_offsetvis"
            command2 = (
                "wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 25asec -niter 1000000 -auto-threshold 3 \
                        -data-column %s %s"
                % (imagename, "OFFSET_DATA", mset)
            )
            os.system(command2)

        if args.scint_vis:
            imagename = mset.split(".")[0] + "_%srd_scint" % (int(args.rdiff / 1000))
            command3 = (
                "wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 25asec -niter 1000000 -auto-threshold 3 \
                            -data-column %s %s"
                % (imagename, "SCINT_DATA", mset)
            )
            os.system(command3)
        os.system("rm -r *dirty* *psf* *-residual* *-model*")


if __name__ == "__main__":
    main()
