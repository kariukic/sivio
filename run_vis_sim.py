import os
import logging
import numpy as np
from argparse import ArgumentParser
from casacore.tables import table

import mset_utils as mtls
from vis_sim import sim_prep, true_vis, offset_vis, scint_vis
import sky_models
from phase_screen import get_antenna_in_uvw, iono_phase_shift
from coordinates import get_time, MWAPOS
import plotting


def main():
    parser = ArgumentParser(
        "python run_vis_sim.py", description="ionospheric effects simulations"
    )
    parser.add_argument("--template_ms", required=True, help="Template measurement set")
    parser.add_argument("--ms_out", required=True, help="Output measurement set")
    parser.add_argument("--metafits", required=True, help="Path to the metafits file")
    parser.add_argument("--model", required=False, help="Sky model")
    parser.add_argument("--offset_vis", action="store_true")
    parser.add_argument("--scint_vis", action="store_true")
    parser.add_argument(
        "--rdiff", type=float, default=5000, help="Diffractive scale [m]"
    )
    parser.add_argument(
        "--size", type=int, default=150000, help="TEC field size per side [m]"
    )
    parser.add_argument(
        "--scale", type=int, default=10, help="pixel to distance scaling [m]"
    )
    parser.add_argument(
        "--height", type=int, default=200000, help="TEC plane height from ground [m]"
    )
    parser.add_argument("--kol", action="store_true", help="use kolmogorov tec")
    parser.add_argument("--true_vis", action="store_true")
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    # print(args, "args")

    logger = logging.getLogger(__name__)
    if args.debug:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    if not args.model:
        ras, decs, fluxes = sky_models.ras, sky_models.decs, sky_models.fluxes
        print("ras, decs, fluxes", ras, decs, fluxes)

    mset = args.ms_out
    if mset not in os.listdir(os.path.abspath(".")):
        os.system("mkdir %s" % (mset))
        os.system("cp -r %s/* %s" % (args.template_ms, mset))
    tbl = table(mset, readonly=False)
    data, uvw_lmbdas, ls, ms, ns = sim_prep(tbl, ras, decs)

    if args.true_vis:
        logger.info("Simulating the true visibilities...")
        true_data = true_vis(data, uvw_lmbdas, fluxes, ls, ms, ns)
        mtls.put_col(tbl, "DATA", true_data)

    if args.offset_vis:
        logger.info("Simulating offset visibilities...")
        "Get the phase screen"
        frequency = 150
        # TODO incorporate phase offset frequency dependence per channel
        phs_screen = iono_phase_shift(
            frequency, scale=args.scale, size=args.size, kolmogorov=args.kol
        )
        time, lst = get_time(args.metafits, MWAPOS)
        print("time", time, "lst", lst)
        us, vs, ws = get_antenna_in_uvw(mset, tbl, lst)
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
            args.height,
        )

        if "OFFSET_DATA" not in tbl.colnames():
            print("Adding OFFSET_DATA column in MS with offset visibilities...")
            mtls.add_col(tbl, "OFFSET_DATA")
        mtls.put_col(tbl, "OFFSET_DATA", offset_data)

    if args.scint_vis:
        scint_data = scint_vis(mset, data, uvw_lmbdas, fluxes, ls, ms, ns, args.rdiff)
        if "SCINT_DATA" not in tbl.colnames():
            print("Adding SCINT_DATA column in MS with simulated visibilities... ...")
            mtls.add_col(tbl, "SCINT_DATA")
        mtls.put_col(tbl, "SCINT_DATA", scint_data)
    tbl.close()

    if args.plot:
        tecdata = np.load("pierce_points.npz")
        ppoints = tecdata["ppoints"]
        params = tecdata["params"]
        tecscreen = tecdata["tecscreen"]
        fieldcenter = (args.size / args.scale) // 2
        plotting.ppoints_on_tec_field(tecscreen, ppoints, params, fieldcenter)

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
        # os.system("rm *dirty* *psf* *residual* *model*")


if __name__ == "__main__":
    main()
