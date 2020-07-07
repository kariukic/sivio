import os
import numpy as np
from casacore.tables import table

import vis_sim


def generate_distribution(mean, sigma, size, dist):
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
        return np.random.normal(loc=mean, scale=sigma, size=size)
    else:
        raise ValueError("Unrecognised distribution ({}).".format(dist))


N = 1000

ras = [0.0, 3.0]  # generate_distribution(0., 4., N, "normal")
decs = [-27.0, -25]  # generate_distribution(-27., 4., N, "normal")
fluxes = [1, 1]  # np.abs(generate_distribution(1., 3., N, "normal"))
rdiffs = [1000]  # diffractive scales.

metafits = "/home/kariuki/Downloads/1065880128.metafits"


def run_vis(
    mset, ras, decs, fluxes, rdiffs, clean_vis=False, offset=True, sc=False, im=True
):
    for rd in rdiffs:
        os.system("mkdir %s" % (mset))
        os.system("cp -r 2sources_truesimvis.ms/* %s" % (mset))

        tbl = table(mset, readonly=False)
        vis_sim.simulate_vis(
            mset,
            tbl,
            ras,
            decs,
            fluxes,
            rd,
            clean_vis=clean_vis,
            offset=offset,
            scintillate=sc,
        )
        tbl.close()

        if im:
            imagename = mset.split(".")[0] + "_truevis"
            datacol = "DATA"
            command = (
                "wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 25asec -niter 1000000 -auto-threshold 3 \
                        -data-column %s %s"
                % (imagename, datacol, mset)
            )
            os.system(command)

            if sc:
                imagename = mset.split(".")[0] + "_%srd_scint" % (int(rd / 1000))
                command2 = (
                    "wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 25asec -niter 1000000 -auto-threshold 3 \
                            -data-column %s %s"
                    % (imagename, "SCINT_DATA", mset)
                )
                os.system(command2)
            if offset:
                imagename = mset.split(".")[0] + "_offsetvis"
                command3 = (
                    "wsclean -name %s -abs-mem 40 -size 2048 2048 -scale 25asec -niter 1000000 -auto-threshold 3 \
                            -data-column %s %s"
                    % (imagename, "OFFSET_DATA", mset)
                )
                os.system(command3)

            os.system("rm *dirty* *psf* *residual* *model*")


if __name__ == "__main__":
    # mset = '%skm_rdiff_2sources_ionosimvis.ms'% (int(rd/1000))
    mset = "2sources_offsettest.ms"
    run_vis(
        mset, ras, decs, fluxes, rdiffs, clean_vis=True, offset=True, sc=False, im=True
    )
    os.system("rm -r %s" % (mset))
    print("Nimemaliza")
