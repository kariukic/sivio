import matplotlib as mpl
import matplotlib.pyplot as plt
from mset_utils import get_uvw
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cthulhu.plot_tools import setup_subplot, plot_vector_arrows
import numpy as np

f = mpl.rcParams["font.size"] + 4


def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def plotuv(tbl):
    """Take a look at the uv plane"""
    ff = mpl.rcParams["font.size"] + 6
    uvw = get_uvw(tbl)
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.plot(uvw[:, 0], uvw[:, 1], "k.")
    # uncomment to plot the complex conjugates
    # plt.plot(-1.*uvw[:,0], -1.*uvw[:,1], 'b.')
    plt.gcf().subplots_adjust(bottom=0.18, left=0.2)
    plt.xlabel("u (m)", fontsize=ff)
    plt.ylabel("v (m)", fontsize=ff)
    plt.savefig("uuvv.png")


def plot_antennas_on_uvplane(us, vs, name="antennas_on_uvplane.png"):
    plt.scatter(
        us, vs, c="b", marker="x", linestyle="None", label="True antenna positions"
    )
    plt.plot(us[0], vs[0], "gx", label="Reference antenna")
    plt.ylabel("V distance (m)")
    plt.xlabel("U distance (m)")
    plt.legend()
    plt.title("Antenna positions projected onto the UV plane")
    plt.savefig("%s" % (name))


def plot_antennas_on_tec_field(tec, u_tec_list, v_tec_list):
    # fig = plt.figure(figsize=(7, 7))
    xmin, xmax = min(u_tec_list), max(u_tec_list)
    ymin, ymax = min(v_tec_list), max(v_tec_list)
    s = plt.imshow(tec, cmap="plasma", extent=[xmin, xmax, ymin, ymax])
    # plt.plot(u_tec_list[0], v_tec_list[0], "rx", label="TEC field center")
    plt.scatter(
        u_tec_list, v_tec_list, c="c", marker="o", s=2, label="antenna positions"
    )
    colorbar(s)
    plt.legend()
    plt.xlabel("Relative Longitude (m*5)")
    plt.ylabel("Relative Latitude (m*5)")
    plt.savefig("zenith_test_source_antennas_on_linear_tec.png")


def ppoints_on_tec_field(tec, ppoints, params, prefix, max_bl, scale):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    fieldcenter = tec.shape[0] // 2

    # fig = plt.figure(figsize=(7, 7))
    # yy1 = int(fieldcenter // 2)
    # yy2 = int(fieldcenter + yy1)
    # xx1 = 0
    # xx2 = fieldcenter

    xnymin = fieldcenter - tec.shape[0] // 2
    xnymax = fieldcenter + tec.shape[0] // 2
    # print(xnymin, xnymax)
    extent = (xnymin, xnymax, xnymin, xnymax)
    s = ax1.imshow(np.rot90(tec), cmap="plasma", origin="upper", extent=extent)
    # s = ax1.imshow(tec[xx1:xx2, yy1:yy2], cmap="plasma", extent=[xx1, xx2, yy1, yy2])
    # ax1.plot(fieldcenter, fieldcenter, "rx", label="screen center")
    """
    axins = ax1.inset_axes([0.6, 0.75, 0.25, 0.25])

    axins.imshow(tec, cmap="plasma", origin="lower")
    # just trying to get a proper size for the inset plot.
    w = 600  # max_bl / 2 * np.sin(np.radians(45)) / scale
    x1, x2, y1, y2 = (
        ppoints[0, 0, 0] - w,
        ppoints[0, 0, 0] + w,
        ppoints[0, 1, 0] - w,
        ppoints[0, 1, 0] + w,
    )
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.scatter(ppoints[0, 0, :], ppoints[0, 1, :], marker="o", s=1, c="c")
    axins.tick_params(labelbottom=False, labelleft=False)
    ax1.indicate_inset_zoom(axins)
    """
    count = 1
    for source in range(ppoints.shape[1]):
        # print("SOURCE %s ORIGIN POINT" % (count), uvlist[0, 0], uvlist[1, 0])
        ax1.scatter(
            ppoints[0, source, :],
            ppoints[1, source, :],
            c="c",
            marker="o",
            s=1,
            label="antenna positions",
        )
        count += 1
    cbar = colorbar(s)
    cbar.ax.set_ylabel("phase [deg]", rotation=270)

    ax1.set_xlabel("Relative Longitude (scale=1:%sm)" % (scale))
    ax1.set_ylabel("Relative Latitude (scale=1:%sm)" % (scale))

    # Plot phase offsets per antenna for just 10 sources to avoid conjestion
    print(params.shape, "paras shape")
    if len(params[0]) > 10:
        params0 = params[0, 0:10, :]
        params1 = params[1, 0:10, :]
    else:
        params0 = params[0, :, :]
        params1 = params[1, :, :]
    for params_list in params0:
        ax2.plot(range(len(params_list)), params_list, marker="*", linestyle="--")
    for params_list in params1:
        ax3.plot(range(len(params_list)), params_list, marker="*", linestyle="--")
    # ax2.set_xlabel("Antenna ID")
    ax2.set_ylabel("x phase [deg]")
    ax3.set_xlabel("Antenna ID")
    ax3.set_ylabel("y phase [deg]")

    fig.tight_layout()
    plt.savefig("%s_antenna_ppoints_on_tec.png" % (prefix))
    plt.close()


def ppoints_on_tec_field_v2(tec, ppoints, scale):
    xnymin = 0
    xnymax = tec.shape[0]
    extent = (xnymin, xnymax, xnymin, xnymax)
    s = plt.imshow(tec, cmap="plasma", extent=extent)

    for source in range(ppoints.shape[1]):
        plt.scatter(
            ppoints[0, source, :],
            ppoints[1, source, :],
            c="c",
            marker="o",
            s=1,
            label="antenna positions",
        )

    cbar = colorbar(s)
    cbar.ax.set_ylabel("phase [deg]", rotation=270, fontsize=f)
    # ax1.legend()

    plt.xlabel("Relative Longitude (scale=1:%sm)" % (scale), fontsize=f)
    plt.ylabel("Relative Latitude (scale=1:%sm)" % (scale), fontsize=f)
    plt.title("Pierce points coverage", fontsize=f)


def cthulhu_plots(o, tecscreen, ppoints, scale=60, plotname="tec_reconstruction.png"):
    # extent = [-9, 9, -35, -18]

    fig = plt.figure(figsize=(12, 9))
    plt.subplots_adjust(hspace=0.25, left=0.05, right=0.95, top=0.95, bottom=0.07)
    ax1 = fig.add_subplot(224)
    j = ax1.imshow(
        np.flipud(o.tec), cmap="plasma", vmin=0, vmax=0.5
    )  # , extent=extent)  # ,vmin=0,vmax=1)
    cbar4 = colorbar(j)
    cbar4.ax.set_ylabel("TEC [TECU]", rotation=270)
    ax1.set_title("Reconstructed TEC field", fontsize=f)

    ax2 = fig.add_subplot(221)
    k = ax2.imshow(
        np.rot90(tecscreen), cmap="plasma"
    )  # , extent=extent)  # ,vmin=0, vmax=1)
    cbar1 = colorbar(k)
    cbar1.ax.set_ylabel("phase [deg]", rotation=270, fontsize=f)
    ax2.set_title("Full phase screen", fontsize=f)

    fig.add_subplot(222)
    ppoints_on_tec_field_v2(np.rot90(tecscreen), ppoints, scale)

    ax4 = fig.add_subplot(223)
    setup_subplot(axis=ax4)
    plot_vector_arrows(axis=ax4, obsid=o, scale=60)
    plot_title = f"""Median offsets: {round(o.metrics[0][0], 4)},
    PCA eigenvalue: {round(o.metrics[1][0], 4)} \n Metric: {round(o.metric, 4)}"""
    ax4.set_title(plot_title, fontsize=f)
    ax4.set_xlabel("RA [deg]", fontsize=f)
    ax4.set_ylabel("Dec [deg]", fontsize=f)

    fig.tight_layout()
    plt.savefig(plotname)
    plt.show()
