import matplotlib.pyplot as plt
from mset_utils import get_uvw


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt

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
    uvw = get_uvw(tbl)
    plt.figure
    plt.plot(uvw[:, 0], uvw[:, 1], "k.")
    # uncomment to plot the complex conjugates
    # plt.plot(-1.*uvw[:,0], -1.*uvw[:,1], 'b.')
    plt.xlabel("u (m)")
    plt.ylabel("v (m)")
    plt.savefig("uuvv.png")


def plot_antennas_on_uvplane(
    us, vs, shift=True, alpha=0.1, name="antennas_on_uvplane.png"
):
    if shift:
        uss = us + alpha * (us[:] - us[0])
        vss = vs + alpha * (vs[:] - vs[0])
        plt.scatter(
            uss,
            vss,
            c="r",
            marker="x",
            linestyle="None",
            label="shifted antenna positions, alpha=%s" % (alpha),
        )
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
    plt.plot(u_tec_list[0], v_tec_list[0], "rx", label="TEC field center")
    plt.scatter(
        u_tec_list, v_tec_list, c="c", marker="o", s=2, label="antenna positions"
    )
    colorbar(s)
    plt.legend()
    plt.xlabel("Relative Longitude (m*5)")
    plt.ylabel("Relative Latitude (m*5)")
    plt.savefig("zenith_test_source_antennas_on_linear_tec.png")
