import numpy as np
from casacore.tables import table

from phase_screen import (
    get_antenna_in_uvw,
    get_tec_value,
    iono_phase_shift,
    phase_center_offset,
    scale_to_pi_range,
    scale_to_pixel_range,
)
from coordinates import MWAPOS, get_time

size = 2500
height = 200000
scale = 10
h_pix = height / scale

metafits = "../Downloads/1065880128.metafits"


time, lst = get_time(metafits, MWAPOS)


constant = 40.30819
tecu = 1e16

mset = "1065880128_template.ms"
tbl = table(mset, readonly=True)

ras = np.radians([0.0])
decs = np.radians([-27.0])


def test_scale_to_pi_range():
    x = np.array([0, 5, 10])
    y = scale_to_pi_range(x)
    z = np.array([-np.pi, 0, np.pi])
    assert np.array_equal(y, z)


def test_scale_to_pixel_range():
    x = np.array([5, 10, 15])
    y = scale_to_pixel_range(x, 1)
    z = np.array([0.0, 7.5, 15.0])
    assert np.array_equal(y, z)


def test_iono_phase_shift():
    screen = iono_phase_shift(scale=1, size=size, tec_type="k")
    assert screen.shape == (size, size)
    # assert screen.max() == np.pi
    # assert screen.min() == -np.pi


us, vs, ws = get_antenna_in_uvw(mset, tbl, lst)


def test_phase_center_offset():
    ra0 = ras[0]
    dec0 = decs[0]
    pp_u_offset, pp_v_offset = phase_center_offset(ra0, dec0, h_pix, time)
    assert pp_u_offset == 3812.4612433581997
    assert pp_v_offset == -259.34713550442814


def test_get_tec_value():
    zen_angle = 0.0
    azimuth = 0.0
    phs_screen = iono_phase_shift(scale=scale, size=size, tec_type="l")
    u_tec_list, v_tec_list, tec_per_ant = get_tec_value(
        phs_screen, us, vs, zen_angle, azimuth, scale, h_pix, 0, 0,
    )
    assert v_tec_list[0] == u_tec_list[0] == (size / scale) / 2
