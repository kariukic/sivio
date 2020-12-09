import os
import sys

import numpy as np


myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../sivio")
import phase_screen as ps
import vis_sim as vs
import coordinates as cds
import beam as bm

print(os.path.abspath("."))
metafits = "tests/1098108248.metafits"
pos = cds.MWAPOS
time, lst = cds.get_time(metafits, pos)
print(time, lst)
print(type(time), type(lst))

ra0, dec0 = np.deg2rad(-27.0), np.deg2rad(0.0)
ras = np.array([0, 20, 30], dtype=np.float64)
decs = np.array([-27.0, -45, -20], dtype=np.float64)

alt, az = cds.radec_to_altaz(np.deg2rad(ras), np.deg2rad(decs), time, pos)
za = np.pi / 2.0 - alt
fluxes = np.ones([len(za), 768, 4], dtype=np.complex64)
fluxes[:, :, 1:3] = 0
frequencies = np.linspace(100, 200, 768) * 1e6


height = 200000
scale = 100
h_pix = height / scale


# vis_sim.py tests
def test_scale_to_pixel_range():
    x = np.array([5, 10, 15])
    y = vs.scale_to_pixel_range(x, 1)
    z = np.array([0.0, 7.5, 15.0])
    assert np.array_equal(y, z)


def test_phase_center_offset():
    offset = vs.phase_center_offset(ra0, dec0, h_pix, time)
    assert offset == (-1110.0055823010644, 1009.8024980345327)


# phase_screen.py tests
def test_scale_to_pi_range():
    x = np.array([0, 5, 10])
    y = ps.scale_to_pi_range(x)
    z = np.array([0, np.pi / 2, np.pi])
    assert np.array_equal(y, z)


# beam.py tests
def test_mwapb_jones():
    apparent_mwapb = bm.mwapbeam(
        za, az, frequencies, fluxes, metafits=metafits, jones=True
    )
    assert (
        np.abs(apparent_mwapb[1, 0, :])
        == np.array([0.24002457, 0.0, 0.0, 0.22877702], dtype=np.float32)
    ).all()


fluxes = np.ones([len(za), 768, 4], dtype=np.complex64)
fluxes[:, :, 1:3] = 0


def test_mwapb_power():
    fluxes = np.ones([len(za), 768, 4], dtype=np.complex64)
    fluxes[:, :, 1:3] = 0
    apparent_mwapbxy = bm.mwapbeam(
        za, az, frequencies, fluxes, metafits=metafits, jones=False
    )
    assert (
        np.abs(apparent_mwapbxy[1, 0, :])
        == np.array([0.24002434, 0.0, 0.0, 0.22877672], dtype=np.float32)
    ).all()


def test_hyperbeam():
    fluxes = np.ones([len(za), 768, 4], dtype=np.complex64)
    fluxes[:, :, 1:3] = 0
    apparent_hyper = bm.hyperbeam(za, az, frequencies, fluxes, metafits=metafits)
    assert (
        np.abs(apparent_hyper[1, 0, :])
        == np.array([0.24002434, 0.0, 0.0, 0.22877672], dtype=np.float32)
    ).all()
