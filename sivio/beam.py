import numpy as np
from astropy.io import fits
from mwa_pb import primary_beam
from coordinates import MWAPOS, get_time, radec_to_altaz

import time

import mwa_hyperbeam


# # @njit
# def compute_mwa_beam_attenuation(
#     ras, decs, freq=150e6, pos=MWAPOS, metafits=None, zenith_pointing=False
# ):
#     """Compute the beam attenuation

#  Parameters
#   ----------
#    ras: float/array
#      source RA
#     decs: float/array
#      source dec
#     metafits: str
#      path to observation
#     pos: str
#      Array longitude and latitude
#     freq: float/int, optional
#      frequency, by default 150e6
#     zenith_pointing: bool, optional
#      True if observation is zenith pointing, by default True

#     Returns
#     -------
#     float/array, float/array
#      XX, YY beam attenuation for the input direction and frequency.
#     """
#     if not metafits:
#         assert_error_message = "A metafits file is needed for non-zenith observations"
#         raise AssertionError(assert_error_message)
#     # for a zenith pointing all delays are zero, and,
#     # you need delays for both XX and YY so need to give it 2 sets of 16 delays
#     elif zenith_pointing:
#         delays = np.zeros((2, 16))
#     else:
#         with fits.open(metafits) as hdu:
#             # print("getting delays from metafits")
#             delays = list(map(int, hdu[0].header["DELAYS"].split(",")))
#             delays = [delays, delays]
#     # print(f"delays: {delays}")

#     ttime, _ = get_time(metafits, pos)
#     alt, az = radec_to_altaz(np.deg2rad(ras), np.deg2rad(decs), ttime, pos)
#     za = np.pi / 2.0 - alt

#     # print(f"zenith angle: {np.radians(za)} azimuth: {np.radians(az)}")
#     XX, YY = primary_beam.MWA_Tile_full_EE(
#         za, az, freq=freq, delays=delays, zenithnorm=True, power=True, interp=False,
#     )
#     return XX, YY


def mwapbeam(za, az, frequencies, fluxes, metafits=None, jones=False):
    with fits.open(metafits) as hdu:
        delays = list(map(int, hdu[0].header["DELAYS"].split(",")))
        delays = [delays, delays]

    if jones:
        pjones = np.zeros((len(za), len(frequencies), 4), dtype=np.complex64)
        for chan in range(len(frequencies)):
            pjones[:, chan, :] = np.reshape(
                primary_beam.MWA_Tile_full_EE(
                    za, az, freq=frequencies[chan], delays=delays, jones=True
                ),
                (len(az), 4),
            )
        fluxes[:, :, 0] = (
            pjones[:, :, 0] * np.conj(pjones[:, :, 0]) * fluxes[:, :, 0]
            + pjones[:, :, 1] * np.conj(pjones[:, :, 1]) * fluxes[:, :, 0]
        )
        fluxes[:, :, 3] = (
            pjones[:, :, 2] * np.conj(pjones[:, :, 2]) * fluxes[:, :, 3]
            + pjones[:, :, 3] * np.conj(pjones[:, :, 3]) * fluxes[:, :, 3]
        )
    else:
        attenuations = np.zeros((len(za), len(frequencies), 4), dtype=np.float32)
        for chan in range(len(frequencies)):
            XX, YY = primary_beam.MWA_Tile_full_EE(
                za,
                az,
                freq=frequencies[chan],
                delays=delays,
                zenithnorm=True,
                power=True,
                interp=False,
            )

            attenuations[:, chan, 0] = XX
            attenuations[:, chan, 3] = YY
        fluxes *= attenuations
    return fluxes


def hyperbeam(za, az, frequencies, fluxes, metafits=None):
    hbeam = mwa_hyperbeam.FEEBeam(
        "/home/kariuki/mwa_pb/mwa_full_embedded_element_pattern.h5"
    )
    with fits.open(metafits) as hdu:
        delays = list(map(int, hdu[0].header["DELAYS"].split(",")))
    amps = [1] * 16
    beam_norm = True
    hjones = np.zeros((len(za), len(frequencies), 4), dtype=np.complex64)
    for chan in range(len(frequencies)):
        hjones[:, chan, :] = hbeam.calc_jones_array(
            az, za, int(frequencies[chan]), delays, amps, beam_norm
        )

    fluxes[:, :, 0] = (
        hjones[:, :, 0] * np.conj(hjones[:, :, 0]) * fluxes[:, :, 0]
        + hjones[:, :, 1] * np.conj(hjones[:, :, 1]) * fluxes[:, :, 0]
    )

    fluxes[:, :, 3] = (
        hjones[:, :, 2] * np.conj(hjones[:, :, 2]) * fluxes[:, :, 3]
        + hjones[:, :, 3] * np.conj(hjones[:, :, 3]) * fluxes[:, :, 3]
    )

    return fluxes


if __name__ == "__main__":
    metafits = "../mset_data/1098108248.metafits"
    ras = np.array([0, 20, 30], dtype=np.float64)
    decs = np.array([-27.0, -45, -20], dtype=np.float64)

    pos = MWAPOS
    ttime, _ = get_time(metafits, pos)
    alt, az = radec_to_altaz(np.deg2rad(ras), np.deg2rad(decs), ttime, pos)
    za = np.pi / 2.0 - alt

    fluxes = np.ones([len(za), 768, 4], dtype=np.complex64)
    fluxes[:, :, 1:3] = 0
    frequencies = np.linspace(100, 200, 768) * 1e6
    print("###############################################################")
    start_time = time.time()
    apparent_mwapb = mwapbeam(
        za, az, frequencies, fluxes, metafits=metafits, jones=True
    )
    mduration = time.time() - start_time
    print(
        f"mwapb jones took {mduration}s for {len(za)} pointings and {len(frequencies)} channels"
    )
    print(np.abs(apparent_mwapb[1, 0, :]))
    print("###############################################################")

    fluxes = np.ones([len(za), 768, 4], dtype=np.complex64)
    fluxes[:, :, 1:3] = 0
    start_time = time.time()
    apparent_mwapbxy = mwapbeam(
        za, az, frequencies, fluxes, metafits=metafits, jones=False
    )
    mjduration = time.time() - start_time
    print(
        f"mwapb xxyy took {mjduration}s to for {len(za)} pointings and {len(frequencies)} channels"
    )
    print(np.abs(apparent_mwapbxy[1, 0, :]))
    print("###############################################################")

    fluxes = np.ones([len(za), 768, 4], dtype=np.complex64)
    fluxes[:, :, 1:3] = 0
    start_time = time.time()
    apparent_hyper = hyperbeam(za, az, frequencies, fluxes, metafits=metafits)
    hduration = time.time() - start_time
    print(
        f"Hyper took {hduration}s to for {len(za)} pointings and {len(frequencies)} channels"
    )
    print(np.abs(apparent_hyper[1, 0, :]))
    print("###############################################################")
