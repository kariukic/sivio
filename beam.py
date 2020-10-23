import numpy as np
from astropy.io import fits
from mwa_pb import primary_beam
from coordinates import MWAPOS, get_time, radec_to_altaz


# @njit
def compute_mwa_beam_attenuation(
    ras, decs, freq=150e6, pos=MWAPOS, metafits=None, zenith_pointing=False
):
    """Compute the beam attenuation

 Parameters
  ----------
   ras: float/array
     source RA
    decs: float/array
     source dec
    metafits: str
     path to observation
    pos: str
     Array longitude and latitude
    freq: float/int, optional
     frequency, by default 150e6
    zenith_pointing: bool, optional
     True if observation is zenith pointing, by default True

    Returns
    -------
    float/array, float/array
     XX, YY beam attenuation for the input direction and frequency.
    """
    if not metafits:
        assert_error_message = "A metafits file is needed for non-zenith observations"
        raise AssertionError(assert_error_message)
    # for a zenith pointing all delays are zero, and,
    # you need delays for both XX and YY so need to give it 2 sets of 16 delays
    elif zenith_pointing:
        delays = np.zeros((2, 16))
    else:
        with fits.open(metafits) as hdu:
            # print("getting delays from metafits")
            delays = list(map(int, hdu[0].header["DELAYS"].split(",")))
            delays = [delays, delays]
    # print(f"delays: {delays}")

    time, _ = get_time(metafits, pos)
    alt, az = radec_to_altaz(np.deg2rad(ras), np.deg2rad(decs), time, pos)
    za = np.pi / 2.0 - alt

    # print(f"zenith angle: {np.radians(za)} azimuth: {np.radians(az)}")
    XX, YY = primary_beam.MWA_Tile_full_EE(
        za, az, freq=freq, delays=delays, zenithnorm=True, power=True, interp=False,
    )
    return XX, YY


def compute_attenuations(za, az, frequencies, metafits=None):
    with fits.open(metafits) as hdu:
        delays = list(map(int, hdu[0].header["DELAYS"].split(",")))
        delays = [delays, delays]
    attenuations = np.zeros((len(za), len(frequencies), 4), dtype=np.float32)
    for freq_chan in range(len(frequencies)):
        XX, YY = primary_beam.MWA_Tile_full_EE(
            za,
            az,
            freq=frequencies[freq_chan],
            delays=delays,
            zenithnorm=True,
            power=True,
            interp=False,
        )
        # XX, YY = compute_mwa_beam_attenuation(
        #     ras, decs, freq=frequencies[freq_chan], metafits=metafits
        # )
        attenuations[:, freq_chan, 0] = XX
        attenuations[:, freq_chan, 3] = YY

    return attenuations


if __name__ == "__main__":
    metafits = "mset_data/1098108248.metafits"
    za = np.array([0.0, 10], dtype=np.float64)
    az = np.array([-27.0, -32], dtype=np.float64)
    fluxes = np.array([10, 20], dtype=np.float64)
    frequencies = np.linspace(100, 200, 100) * 1e6

    attens = compute_attenuations(za, az, frequencies, metafits=metafits)
    print(np.array(attens).shape)
    # fluxes *= attens[0]
    # print(fluxes)
