import numpy as np
from astropy.io import fits
from mwa_pb import primary_beam
from coordinates import MWAPOS, get_time, radec_to_altaz


def compute_mwa_beam_attenuation(ras, decs, freq=150e6, pos=MWAPOS, metafits=None):
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
        assert_error_message = "A metafits file is needed"
        raise AssertionError(assert_error_message)
    # for a zenith pointing all delays are zero, and,
    # you need delays for both XX and YY so need to give it 2 sets of 16 delays
    # if zenith_pointing:
    #     delays = np.zeros((2, 16))
    else:
        with fits.open(metafits) as hdu:
            print("getting delays from metafits")
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


if __name__ == "__main__":
    metafits = "/home/kariuki/scint_sims/mset_data/1098108248.metafits"
    ras = np.array([50.0, 10])
    decs = np.array([-20.0, -32])
    fluxes = np.array([10, 20])
    mid_frequency = 150e6

    attens = compute_mwa_beam_attenuation(
        ras, decs, freq=mid_frequency, metafits=metafits
    )
    print(attens)
    afs = attens[0] * fluxes
    print(afs)
