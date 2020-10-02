from astropy.io.fits import getheader
from astropy.time import Time
import mwa_pb.config
import mwa_pb.primary_beam as pb
import numpy as np

from coordinates import radec_to_altaz


class MWABeam(object):
    def __init__(self, metafits):
        # Open metafits and extract beam delays
        metafits = getheader(metafits)
        self.time = Time(metafits["DATE-OBS"], location=mwa_pb.config.MWAPOS)
        delays = [int(d) for d in metafits["DELAYS"].split(",")]
        self.delays = [delays, delays]
        self.location = mwa_pb.config.MWAPOS

    def jones(self, ra, dec, freq):
        alt, az = radec_to_altaz(ra, dec, self.time, self.location)
        return pb.MWA_Tile_full_EE(
            np.pi / 2 - alt, az, freq, delays=self.delays, jones=True
        )


if __name__ == "__main__":
    metafits = "/home/kariuki/scint_sims/mset_data/1098108248.metafits"
    beam = MWABeam(metafits)
    print(beam.jones(0.0, -27.0, 150e6))
