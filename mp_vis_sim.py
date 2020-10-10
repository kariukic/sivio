import numpy as np
from scipy import constants
from phase_screen import (
    phase_center_offset,
    scale_to_pixel_range,
)

# import time
# import os

# import multiprocessing as mp
# import ctypes
from numba import njit, float64, complex64, prange
import ray
import mset_utils as mtls
import psutil

phys_cores = psutil.cpu_count(logical=False)
logical_cpus = psutil.cpu_count(logical=True)
print(f"No. of cores: {phys_cores}")
print(f"No. of logical CPUS: {logical_cpus}")
mem = psutil.virtual_memory()
print(f"Total memory in bytes: {mem.total}")


# ray.shutdown()
# ray.init(num_cpus=logical_cpus)

# from casacore.tables import table
# ctypes.c


def sim_prep(tbl, ras, decs):
    """Preparing the requiremnts before simulating"""
    # grab uvw values from the ms and divide them with the wavelengths for each channel
    uvw = mtls.get_uvw(tbl)
    lmbdas = mtls.get_channels(tbl, ls=True)
    frequencies = mtls.get_channels(tbl, ls=False)
    dnu = frequencies[1] - frequencies[0]
    uvw_lmbdas = uvw[:, None, :] / lmbdas[None, :, None]

    # Now lets grab the data from the ms table
    data = mtls.get_data(tbl)
    print("Shape of the MS data", data.shape)

    # Delete all the grabbed data because we will put the simulated data there
    data[:] = 0

    # grab our lmns
    ls, ms, ns = mtls.get_lmns(tbl, ras, decs, phase_center_shift=0)
    # assert len(list(A)) == len(list(ls))
    print("dnu", dnu)
    return data, lmbdas, uvw_lmbdas, dnu, ls, ms, ns


@njit(
    [
        complex64[:, :, :](
            complex64[:, :, :],
            float64[:, :, :],
            float64[:],
            float64[:],
            float64[:],
            float64[:],
        )
    ],
    parallel=True,
)
def true_vis_numba(data, uvw_lmbdas, fluxes, ls, ms, ns):
    data[:] = 0
    u = uvw_lmbdas[:, :, 0]
    v = uvw_lmbdas[:, :, 1]
    w = uvw_lmbdas[:, :, 2]
    source_count = 1
    for source in prange(len(fluxes)):
        print("True Source: ", source_count, "...")
        source_visibilities = fluxes[source] * np.exp(
            2j * np.pi * (u * ls[source] + v * ms[source] + w * ns[source])
        )
        data[:, :, 0] += source_visibilities  # feed xx data
        data[:, :, 3] += source_visibilities  # feed yy data
        source_count += 1

    return data


@njit
def single_source_pierce_point(
    du,
    dv,
    source_zenith_angle,
    source_azimuth_angle,
    scaled_u_axis_ants_pos,
    scaled_v_axis_ants_pos,
    h_pix,
    u_shift_factor,
    v_shift_factor,
    refant=0,
):
    ants_u_pierce_points, ants_v_pierce_points, ants_u_gradients, ants_v_gradients = (
        [np.float64(x) for x in range(0)],
        [np.float64(x) for x in range(0)],
        [np.float64(x) for x in range(0)],
        [np.float64(x) for x in range(0)],
    )
    for scaled_u_ant_pos, scaled_v_ant_pos in zip(
        scaled_u_axis_ants_pos, scaled_v_axis_ants_pos
    ):
        # For each antenna, the antenna position becomes a new origin
        # This antenna position first has to be in reference to the refant.
        new_u0 = scaled_u_ant_pos - scaled_u_axis_ants_pos[refant]
        new_v0 = scaled_v_ant_pos - scaled_v_axis_ants_pos[refant]

        # For each antenna, the zenith angle projects a circle onto the tec screen.
        # the radius of the circle is given by:
        # zen_angle should be in radians
        zen_angle_radius = h_pix * np.tan(source_zenith_angle)
        # The azimuth angle gives us the arc on this circle from some starting point
        # We can then obtain the u and v coordinates for the pierce point.
        pp_u_coord = zen_angle_radius * np.sin(source_azimuth_angle) + new_u0
        pp_v_coord = zen_angle_radius * np.cos(source_azimuth_angle) + new_v0
        # Shift the u and v piercepoint values in reference to the phase screen center
        u_axis_pierce_point = u_shift_factor - pp_u_coord
        v_axis_pierce_point = v_shift_factor + pp_v_coord
        # Collect pierce points for each antenna for the source.
        ants_u_pierce_points.append(u_axis_pierce_point)
        ants_v_pierce_points.append(v_axis_pierce_point)

        # get the u and v direction gradients at the antenna pierce point.
        u_gradient = du[
            int(round(u_axis_pierce_point)), int(round(v_axis_pierce_point))
        ]
        # print(type(u_gradient))
        v_gradient = dv[
            int(round(u_axis_pierce_point)), int(round(v_axis_pierce_point))
        ]

        ants_u_gradients.append(u_gradient)
        ants_v_gradients.append(v_gradient)

    return (
        np.array(ants_u_pierce_points),
        np.array(ants_v_pierce_points),
        np.array(ants_u_gradients),
        np.array(ants_v_gradients),
    )


@njit
def add_phase_offsets(antenna1, antenna2, uparams, vparams):
    """Compute the phase differences for each baseline.

    Parameters
    ----------
    mset :  Measurement set.\n
    params : 1D array of len(no. of antennas.)
        The phase offset value for each antenna.

    Returns
    -------
    array.
        The phase difference per baseline.
    """
    # u_phasediffs, v_phasediffs = [], []
    # for uparam in u_params:
    #     diff_u = uparam[antenna1] - uparam[antenna2]
    #     u_phasediffs.append(diff_u)

    # for vparam in v_params:
    #     diff_v = vparam[antenna1] - vparam[antenna2]
    #     v_phasediffs.append(diff_v)
    u_phasediffs = uparams[antenna1] - uparams[antenna2]
    v_phasediffs = vparams[antenna1] - vparams[antenna2]

    return u_phasediffs, v_phasediffs


@njit
def single_source_offset_vis(
    lmbdas, uvw_lmbdas, flux, ll, mm, nn, u_phasediffs, v_phasediffs, spar
):
    """Compute single source offset visibilities"""
    np.expand_dims(u_phasediffs, axis=1)
    u_phasediff = float(spar) * np.expand_dims(u_phasediffs, axis=1) * lmbdas ** 2
    v_phasediff = float(spar) * np.expand_dims(v_phasediffs, axis=1) * lmbdas ** 2

    # u_phasediff = float(spar) * u_phasediffs[:, np.newaxis] * lmbdas ** 2
    # v_phasediff = float(spar) * v_phasediffs[:, np.newaxis] * lmbdas ** 2

    phase = np.exp(
        2j
        * np.pi
        * (
            (uvw_lmbdas[:, :, 0] * ll + u_phasediff)
            + (uvw_lmbdas[:, :, 1] * mm + v_phasediff)
            + uvw_lmbdas[:, :, 2] * nn
        )
    )

    return flux * phase


def compute_initial_setup_params(tec, us, vs, height, scale, ra0, dec0, time):
    # Lets first get the gradient of all pixels in the tecscreen
    du, dv = np.gradient(tec)  # save this in memory: first shared array.!!!
    # Apply scaling to the array field and height
    scaled_u_axis_ants_pos = scale_to_pixel_range(us, scale=scale)
    scaled_v_axis_ants_pos = scale_to_pixel_range(vs, scale=scale)

    u_tec_center = tec.shape[0] // 2  # + us_scaled[refant]
    v_tec_center = tec.shape[1] // 2

    h_pix = height / scale
    ra0_pp_u_coord, dec0_pp_v_coord = phase_center_offset(ra0, dec0, h_pix, time)

    u_coord_shift_factor = tec.shape[0] + ra0_pp_u_coord - u_tec_center
    v_coord_shift_factor = v_tec_center - dec0_pp_v_coord

    # And now at this point we have:
    initial_setup_params = (
        du,
        dv,
        scaled_u_axis_ants_pos,
        scaled_v_axis_ants_pos,
        u_coord_shift_factor,
        v_coord_shift_factor,
    )

    return initial_setup_params


@njit
def get_uv_phase_offsets(
    source_zenith_angle, source_azimuth_angle, initial_setup_params, h_pix, ant1, ant2,
):
    (
        du,
        dv,
        scaled_u_axis_ants_pos,
        scaled_v_axis_ants_pos,
        u_coord_shift_factor,
        v_coord_shift_factor,
    ) = initial_setup_params

    (
        ants_u_pierce_points,
        ants_v_pierce_points,
        ants_u_gradients,
        ants_v_gradients,
    ) = single_source_pierce_point(
        du,
        dv,
        source_zenith_angle,
        source_azimuth_angle,
        scaled_u_axis_ants_pos,
        scaled_v_axis_ants_pos,
        h_pix,
        u_coord_shift_factor,
        v_coord_shift_factor,
        refant=0,
    )

    # calculate the phase diffs for the source
    u_phasediffs, v_phasediffs = add_phase_offsets(
        ant1, ant2, ants_u_gradients, ants_v_gradients
    )

    return u_phasediffs, v_phasediffs


# @ray.remote
@njit
def single_source_offset_vis_(
    lmbdas,
    uvw_lmbdas,
    zen_a,
    az,
    ff,
    ll,
    mm,
    nn,
    initial_setup_params,
    h_pix,
    ant1,
    ant2,
    spar,
):
    u_phasediffs, v_phasediffs = get_uv_phase_offsets(
        zen_a, az, initial_setup_params, h_pix, ant1, ant2,
    )
    source_offset_vis = single_source_offset_vis(
        lmbdas, uvw_lmbdas, ff, ll, mm, nn, u_phasediffs, v_phasediffs, spar
    )
    return source_offset_vis


def data_incremental(offdata, source_offset_vis):
    offdata += source_offset_vis
    del source_offset_vis
    return offdata


@njit(parallel=True)
def compute_offset_vis_parallel(
    data,
    initial_setup_params,
    zenith_angles,
    azimuths,
    lmbdas,
    uvw_lmbdas,
    fluxes,
    ls,
    ms,
    ns,
    spar,
    h_pix,
    ant1,
    ant2,
):
    # result_ids = []
    # for zen_a, az, f, l, m, n in zip(zenith_angles, azimuths, fluxes, ls, ms, ns):
    source_num = 0
    for source in prange(len(fluxes)):
        print("source: ", source_num)
        #     result_ids.append(
        #         single_source_offset_vis_.remote(
        #             lmbdas,
        #             uvw_lmbdas,
        #             zen_a,
        #             az,
        #             f,
        #             l,
        #             m,
        #             n,
        #             initial_setup_params,
        #             h_pix,
        #             ant1,
        #             ant2,
        #             spar,
        #         )
        #     )

        #     offset_data = np.zeros_like(data[:, :, 0])
        #     while len(result_ids):
        #         print("adding a done offset source visibilities to data")
        #         done_ids, result_ids = ray.wait(result_ids)
        #         offset_data = data_incremental(offset_data, ray.get(done_ids[0]))
        #         del done_ids[0]

        offset_data = single_source_offset_vis_(
            lmbdas,
            uvw_lmbdas,
            zenith_angles[source],
            azimuths[source],
            fluxes[source],
            ls[source],
            ms[source],
            ns[source],
            initial_setup_params,
            h_pix,
            ant1,
            ant2,
            spar,
        )
        source_num += 1
    data[:, :, 0] += offset_data  # feed xx data
    data[:, :, 3] += offset_data  # feed yy data

    return data


'''
def single_offset_vis(
    uvw_lmbdas, amp, l, m, n, lmbdas, u_phasediff, v_phasediff, spar
):
    """Offset visibilities"""
    u_phasediff = float(spar) * u_phasediff[:, np.newaxis] * lmbdas ** 2
    v_phasediff = float(spar) * v_phasediff[:, np.newaxis] * lmbdas ** 2

    phse = np.exp(
        2j
        * np.pi
        * (
            (uvw_lmbdas[:, :, 0] * l + u_phasediff)
            + (uvw_lmbdas[:, :, 1] * m + v_phasediff)
            + uvw_lmbdas[:, :, 2] * n
        )
    )
    return amp * phse


def mp_offset_vis(data, lmbdas, uvw_lmbdas, A, ls, ms, ns, u_phasediffs, v_phasediffs, spar):
    def make_inputs():
        print("packing source parameters")
        pars_per_source = []
        source_count = 1
        for amp, l, m, n in zip(A, ls, ms, ns):
            print(f"source; {source_count}")
            single_source_pars = []
            single_source_pars.append(uvw_lmbdas)
            single_source_pars.append(amp)
            single_source_pars.append(l)
            single_source_pars.append(m)
            single_source_pars.append(n)
            single_source_pars.append(lmbdas)

            ind = source_count-1
            single_source_pars.append(u_phasediffs[ind])
            single_source_pars.append(v_phasediffs[ind])
            single_source_pars.append(spar)

            pars_per_source.append(tuple(single_source_pars))
            source_count += 1
        print("Done packing parameters")
        return pars_per_source

    with mp.Pool(processes=mp.cpu_count()) as pool:
        offset_data = sum(pool.starmap(single_offset_vis, make_inputs()))

    data[:, :, 0] += offset_data  # feed xx data
    data[:, :, 3] += offset_data  # feed yy data
    return data



#@ray.remote
def single_offset_vis(
    uvw_lmbdas, lmbdas, u_phasediffs, v_phasediffs, spar, source_pars
):
    """Offset visibilities"""
    amp, l, m, n, index = source_pars

    u_phasediff = float(spar) * u_phasediffs[index][:, np.newaxis] * lmbdas ** 2
    v_phasediff = float(spar) * v_phasediffs[index][:, np.newaxis] * lmbdas ** 2

    phse = np.exp(
        2j
        * np.pi
        * (
            (uvw_lmbdas[:, :, 0] * l + u_phasediff)
            + (uvw_lmbdas[:, :, 1] * m + v_phasediff)
            + uvw_lmbdas[:, :, 2] * n
        )
    )
    return amp * phse


def data_incremental(offdata, source_offset_vis):
    offdata += source_offset_vis
    del source_offset_vis
    return offdata


def mp_offset_vis(
    data, uvw_lmbdas, lmbdas, u_phasediffs, v_phasediffs, spar, A, ls, ms, ns
):
    uvw_lmbdas_id = ray.put(uvw_lmbdas)
    lmbdas_id = ray.put(lmbdas)
    u_phasediffs_id = ray.put(u_phasediffs)
    v_phasediffs_id = ray.put(v_phasediffs)

    source_indices = np.arange(len(A))
    result_ids = []  #

    for source_pars in zip(A, ls, ms, ns, source_indices):
        result_ids.append(
            single_offset_vis.remote(
                uvw_lmbdas_id,
                lmbdas_id,
                u_phasediffs_id,
                v_phasediffs_id,
                spar,
                source_pars,
            )
        )

    # result_ids = np.zeros_like(data[:, :, 0])
    # i = 0
    # for source_pars in zip(A, ls, ms, ns, source_indices):
    #     print(f"offset source {i} spar {spar}")
    #     result_ids += np.array(ray.get(single_offset_vis.remote(uvw_lmbdas_id, lmbdas_id,
    #                                                             u_phasediffs_id, v_phasediffs_id, spar, source_pars)))
    #     i += 1
    offset_data = np.zeros_like(data[:, :, 0])
    while len(result_ids):
        print("adding a done offset source visibilities to data")
        done_ids, result_ids = ray.wait(result_ids)
        offset_data = data_incremental(offset_data, ray.get(done_ids[0]))
        del done_ids[0]

    data[:, :, 0] += offset_data  # feed xx data
    data[:, :, 3] += offset_data  # feed yy data

    return data



def add_to_data(data, d, lock):
    lock.acquire()
    data.value[:, :, 0] += d
    data.value[:, :, 3] += d
    lock.release()


def _merge_func(params):
    """
This internal function is designed to be called in parallel, and takes a
tuple of parameters. `spectra` passed in through `params` should contain
data collected from the same place on the sky
these data are merged
together and returned as a single spectrum.
"""
    true_vissed = true_vis2(
        uvw_lmbdas=params["uvw_lmbdas"],
        A=params["amps"],
        ls=params["ls"],
        ms=params["ms"],
        ns=params["ns"],
    )

    return true_vissed  # (params["source"], ra, dec, bandpassed, mask)


def merge_true_vis(data, uvw_lmbdas, A, ls, ms, ns):
    processes = mp.cpu_count()
    pool = mp.Pool(processes=processes)

    to_be_merged = []
    for amp, l, m, n in zip(A, ls, ms, ns):
        to_be_merged.append(
            {"uvw_lmbdas": uvw_lmbdas, "amps": amp, "ls": l, "ms": m, "ns": n}
        )

    all_true_vissed = list(pool.imap(_merge_func, to_be_merged, chunksize=4))
    print(len(all_true_vissed))
    print(all_true_vissed)
    pool.close()
    pool.join()

    data[:, :, 0] += all_true_vissed  # feed xx data
    data[:, :, 3] += all_true_vissed  # feed yy data

    return data
'''


# Adopted from Bella's code
def thermal_variance_baseline(
    dnu=40000.0000000298, Tsys=240, timestamps=1, effective_collecting_area=21
):
    """
        The thermal variance of each baseline (assumed constant across baselines/times/frequencies.
        Equation comes from Trott 2016 (from Morales 2005)
        """
    # dnu = frequencies[1] - frequencies[0]
    integration_time = timestamps * 8

    sigma = (
        2
        * 1e26
        * constants.k
        * Tsys
        / effective_collecting_area
        / np.sqrt(dnu * integration_time)
    )
    return sigma ** 2


def add_thermal_noise(visibilities, dnu):
    """
    Add thermal noise to each visibility.
    Parameters
    ----------
    visibilities : (n_baseline, n_freq)-array
        The visibilities at each baseline and frequency.
    frequencies : (n_freq)-array
        The frequencies of the observation.
    beam_area : float
        The area of the beam (in sr).
    delta_t : float, optional
        The integration time.
    Returns
    -------
    visibilities : array
        The visibilities at each baseline and frequency with the thermal noise from the sky.
    """

    print("Adding thermal noise...")
    rl_im = np.random.normal(0, 1, (2,) + visibilities.shape)

    # NOTE: we divide the variance by two here, because the variance of the absolute value of the
    #       visibility should be equal to thermal_variance_baseline, which is true if the variance of both
    #       the real and imaginary components are divided by two.
    return visibilities + np.sqrt(thermal_variance_baseline(dnu) / 2) * (
        rl_im[0, :] + rl_im[1, :] * 1j
    )
