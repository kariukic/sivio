from casacore.tables import table
import scint_equations as sqs
import numpy as np

c = 299792458


def add_scint(data, bls, rdiff):
    """
    Corrupt visibilities with ionospheric scintillation.
    V_meas(b,l) = V_true(b,l) * exp[-1/2*D(b)]
    b is the baseline length
    """

    # calculate our structure function d(b).
    db = sqs.structure_fn_approx(bls, rdiff)
    print('structure function shape: ', db.shape)

    for i in range(data.shape[0]):
        data[i, :] *= np.exp(-0.5*db[i])

    return(data)


def add_phase_offsets(ms):
    mset = table(ms, readonly=False, ack=False)
    antenna1 = mset.getcol("ANTENNA1")
    antenna2 = mset.getcol("ANTENNA2")
    # freqs = mset.SPECTRAL_WINDOW.getcell('CHAN_FREQ', 0)
    # Antennas runs from 1-128
    # antids = np.array(range(0, len(mset.ANTENNA)))

    from phase_screen import get_antenna_in_uvw
    us, vs, ws = get_antenna_in_uvw(ms, mset)

    # ds = np.sqrt(np.abs(us**2 + vs**2))
    ds = vs
    params = 0.0008*ds
    # params[0] = 0
    # print(params)

    # from phase_screen import run_all
    # params = run_all(ms)
    print(params[antenna1] - params[antenna2])
    return params[antenna1] - params[antenna2]

    # ant1_coef = np.zeros((len(antenna1), 768, 3))
    # ant2_coef = np.zeros((len(antenna2), 768, 3))

    # for a in range(len(antids)):
    #    for f in range(768):
    #        ant1_coef[antenna1==a,f] = params[a]
    #       ant2_coef[antenna2==a,f] = -params[a]

    # return ant1_coef + ant2_coef

    # phase1 = ant1_coef[:,:,0] +
    #           ant1_coef[:,:,1] * l * uvw_lmbdas[:, :, 0] +
    #           ant1_coef[:,:,2]*m*uvw_lmbdas[:, :, 1] +
    #           uvw_lmbdas[:, :, 2] * n
    # phase2 = ant2_coef[:,:,0] +
    #           ant2_coef[:,:,1] * l * uvw_lmbdas[:, :, 0] +
    #           ant2_coef[:,:,2] * m *uvw_lmbdas[:, :, 1] +
    #           uvw_lmbdas[:, :, 2] * n
    # phase2 = ant2_coef[:,0] + l*ant2_coef[:,1] + m*ant2_coef[:,2] * n

    # phasediff = 1j*(phase1 - phase2)
    # phasediff = phase1 - phase2
    # print(phasediff[:50], '######phasediff#####')
    # print(phasediff.shape, '-----------------------phasediff')

    # return phasediff