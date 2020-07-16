
import numpy as np

import phase_screen


constant = 40.30819
tecu = 1e16


def test_linear_tec():
    assert phase_screen.linear_tec(100).shape[1] == 100


def test_scale_to_pi_range():
    x = np.array([0, 5, 10])
    y = phase_screen.scale_to_pi_range(x)
    z = np.array([-np.pi, 0, np.pi])
    assert np.array_equal(y, z)


def test_scale_to_pixel_range():
    x = np.array([5, 10, 15])
    y = phase_screen.scale_to_pixel_range(x, 1)
    z = np.array([0., 7.5, 15.])
    assert np.array_equal(y, z)


def delta_stec(del_theta, freq):
    # \Delta STEC [1/m^3] = \Delta\theta [rad] * \nu^2 [(MHz)^2] / k
    # k = 1/(8\pi^2) e^2/(\epsilon_0 m_e) [m^3 / s^2]
    # 1 TECU = 10^16 [1/m^2]
    stec = del_theta * (freq * 1e6) ** 2 / constant
    stec /= tecu
    return stec


print("stec", delta_stec(24.514967664510966, 150))

"""
ras = [0.0]
decs = [-27.0]

new_ras = [0.017]
new_decs = [-27.021]


mset = "simulated_1source_phase_center.ms"
npz = "simulated_1source_phase_center_pierce_points.npz"


tbl = table(mset, readonly=True)
print(tbl.colnames())

data, uvw_lmbdas, ls, ms, ns = vis_sim.sim_prep(tbl, ras, decs)
offset_data = tbl.getcol("OFFSET_DATA")

nls, nms, nns = mset_utils.get_lmns(tbl, new_ras, new_decs)

print(nls, nms, nns)

for i in range(10):
    plt.plot(range(768), np.angle(data[i, :, 0]))
    plt.plot(range(768), np.angle(offset_data[i, :, 0]))
plt.legend()
plt.savefig("comparison_1row.png")

tbl.close()



def scale_to_pi_range(phscreen):
    scale the phas screen to[-pi, pi] range.

 Parameters
  ----------
   phscreen: array.
     The phase screen

    Returns
    -------
    array.
     scaled phasescreen.

    return ((phscreen - np.min(phscreen)) / (np.max(phscreen) - np.min(phscreen))) * 2*np.pi - np.pi

"""
