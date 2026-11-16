import numpy as np

import batman


def test_quickstart():
    expected_quickstart = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.99986248,
            0.99931412,
            0.9985543,
            0.99766504,
            0.99669959,
            0.99569984,
            0.99470166,
            0.99373764,
            0.99283893,
            0.99203774,
            0.99137274,
            0.99091619,
            0.99074759,
            0.99061728,
            0.99050943,
            0.9904186,
            0.99034147,
            0.99027575,
            0.99021981,
            0.9901724,
            0.99013257,
            0.99009957,
            0.99007285,
            0.99005195,
            0.99003655,
            0.99002642,
            0.99002138,
            0.99002138,
            0.99002642,
            0.99003655,
            0.99005195,
            0.99007285,
            0.99009957,
            0.99013257,
            0.9901724,
            0.99021981,
            0.99027575,
            0.99034147,
            0.9904186,
            0.99050943,
            0.99061728,
            0.99074759,
            0.99091619,
            0.99137274,
            0.99203774,
            0.99283893,
            0.99373764,
            0.99470166,
            0.99569984,
            0.99669959,
            0.99766504,
            0.9985543,
            0.99931412,
            0.99986248,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )
    params = batman.TransitParams()
    params.t0 = 0.0
    params.per = 1.0
    params.rp = 0.1
    # b and T14 equivalent to the original a = 15, inc = 87 deg circular orbit
    a, inc = 15.0, np.radians(87.0)
    params.b = a * np.cos(inc)
    params.T14 = (
        params.per
        / np.pi
        * np.arcsin(np.sqrt((1 + params.rp) ** 2 - params.b**2) / (a * np.sin(inc)))
    )
    params.u = [0.1, 0.3]
    params.limb_dark = "quadratic"

    t = np.linspace(-0.015, 0.015, 100)
    m = batman.TransitModel(params, t)
    flux = m.light_curve(params)
    # the straight-chord (b, T14) model departs ~4e-6 from the curved a = 15 orbit
    np.testing.assert_allclose(flux, expected_quickstart, atol=5e-6)
