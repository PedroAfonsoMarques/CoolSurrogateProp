# SS304.py
# https://trc.nist.gov/cryogenics/materials/304Stainless/304Stainless_rev.htm

DATA = {
    "Rho":   7930.0,
    "Kappa1": (-1.4087,  1.3982,  0.2543, -0.6260,  0.2334,
                0.4256, -0.4658,  0.1650, -0.0199),
    "C1":     (22.0061, -127.5528, 303.647, -381.0098,  274.0328,
              -112.9212,   24.7593,  -2.239153,   0.0),
    "C2":     None,    # single regime
    "T_C12":  None,
    "single_C": True,
}