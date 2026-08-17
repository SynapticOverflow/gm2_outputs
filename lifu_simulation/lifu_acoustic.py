"""
lifu_acoustic.py

Acoustic / bio-effect model for transfontanellar LIFU (low-intensity focused
ultrasound) delivered through the infant skull fontanelles, used to compute a
region-specific blood-brain-barrier (BBB) opening efficiency eta in [0, 1].

    eta(f, P, DC, d) = T(f, d) * P_cav(P_tissue, MB_conc) * DC_factor(DC)

where
    T(f, d)      = fractional acoustic transmission efficiency at depth d
                   (dimensionless, in [0, 1])
    P_cav(...)   = probability of stable cavitation given the *actual* local
                   pressure at depth d (dimensionless, in [0, 1])
    DC_factor(DC)= linear duty-cycle scaling (dimensionless, in [0, 1])

Design note on units
---------------------
The task brief defines eta as the product of a "pressure at depth" term, a
cavitation-probability term, and a duty-cycle term, with eta constrained to
[0, 1]. A raw acoustic pressure (MPa) is not itself bounded to [0, 1], so it
cannot be used directly as a multiplicative factor in a [0, 1]-valued
efficiency. We resolve this the standard way: the *absolute* pressure at
depth (in MPa) is computed and used only to evaluate the cavitation
threshold condition (physically meaningful, since cavitation depends on
absolute pressure). The *fractional transmission efficiency*
T = P_tissue / P0 (dimensionless, automatically in [0, 1] since attenuation
and insertion loss only ever reduce pressure) is what enters the eta product.
This preserves the intent of the specification (delivery efficiency should
fall off with depth/frequency, gate on cavitation threshold, and scale with
duty cycle) while keeping eta well-defined and bounded.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Regional path lengths from the fontanelle acoustic window to each target
# region (mm). Values as specified for the 8 regions in scope. Basal Ganglia
# and other mastoid-fontanelle regions already include their extra path
# length in the quoted depth.
# ---------------------------------------------------------------------------
REGION_DEPTHS_MM = {
    "Frontal Cortex":  25.0,   # anterior fontanelle
    "Parietal Cortex": 35.0,   # anterior fontanelle
    "Thalamus":        55.0,  # anterior fontanelle
    "Basal Ganglia":   50.0,  # mastoid fontanelle (+5 mm path)
    "Temporal Lobe":   45.0,  # mastoid fontanelle
    "Occipital Lobe":  30.0,  # posterior fontanelle
    "Cerebellum":       40.0,  # posterior fontanelle
    "Hippocampus":      48.0,  # mastoid fontanelle
}

REGION_NAMES = list(REGION_DEPTHS_MM.keys())

# Fixed acoustic / bio-effect constants -------------------------------------
INSERTION_LOSS_DB = 0.5        # open-fontanelle transmission loss (<1 dB)
P_THRESHOLD_MPA = 0.15         # stable-cavitation pressure threshold
P_SCALE_MPA = 0.08             # sigmoid width for cavitation onset
DC_SATURATION = 0.02           # duty cycle (fraction) at which DC_factor -> 1.0
MB_CONCENTRATION_PER_ML = 1.0e8  # Definity 0.01 mL/kg -> fixed bubble conc.


def tissue_attenuation_db_per_cm(f_mhz):
    """Brain tissue attenuation coefficient, dB/cm, at frequency f (MHz).

    alpha(f) = 0.5 * f^1.1  [dB/cm]  (per task brief / NIST reference values)
    """
    return 0.5 * np.power(f_mhz, 1.1)


def pressure_at_depth_mpa(P0_mpa, f_mhz, depth_mm, insertion_loss_db=INSERTION_LOSS_DB):
    """Absolute peak-negative pressure (MPa) reaching depth `depth_mm`.

    Total loss (dB) = insertion loss (fontanelle) + attenuation * path length.
    Pressure (not intensity) attenuates as 10^(-dB/20).
    """
    depth_cm = depth_mm / 10.0
    alpha_db_cm = tissue_attenuation_db_per_cm(f_mhz)
    total_loss_db = insertion_loss_db + alpha_db_cm * depth_cm
    pressure_ratio = 10.0 ** (-total_loss_db / 20.0)
    return P0_mpa * pressure_ratio


def transmission_efficiency(f_mhz, depth_mm, insertion_loss_db=INSERTION_LOSS_DB):
    """Fractional transmission efficiency T = P_tissue / P0, in [0, 1].

    Independent of P0 because attenuation is linear in pressure; expressing
    it this way keeps the eta product dimensionless and bounded.
    """
    depth_cm = depth_mm / 10.0
    alpha_db_cm = tissue_attenuation_db_per_cm(f_mhz)
    total_loss_db = insertion_loss_db + alpha_db_cm * depth_cm
    return 10.0 ** (-total_loss_db / 20.0)


def cavitation_probability(P_tissue_mpa, P_threshold=P_THRESHOLD_MPA, P_scale=P_SCALE_MPA):
    """Sigmoid probability of stable cavitation at the local tissue pressure."""
    return 1.0 / (1.0 + np.exp(-(P_tissue_mpa - P_threshold) / P_scale))


def duty_cycle_factor(DC, dc_saturation=DC_SATURATION):
    """Linear 0->1 scaling of duty cycle (fraction, e.g. 0.01 = 1%) over
    [0, dc_saturation], capped at 1.0 above dc_saturation. Works elementwise
    on scalars or numpy arrays."""
    return np.clip(np.asarray(DC) / dc_saturation, 0.0, 1.0)


def compute_eta(f_mhz, P0_mpa, DC, depth_mm):
    """BBB-opening efficiency eta in [0, 1] for one (frequency, pressure,
    duty cycle, depth) combination. All arguments may be scalars or numpy
    arrays of matching shape (elementwise / broadcasted); returns a scalar
    float for scalar inputs and an ndarray for array inputs."""
    T = transmission_efficiency(f_mhz, depth_mm)
    P_tissue = pressure_at_depth_mpa(P0_mpa, f_mhz, depth_mm)
    p_cav = cavitation_probability(P_tissue)
    dc_f = duty_cycle_factor(DC)
    eta = np.clip(T * p_cav * dc_f, 0.0, 1.0)
    return float(eta) if np.isscalar(f_mhz) or np.ndim(eta) == 0 else eta


def compute_eta_per_region(f_mhz, P0_mpa, DC, region_names=None):
    """Return eta for every region as an (n_regions,) numpy array, in the
    order of `region_names` (defaults to REGION_NAMES)."""
    if region_names is None:
        region_names = REGION_NAMES
    return np.array(
        [compute_eta(f_mhz, P0_mpa, DC, REGION_DEPTHS_MM[r]) for r in region_names],
        dtype=np.float64,
    )


if __name__ == "__main__":
    # Quick sanity check when run directly.
    for f in (0.5, 0.65, 1.0):
        for P in (0.20, 0.35, 0.45):
            for DC in (0.005, 0.01, 0.02):
                etas = compute_eta_per_region(f, P, DC)
                print(f"f={f:.2f} MHz  P={P:.2f} MPa  DC={DC*100:.1f}%  "
                      f"eta range=[{etas.min():.3f}, {etas.max():.3f}]")
