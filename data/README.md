# Columns

## Identity and Disposition

- `kepid`: Kepler Input Catalog ID for the target star; a unique numeric identifier used to join with stellar catalogs and light curves.
- `kepoi_name`: Kepler Object of Interest identifier (e.g., KOI-1234.01) for a candidate transit signal around a star.
- `kepler_name`: Official Kepler planet name if confirmed (e.g., Kepler-22 b); empty if unconfirmed.
- `koi_disposition`: Final vetting status (e.g., CONFIRMED, FALSE POSITIVE, CANDIDATE) used commonly as the ground-truth label.
- `koi_pdisposition`: Pipeline disposition before human vetting; indicates the automated pipeline’s initial assessment.
- `koi_score`: Confidence score associated with the KOI’s likelihood of being a real planet signal (scale defined by the archive release).

## Core Transit Parameters

- `koi_period`: Orbital period in days between successive transits of the candidate planet.
- `koi_duration`: Transit duration in hours (ingress to egress), reflecting planet size, orbit geometry, and stellar radius.
- `koi_depth`: Fractional transit depth (relative flux drop) indicating how much the star’s brightness decreases during transit.
- `koi_time0bk`: Reference transit epoch (BJD time of mid-transit) used to phase-fold the light curve.
- `koi_prad`: Planet radius in Earth radii, typically derived from transit depth and stellar radius estimates.
- `koi_ror (Rp/Rs)`: Radius ratio of planet to star; a geometry-driven feature tightly linked to transit depth.
- `koi_dor (a/Rs)`: Scaled semi-major axis (orbital distance divided by stellar radius) affecting transit duration and shape.
- `koi_impact`: Impact parameter (0=centered transit, ~1=grazing) controlling the V-shape vs U-shape of the transit.
- `koi_incl`: Orbital inclination in degrees relative to line of sight; transiting systems are typically near 90 degrees.

## Stellar Parameters

- `koi_srad`: Stellar radius in solar radii; critical for converting transit depth into an absolute planet size.
- `koi_smass`: Stellar mass in solar masses; informs orbital dynamics and derived parameters like semi-major axis.
- `koi_steff`: Stellar effective temperature in Kelvin; useful for stellar type and noise characteristics.
- `koi_slogg`: Surface gravity (log g) of the star; helps distinguish dwarfs vs giants, impacting transit interpretation.
- `koi_smet`: Stellar metallicity [Fe/H]; correlates with planet occurrence rates and can inform priors.
- `kic_kepmag (or koi_kepmag)`: Kepler magnitude (brightness) of the target; influences signal-to-noise and detection reliability.

## False Positive Diagnostic Flags

- `koi_fpflag_nt`: Flag indicating the event is “Not transit-like” (likely systematics or non-planet variability).
- `koi_fpflag_ss`: Flag for a secondary event suggesting an eclipsing binary rather than a planet.
- `koi_fpflag_co`: Flag for centroid offset (signal likely from a nearby star, not the target).
- `koi_fpflag_ec`: Flag for ephemeris match contamination—timing matches another known source’s variability.

## Vetting and Provenance

- `koi_vet_stat`: Vetting status (e.g., DONE/ACTIVE) indicating whether manual/automated review is completed.
- `koi_vet_date`: Date of last vetting action or status update for traceability.
- `koi_disp_prov`: Provenance of the disposition (pipeline/human review/version) for auditability.
- `koi_comment`: Free-text notes from vetting providing context on peculiarities or cautionary details.

## Uncertainties (Recommended)

- `koi_period_err1`, `koi_period_err2`: Asymmetric upper/lower uncertainties on orbital period (typically +/−).
- `koi_duration_err1`, `koi_duration_err2`: Uncertainties on transit duration.
- `koi_depth_err1`, `koi_depth_err2`: Uncertainties on transit depth.
- `koi_prad_err1`, `koi_prad_err2`: Uncertainties on planet radius.
- `koi_steff_err1`, `koi_steff_err2 (and similar for other stellar params)`: Uncertainties to propagate into downstream modeling.

Notes for ML usage

-  **Labeling**: koi_disposition is commonly used as the target label; some workflows merge CANDIDATE with CONFIRMED or treat them separately depending on goals.
- **Feature selection**: Start with core transit parameters plus stellar parameters and include diagnostic flags; add uncertainties as features or sample weights.
- **Data hygiene**: Filter ACTIVE vetting states if a stable ground truth set is required; review comments and flags to reduce false-positive leakage.

[Data Columns in Kepler Objects of Interest Table](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html)

[KEPLER_KOI](https://archive.stsci.edu/kepler/koi/help/columns.html)
