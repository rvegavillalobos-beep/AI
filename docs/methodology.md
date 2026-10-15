# How RICKHARD X computes its numbers

Everything below lives in the `rickhard` package (no Streamlit inside), and the formulas inherited from the
original RICKHARDV app are checked against a verbatim copy of its code in `tests/test_parity.py`.

## 1. Reading the export

| Step | Rule |
|---|---|
| Header | The first of the first 60 rows that contains a time, a part ID, a feature name and X / Y deviations. Column names are matched by meaning (`Time`, `Timestamp`, `Part ID`, `Serial`, `Featurename`, `X deviation`, `dX`, …); a manual mapping in *Data & Settings* overrides the guess. |
| CSV | Delimiter sniffed among `,` `;` tab `|`; with `;` and numbers like `0,25` the decimal comma is used. |
| Times | Known export formats are tried on a sample (`Aug 19, 2026 7:16pm`, ISO, `19.08.2026 19:16`, `08/19/2026 7:16 PM`, Excel serial numbers); otherwise each value is parsed on its own. |
| Time zone | Export time (default Europe/Berlin) → plant time (default America/Mexico_City). The repeated autumn hour is read as summer time and the skipped spring hour is shifted forward, so no reading is dropped; both are counted in the data-quality report. |
| Clean-up | Exact duplicate rows (repeated lines, overlapping exports) are removed. Readings of exactly 0 / 0 (/ 0) are treated as failed reads (switchable). Part IDs that differ only by the letter O vs the digit 0 are reported and can be merged. |
| Corners | Feature name → FL / FR / RL / RR with the original rules (plus optional custom rules). |
| Battery type | From the part ID and every feature ever measured on that part (original rules: `_DJ` / `_DI` → Type M, `_DA` → Type S, …). |
| Runs | Walking the readings in time order, a part starts a new run when a corner it already has in the current run appears again (original rule). Optionally also when the part comes back after more than *N* minutes (default 30). Run 1 is the first measurement and the only one that counts for FPY. |

## 2. Pass / fail and first-pass yield

* A corner is missing when its X or Y is missing. A complete run **FAILs** when any corner has |X| or |Y| above the
  tolerance (default ±3 mm; on the limit still passes, as in the original). With *Z decides PASS/FAIL* on, Z outside
  its limits fails too.
* **FPY** = share of modules whose Run 1 passes. Incomplete Run 1s count as NOK by default (original behaviour).
* Weekly FPY carries a **Wilson 95 % interval**; weeks with fewer than 5 modules are marked as low sample. A
  *near miss* is a PASS with less than 0.5 mm margin to the limit.

## 3. Error anatomy

For one battery type, the eight deviations of a module (X and Y of four corners) are written in an orthonormal basis
of eight physical modes, built by Gram–Schmidt in this order:

| Mode | Raw shape (corner offsets relative to the nominal module centre `rx`, `ry`) |
|---|---|
| Shift X, Shift Y | (1, 0), (0, 1) at every corner |
| Yaw | (−ry, rx): a small rotation about the nominal centre |
| Length / width stretch | (rx, 0), (0, ry) |
| Parallelogram (shear) | (ry, rx) |
| Trapezoids | (rx·ry, 0), (0, rx·ry), scaled |

Because shift and yaw come first they stay exact: their coefficients are the **least-squares placement** of the
module (`Fit_Tx`, `Fit_Ty` in mm, `Fit_Rot` in degrees). The decomposition is exact — the eight amplitudes rebuild the
eight deviations — and each amplitude is the RMS displacement the mode gives the four corners.

* **Error budget**: per mode, the systematic part (mean over modules — a fixed correction removes it) and the random
  part (σ — only reducing variation removes it).
* **Data-driven modes**: a PCA of the deviations shows the directions this process actually varies in, labelled with
  the physical mode they resemble most.
* **Failure mechanism** of every NOK Run 1: the deviations are split into shift T, yaw R and shape S (D = T + R + S).
  On each out-of-tolerance coordinate each part either pushes the corner outwards or holds it back; the mechanism
  names the part doing ≥ 60 % of the pushing (Shift, Yaw or Shape), *Shift + yaw* when placement as a whole does
  ≥ 75 %, otherwise *Placement + shape*. *Height (Z)* when X and Y are in tolerance, *Missing corner* when a corner
  was not reported. Two counterfactuals are kept per module: would it pass if perfectly placed (own shape), and if
  perfectly square (own placement)?

## 4. Shape and squareness (original formulas)

Diagonal delta = |(FL–RR − FR–RL) − nominal|. Above the limit (default 1.5 mm) the module is *DEFORMED* and the pattern
is named in this order: parallelogram tilt (FL angle deviation > 0.15° with width delta < 0.8 mm), trapezoidal width
(|width delta| ≥ 0.8 mm), trapezoidal length (|length delta| ≥ 0.8 mm), combined asymmetry. The link to NOK is tested
with Fisher's exact test (deformed vs square modules) and shown as NOK rate per deformation band.

## 5. Capability

Per corner and axis on Run 1: **Cp / Cpk** with the short-term σ from the average moving range (MR̄ / 1.128),
**Pp / Ppk** with the overall σ, Bissell's 95 % interval for Ppk, and the expected ppm out of tolerance from a normal
model (also for the same σ perfectly centred — the gap between the two is what centring is worth). The diagnosis
says whether a coordinate suffers from **offset** (a correction helps) or **scatter** (find the source of variation).

## 6. SPC and process shifts

* **I-MR** charts of the placement (`Fit_Tx`, `Fit_Ty`, `Fit_Rot`) and of the worst corner, with Nelson rules 1
  (beyond 3σ), 2 (9 on one side), 3 (6 in a row rising or falling), 5 (2 of 3 beyond 2σ) and 6 (4 of 5 beyond 1σ).
  Weekly X̄ charts use limits that narrow with the number of modules per week.
* **Change points**: binary segmentation of the mean with penalty 3·σ²·ln n (σ from the point-to-point noise),
  minimum segment length max(12, n/40), at most 4 shifts; each is dated and tested with Welch's t-test.
* **Alarm board**: the last *N* days (default 14) against a baseline — the data since the last change point before the
  window (if at least 30 modules), else everything before it. Level: Welch's test, flagged only if the move is at
  least 0.25 σ; spread: Brown–Forsythe, flagged if σ grew by 25 % or more; outliers: points beyond the baseline's 3σ
  against the 0.27 % expected by chance (binomial). *Signal* when any check has p < 0.001, *Watch* when p < 0.05.

## 7. Time patterns

Shifts follow the editable shift plan (default 06 / 14 / 22 h). NOK rates by shift, weekday or hour are tested with
a χ² test of independence, the placement by group with Kruskal–Wallis. A *line stop* is a gap of 5 × the median time
between modules (at least 60 min); the first 3 modules after it are the *start-up* group, compared with the rest by
Fisher's exact test (FPY) and Welch's test (placement).

## 8. Re-measurement and MSA

Consecutive runs of the same module less than 120 minutes apart are pairs of repeated measurements. Per type and
coordinate: σ_repeat = robust SD(Δ) / √2 (SD from the MAD), **%GRR = 6 σ_repeat / tolerance width** (< 10 % good,
10–30 % acceptable, > 30 % not acceptable). Taking a module out and re-seating it adds placement variation, so this is
an upper bound for the gauge itself; a formal gauge R&R study gives the definitive number. The transition table shows
how often a NOK becomes OK at the next run.

## 9. Height (Z)

With four corners the surface z = a + b·x + c·y + d·x·y (x, y relative to the nominal centre) fits exactly: level a,
pitch b and roll c (mm per metre) and twist d (the part no plane follows).

## 10. Compensation

A compensation is a rigid move of the placement: rotation by the yaw about the nominal module centre, then a shift in
X and Y. Every method is evaluated exactly on the modules (each corner re-judged against the tolerance).

* **Median-based** (original): minus the median centroid offset and minus the median yaw angle.
* **Rigid fit**: minus the mean least-squares placement.
* **Robust optimum**: for every yaw on a grid, the X and Y intervals in which each module passes are computed exactly;
  the setting maximises the **expected FPY** when the line places each module with Gaussian noise (σ 0.10 mm in X/Y and
  0.01° in yaw by default), and the smallest move wins ties. It optimises what matters — modules in tolerance — rather
  than a central value.
* **Backtest**: each of the last weeks gets the setting computed from the weeks before it, as if it had been used —
  the honest, out-of-sample comparison of the methods.
* **Readiness**: enough modules and weeks, stable weekly placement (medians within 0.5 mm and 0.05°), no change point
  in the last two weeks, and a backtest gain of at least 1 pp. A paired bootstrap gives the uncertainty of the gain.

## 11. Path to target and the action plan

The levers act on the mode coefficients of your own Run 1 modules: optional compensation, then shrinking the spread
of shift, yaw or shape around their means (and optionally removing the systematic shape). Every module is re-judged,
so no distribution is assumed. *Required reduction* finds by bisection the smallest spread reduction that reaches the
target. The Command Center ranks actions by the FPY they are estimated to be worth this way; findings carry the test
and p-value behind them.

## 12. Interventions

For a logged change: the 14 days before against the 14 days after (scope: the battery types it affects). FPY with
Fisher's exact test, placement and corner means with Welch's test, spread with Brown–Forsythe. The verdict says
whether FPY changed significantly, or whether the process moved without a significant FPY change yet.
