# Specification: Fast Correlated-Noise Generator (`openspin.noise`)

**Status:** Ready for implementation
**Owner:** (spin-qubit repo maintainer)
**Primary constraint:** *Everything must be fast.* Vectorized, FFT/`O(N log N)` where possible, no Python-level per-sample loops on the hot path.

---

## 1. Goal

A user-friendly generator of correlated (temporal and spatial) charge/qubit noise for spin-qubit simulation. The user gives **either**:

- a **process name + parameters** (e.g. `"1/f"`, `"ou"`, `"ou_sum"`, `"rtn"`, `"spatial_fluctuators"`), **or**
- a **measured charge-sensor trace** (1D or multi-channel array),

and gets back **many trajectories** on a requested time grid, generated quickly, plus a set of **analysis quantities** (spectrum, Allan deviation, autocorrelation, cross-correlation, cross-spectrum) packaged as **Cowork/Claude Skills**.

## 2. Design principles (non-negotiable)

1. **NumPy-first.** Baseline implementation is pure vectorized NumPy + `numpy.fft`. No SciPy hard-dependency for core generation (reimplement Welch/ADEV in NumPy). SciPy allowed only as an optional convenience.
2. **Optional acceleration behind one API.** A `backend=` switch (`"numpy"` default, `"numba"`, `"jax"`) selects the implementation of the *only* unavoidable sequential kernel (the OU time recursion). Public API and results (for a fixed seed policy) must be identical across backends up to floating-point.
3. **Vectorize over the ensemble.** All generators produce shape `(n_traj, [n_sites,] n_time)` in one shot. The trajectory axis is always a batch dimension — never loop over it.
4. **Exactness where it's free.** Use exact discrete-time schemes (exact OU update, circulant embedding) rather than Euler–Maruyama. Correct statistics at *any* step size.
5. **Reproducibility.** Explicit `numpy.random.Generator` / `SeedSequence`. Parallel/independent streams via `SeedSequence.spawn`. Every returned object records its seed and full process spec.
6. **Round-trip validated.** Every generator ships with a test asserting the *measured* PSD/ACF of the output matches the target within tolerance.

## 3. Package layout

```
openspin/noise/
  __init__.py            # generate(), calibrate(), NoiseResult
  core.py                # NoiseResult container, seeding, time grids, units
  registry.py            # named-process registry + presets
  generators/
    spectral.py          # 1/f via circulant embedding / Timmer-König  (method 1)
    ou.py                # scalar + multivariate exact OU              (method 2)
    ou_sum.py            # 1/f as a sum of OU / Lorentzians            (method 3)
    spatial.py           # separable + fluctuator-ensemble spatial     (method 4)
    rtn.py               # random telegraph / two-level fluctuators    (extra)
    compose.py           # sums/mixtures, drift, quasi-static
  calibrate.py           # trace -> estimate -> resynthesize
  analysis/              # numeric backends for the Skills (see §8)
    psd.py  allan.py  acf.py  xcorr.py  xspec.py
  backends/
    _numpy.py _numba.py _jax.py
skills/                  # Cowork Skills wrapping analysis/ (see §8)
tests/
benchmarks/
```

## 4. Public API

```python
from openspin.noise import generate, calibrate

# --- by name ---
res = generate(
    "1/f",
    n_traj=1000,
    fs=1e4,               # sample rate [Hz]  (or pass `t=` for arbitrary grid)
    n_points=2**16,
    alpha=1.0,            # 1/f^alpha
    S0=1e-12, f0=1.0,     # PSD normalization: S(f0)=S0  [units^2/Hz]
    seed=0,
    backend="numpy",
)
res.traj        # ndarray (n_traj, n_time), float64 (float32 option)
res.t           # ndarray (n_time,)
res.fs, res.spec, res.seed, res.units

# --- from a measured trace (estimate + resynthesize) ---
gen = calibrate(trace, fs=1e4, method="circulant")  # model-free surrogate
res = gen.sample(n_traj=1000, n_points=2**16, seed=0)
```

Rules:
- Accept **either** `fs` + `n_points` (uniform grid) **or** an explicit `t=` array (arbitrary/non-uniform grid). Non-uniform grids are required to work for OU-family generators (§6.2–6.3).
- `NoiseResult` is a thin dataclass wrapping the array + metadata. Optional `to_xarray()` with `time`/`site` coords. Optional `.save()/.load()` (`.npz`).
- Units: carry a `units` string; provide helpers to convert V ↔ eV ↔ detuning via a lever-arm α (integration point with the electrostatics/CSD modules).

## 5. Conventions (write these down once, enforce everywhere)

- **PSD is one-sided**, `S(f)` for `f ≥ 0`, in `units²/Hz`, satisfying `var(x) = ∫₀^{fs/2} S(f) df`.
- Wiener–Khinchin: `R(τ) = ∫₀^∞ S(f) cos(2πfτ) df`.
- Cyclic frequency `f` in **Hz** everywhere (not angular ω). OU rate `θ` in `1/s`; its Lorentzian corner is at `f = θ/2π`.
- A single "convention test" in `tests/` verifies generator output PSD (measured by `analysis/psd.py`) integrates to the sample variance. This catches the classic one-sided/two-sided and windowing-normalization bugs.

## 6. Generators

### 6.1 Method 1 — 1/f from a spectrum (circulant embedding / Timmer–König)
Goal: stationary Gaussian trace with arbitrary target `S(f)` (default `S(f)=S0·(f0/f)^alpha`).

Algorithm (FFT, `O(N log N)`, fully vectorized over `n_traj`):
1. Build target one-sided `S(f)` on the rFFT grid `f_k = k·fs/N`. Regularize DC and (optionally) a low-f knee `f_min` to keep variance finite for `alpha ≥ 1`.
2. **Timmer–König draw:** for each positive frequency draw complex coefficients `c_k = sqrt(S(f_k)·fs·N/2)·(a_k + i b_k)`, `a,b ~ N(0,1)` i.i.d. per trajectory; set DC and Nyquist real; enforce Hermitian symmetry.
3. `x = irfft(c)`. Normalize so measured variance/PSD matches the analytic target.
- Provide the **circulant-embedding** variant as the general path (accepts an arbitrary ACF or tabulated `S(f)` from data — this is what `calibrate(..., method="circulant")` calls). Embed the covariance in a circulant of length `≥ 2N`, `sqrt` its FFT, multiply by complex white noise, transform back. This is the exact, model-free engine reused in §7.
- Batch all trajectories as columns; a single `irfft` over the batch does the whole ensemble.

### 6.2 Method 2 — many OU from the vectorized Langevin equation (exact/exponential update)
Overdamped Langevin **with an explicit force term** `F`: `dx = (F - γx) dt + σ dW`.
**Exact exponential propagator** (valid for *any* Δt):
```
x[n+1] = x[n]·e^(-γ·Δt[n]) + (F/γ)·(1 - e^(-γ·Δt[n])) + σ·sqrt( (1 - e^(-2γ·Δt[n]))/(2γ) )·Z[n]
```
- **This is the exact solution — do NOT use Euler–Maruyama** (`x[n+1] = x[n] + (F - γx[n])·Δt + σ·sqrt(Δt)·Z`). The Euler drift is *linear* in Δt and only valid for `γΔt ≪ 1`; the propagator above advances the deterministic part as `(F/γ)(1 - e^{-γΔt})` and is correct at any step size. The whole point of the exp form is arbitrary Δt.
- **`F` defaults to 0** → zero-mean OU, stationary variance `σ²/2γ`, ACF `e^{-γ|τ|}`. Set `F ≠ 0` for a nonzero steady-state mean `F/γ`.
- `F` may be **constant or per-step piecewise-constant** `F[n]` — this injects a deterministic external drive/tilt (e.g. a control ramp) while keeping the update exact over each step.
- Initialize `x[0]` from the stationary distribution `N(F/γ, σ²/2γ)` (no burn-in).
- `Δt[n]` is per-step → **arbitrary/non-uniform time grids supported directly** (precompute the exp factors per unique Δt).
- Vectorize the recursion over `(n_traj, n_components)`; the only loop is over time. In `backend="numpy"` this loop is a vectorized array op per step; provide Numba/JAX (`lax.scan`) kernels when `n_time` is large.

**Multivariate / coupled OU (the "with the exp" version for arbitrary transitions):**
`dx = (F - B·x) dt + L dW`, with force vector `F` (**default 0**). Exact exponential scheme:
```
x[n+1] = A[n]·x[n] + B⁻¹·(I - A[n])·F + η[n],   A[n] = expm(-B·Δt[n])
Cov(η[n]) = Σ_ss - A[n]·Σ_ss·A[n]ᵀ,           B·Σ_ss + Σ_ss·Bᵀ = L·Lᵀ (Lyapunov)
```
- The deterministic term `B⁻¹(I - A)F` is the exact matrix-exp propagation of the drive (it reduces to the scalar `(F/γ)(1 - e^{-γΔt})` when `B=γ`); again **not** a linear `F·Δt` step. Steady-state mean `B⁻¹F`.
- `F` may be per-step piecewise-constant for an exact deterministic drive on the coupled system.
- Uniform grid → `A`, `B⁻¹(I - A)`, and `chol(Cov(η))` are constant: **precompute once**, then the per-step cost is one small matvec + one matmul with a cached Cholesky factor → extremely fast.
- This matrix-exponential form is what gives arbitrary Δt transitions and cross-correlated components; it doubles as a spatial-correlation engine (§6.4).

### 6.3 Method 3 — 1/f as a sum of OU (Lorentzian superposition / TLS ensemble)
Physical model: an ensemble of fluctuators with relaxation rates `θ_k` log-spaced over `[θ_min, θ_max]`. Each OU has one-sided PSD `S_k(f) = 2σ_k²/(θ_k² + (2πf)²)` (Lorentzian, corner `θ_k/2π`). Their sum approximates `1/f^alpha` between `θ_min/2π` and `θ_max/2π`.
- Canonical `alpha=1`: rates log-uniform, equal weight per decade. General `alpha`: pick weights `w_k` by a **fast non-negative least-squares fit** of `Σ_k w_k S_k(f)` to the target on a log-f grid — do this once at setup, cache.
- Choose `[θ_min, θ_max]` to bracket the requested band with ~a few components per decade (expose `n_components`, default e.g. 8/decade).
- **Reuses the vectorized OU engine (§6.2)** — generate all `n_components` OU processes as a batch and sum. This is the recommended default for physically-motivated 1/f because it's cheap, exactly stationary, and extends naturally to spatial correlations.

### 6.4 Method 4 — spatially correlated noise (efficient)
Two paths, both fast:

**(a) Separable covariance** `C(i,j;τ) = K_ij · R(τ)` — recommended default when you just need M correlated sites.
1. Generate `M` independent temporal traces `W` (shape `(n_traj, M, n_time)`) with the desired temporal spectrum via §6.1 or §6.3.
2. Mix: `Y = chol(K) · W` along the site axis. `K` is `M×M` (M = number of dots/sensors, small) → one tiny Cholesky + one `einsum`. Cost is dominated by the temporal generation, not the spatial mixing.

**(b) Distributed-fluctuator ensemble** — recommended when spatial *and* spectral structure must be physical (realistic charge noise).
- Place `N_f` two-level fluctuators (or OU sources) at random positions `r_f`; each couples to site `i` via a kernel `g_{i,f} = kernel(|x_i - r_f|)` (e.g. monopole `1/r`, dipole `1/r³`, or screened).
- Site noise `n_i(t) = Σ_f g_{i,f} s_f(t)`, i.e. `N = G · S` where `G` is `(M, N_f)` and `S` is `(N_f, n_time)` from the RTN/OU batch engines. One matmul → naturally spatially correlated 1/f with the right cross-spectra.
- Fully vectorized; `N_f` up to ~10⁴ is a single BLAS `matmul`.

Also provide **circulant embedding in ≥2D** for stationary random *fields* on a grid (FFT, `O(N log N)`) as an optional path for continuous spatial disorder — integration point with the FEM disorder module.

### 6.5 Extras (low cost, high value — include these)
- **RTN / random telegraph noise** (`rtn.py`): single and ensemble two-level fluctuators with switching rates `γ_up, γ_down`. Exact via exponentially-distributed dwell times, then resample onto the grid; vectorized over the ensemble. Building block for §6.4(b) and for non-Gaussian noise.
- **Compose / mix** (`compose.py`): sum independent processes, add slow **drift** and **quasi-static** (per-shot frozen) offsets, and mixtures (e.g. 1/f + white + a dominant TLS). One `NoiseResult` out.
- **Arbitrary tabulated `S(f)` from data** → time domain (feeds §7).
- **Non-uniform / requested time grids** honored across OU-family and (via interpolation of the target ACF) circulant paths.
- **Filter-function hook:** expose traces/PSD through a stable interface so a dephasing/coherence calculator (T2*, CPMG) can consume them — the other adjacent module in this repo. Keep it an integration point, not in scope here.

## 7. From a measured trace: estimate + resynthesize (model-free)

`calibrate(trace, fs, method="circulant")` returns a `SurrogateGenerator`:
1. **Estimate** the target second-order statistics from the trace: one-sided PSD (Welch, `analysis/psd.py`) and/or ACF (Wiener–Khinchin, `analysis/acf.py`). Handle detrending, windowing, and (optionally) a robust log-log fit only for *reporting* `alpha` — generation stays model-free.
2. **Resynthesize** surrogate trajectories that reproduce those statistics via **circulant embedding** (§6.1 general path). This preserves the measured spectrum/ACF exactly (Gaussian surrogate) without committing to a parametric model.
3. Multi-channel input → estimate the **cross-spectral matrix** and resynthesize jointly (per-frequency Cholesky of the CSD matrix), preserving inter-channel coherence.
`gen.sample(n_traj, n_points, seed)` then emits as many trajectories as wanted, fast.

(Parametric fitting to a named model is explicitly **out of scope** per decision; leave a `TODO`/hook only.)

## 8. Analysis quantities as Cowork Skills

Implement the numerics in `openspin/noise/analysis/` (NumPy, FFT-based, vectorized and ensemble-averaged), then wrap each as a **Cowork/Claude Skill** (a `SKILL.md` + thin script under `skills/`) that takes a trace file (`.npy`/`.csv`/`.npz`) or a `NoiseResult` and returns numbers + a quick plot.

Skills to ship (one each):
1. **`noise-spectrum`** — one-sided PSD via Welch (NumPy reimpl; segment, window, average), plus periodogram and optional multitaper. Ensemble-averaged over trajectories. Overlays target if known.
2. **`noise-allan`** — overlapping Allan deviation `σ_A(τ)` via cumulative sums (`O(N)` per τ, τ log-spaced). Good 1/f diagnostic; annotate slopes.
3. **`noise-autocorrelation`** — ACF via FFT (Wiener–Khinchin), biased/unbiased option, ensemble-averaged.
4. **`noise-crosscorrelation`** — cross-correlation between two traces/sites, FFT-based, with lag axis.
5. **`noise-crossspectrum`** — cross-spectral density, magnitude-squared **coherence** `γ²(f)`, and phase.

Skill requirements: each `SKILL.md` states inputs (file/array, `fs`), outputs (values + figure saved to outputs), and calls the shared `analysis/` functions so results are identical whether invoked as a library or a Skill. Keep them fast (FFT, vectorized) and dependency-light.

> Note: Skills are created in the repo by the implementing agent; they cannot be authored from this session's read-only skill cache.

## 9. Performance requirements

- **Vectorization mandate:** no Python loop over trajectories, sites, or samples on any hot path except the single OU time recursion (which must be Numba/JAX-accelerable).
- **Complexity:** spectral/circulant and all analysis quantities `O(n_traj · N log N)`; OU-family `O(n_traj · n_comp · N)`; spatial mixing adds only a small `M×M` (or `M×N_f`) matmul.
- **Indicative targets (laptop, single core, `numpy` backend):**
  - 1/f (method 1): `10⁴` traj × `2¹⁶` pts in ≲ 1 s.
  - OU-sum (method 3, 8 comps/decade, 4 decades): `10³` traj × `10⁵` pts in ≲ 1 s.
  - All analysis Skills: `10³ × 10⁵` in ≲ 1 s.
  - Numba/JAX backends: ≥5× on the OU recursion; JAX GPU for `≥10⁶` trajectories.
- **Memory:** `float32` option; chunked/streaming generation (iterator API) for traces longer than RAM; pre-allocate output buffers.
- **Seeding:** `SeedSequence.spawn` for independent parallel streams; identical results across chunk sizes.

## 10. Testing & validation

- **Round-trip:** for each generator, measure output PSD/ACF (via `analysis/`) and assert it matches the target within tolerance (statistical, `n_traj`-averaged).
- **Convention test** (§5): variance = ∫ PSD.
- **OU exactness:** stationary variance `σ²/2θ` and ACF `e^{-θ|τ|}` at large and pathologically large Δt (proves the exact scheme vs Euler).
- **OU-sum:** slope ≈ `-alpha` in log-log PSD across `[θ_min, θ_max]/2π`.
- **Spatial:** recovered `K`/coherence matches the target `K`/kernel.
- **From-trace:** surrogate PSD matches input-trace PSD; multi-channel coherence preserved.
- **Backend parity:** numpy vs numba vs jax agree within fp tolerance for a fixed seed policy.
- **Benchmarks** in `benchmarks/` asserting the §9 targets (regression guard).

## 11. Named-process registry & presets

- `registry.py` maps names → generator + default params, so `generate("charge_noise_SiGe", ...)` works. Ship a handful of literature-ish presets (a `1/f` charge-noise default with a sensible `S0`, an OU-sum default, an RTN-dominated default). Presets are overridable and documented with their assumptions/units.
- `generate(name_or_trace, ...)` dispatches: string → registry; array → `calibrate(...).sample(...)`.

## 12. Deliverables checklist

- [ ] `generate()` / `calibrate()` / `NoiseResult` public API (§4)
- [ ] Methods 1–4 + RTN + compose (§6)
- [ ] Estimate→resynthesize from trace, incl. multi-channel (§7)
- [ ] 5 analysis functions + 5 Cowork Skills (§8)
- [ ] numpy backend + numba/jax kernels for OU recursion (§2, §9)
- [ ] Registry + presets (§11)
- [ ] Tests, convention test, benchmarks (§10)
- [ ] README with 3 copy-paste examples (name, OU-sum, from-trace) + one figure

## 13. References (for the implementer)

- Timmer & König (1995), *On generating power law noise* — method 1 draw.
- Circulant embedding (Dietrich & Newsam 1997; Wood & Chan 1994) — exact stationary Gaussian via FFT.
- Gillespie (1996), *Exact numerical simulation of the OU process* — exact update, methods 2–3.
- Kubo/telegraph-noise & TLS-ensemble → 1/f (sum-of-Lorentzians) literature.
- Allan variance (overlapping estimator) — IEEE frequency-stability conventions.
