# What summer4 sets up for you

When you `import summer4`, it makes four choices on your behalf so that models are accurate
and fast without any set-up. This page says what each choice is, how to check it, and how to
change it. If you never read this page, nothing here needs your attention.

You can see all four at once:

```python
import summer4

summer4.runtime_info()
```

`runtime_info()` reports whether 64-bit numbers are on (`x64`), how many CPU devices summer4
asked for and why (`host_devices_requested`, `host_devices_source`), and the solver that
`run()` uses when you do not name one (`default_solver`). It does not start JAX, so it is
cheap to call. `runtime_info().devices()` lists the devices JAX can use; calling it starts JAX.

Environment variables must be set **before Python starts** (or at least before the first
`import summer4`). In a notebook, set them in the shell that starts Jupyter, or use the
function given next to each one.

## 1. 64-bit numbers

**What happens.** summer4 turns on JAX's 64-bit mode when it is imported. Epidemic models mix
very large and very small numbers (a million people, a rate of one in a million), and 32-bit
numbers lose the small ones.

**Check.** `summer4.runtime_info().x64` is `True`.

**Change.** Set `JAX_ENABLE_X64=0` before starting Python, and summer4 leaves JAX's choice
alone. Or, after importing summer4 and before compiling a model, call
`jax.config.update("jax_enable_x64", False)`. The setting is for the whole process, so it
applies to every JAX library you use in it.

## 2. One CPU device per core

**What happens.** JAX normally treats your whole CPU as one device and runs one computation
at a time on it. On import, summer4 asks JAX to report one CPU device for each core of your
computer, so that work can be spread across the cores. A single run does not become faster or
slower by this (it still runs on one device); it matters for running several MCMC chains at
once (section 4).

**Check.** `summer4.runtime_info().host_devices_requested` is the number summer4 asked for, and
`host_devices_source` says why: `"default"` (the core count), `"SUMMER4_HOST_DEVICES"`,
`"XLA_FLAGS"` (you chose), `"disabled"` or `"too-late"`. `jax.device_count()` is what JAX
actually reports.

**Change.**

- `SUMMER4_HOST_DEVICES=4` asks for 4 devices instead of one per core.
- `SUMMER4_HOST_DEVICES=0` (also `off`, `false`, `no`) switches the feature off: summer4 sets
  nothing, and JAX reports its usual single CPU device. `SUMMER4_HOST_DEVICES=1` asks for exactly one.
- Setting `XLA_FLAGS=--xla_force_host_platform_device_count=N` yourself also works, and
  summer4 will not touch it.

**Import summer4 before using JAX.** JAX reads this setting when it starts up. If your code
calls JAX (even `jax.devices()` or `jnp.array(...)`) before `import summer4`, the setting
comes too late: summer4 changes nothing and issues a warning, and
`host_devices_source` is `"too-late"`. Put `import summer4` first.

## 3. The default solver

**What happens.** A model is a set of differential equations, and a *solver* steps them
forward in time. When you call `run()` without `solver=`, summer4 uses **Tsit5**, a
fifth-order Runge-Kutta method, with an **adaptive step**: the solver takes short steps where
the model changes quickly and long steps where it changes slowly, and keeps the error under a
tolerance (relative `1e-6`, absolute `1e-9`).

Why this default:

- **Accurate.** On the predator-prey model of the previous page, a fixed-step Euler solver
  with `dt=0.1` spirals away from the true closed cycle. The default stays on it.
- **Fast.** It takes far fewer steps than a fixed-step solver that is accurate enough. On
  the mini benchmark machine (12 cores), `run()` was about 30 times faster than Euler at the
  documented step size for the small models, and about 20 times faster for the tuberculosis-scale
  model, with errors thousands of times smaller.

`dt` is still required. With the default it is the **first** step the solver tries, and it
sets the times at which results are saved when you do not give a `SavePlan` with `ts`. The
solver then chooses its own steps in between.

**Check.** `summer4.runtime_info().default_solver` names it, for example
`"tsit5 (rtol=1e-06, atol=1e-09)"`.

**Change.**

- For one run: `run(..., solver="euler")`, or any other solver (`"heun"`, `"tsit5"`,
  `"dopri5"`, a diffrax solver, or a backend object). A `solver=` argument always wins.
- Tolerances for one run: `run(..., rtol=1e-8, atol=1e-10)`.
- Tolerances as an object you can reuse: `summer4.solvers.default_solver(rtol=1e-8, atol=1e-10)`.
  Its docstring shows the one line it stands for:
  `Diffrax(diffrax.Tsit5(), stepsize_controller=diffrax.PIDController(rtol=..., atol=...))`.
- For the whole process: `summer4.solvers.set_default_solver("euler")`, and
  `set_default_solver(None)` puts the built-in default back.
- Before Python starts: `SUMMER4_SOLVER=euler` (any of `euler`, `heun`, `tsit5`, `dopri5`).
  `set_default_solver` overrides it.

## 4. MCMC chains run in parallel

**What happens.** A Bayesian fit runs several chains (independent random walks over the
parameters) so that you can check that they agree. With `chain_method="auto"`, the default, the chains
run in parallel, one per CPU device, when there are at least as many devices as chains.
Otherwise they run together on one device ("vectorized"). With 4 chains on a 12-core machine,
the benchmark ran about 1.6 times faster in parallel (0.86 s against 1.36 s per call),
with identical convergence statistics.

**Check.** `summer4.epi.calibration.resolve_chain_method("auto", 4)` returns `"parallel"` or
`"vectorized"`, the method the next 4-chain run would use.

**Change.** Pass `chain_method="vectorized"` (or `"parallel"`, or `"sequential"`) to
`bm.sample(...)` or `wf.run_mcmc(...)`. If you build your own numpyro `MCMC`, one line makes
it follow the same rule:

```python
MCMC(kernel, num_warmup=500, num_samples=500, num_chains=4,
     chain_method=resolve_chain_method("auto", 4))
```

Draws for a given seed can differ between the two methods. Both sample the same posterior, so
compare distributions and summaries, not individual draws. Ensemble kernels stay vectorized.

## Using a GPU

summer4 only adds **CPU** devices. If JAX has a GPU, JAX still uses it as its default device,
and nothing on this page changes that. To choose the platform yourself, set `JAX_PLATFORMS`
before starting Python (for example `JAX_PLATFORMS=cpu` to ignore the GPU, or
`JAX_PLATFORMS=cuda`). `chain_method="auto"` counts the local devices of JAX's default platform, so with a GPU it
usually sees one device and keeps the chains vectorized. State `chain_method` yourself if you
want a particular behaviour.

## For test suites

The default solver is adaptive, so a test that compares numbers against a hand-worked
fixed-step result will not match it. For a reproducible fixed-step run:

- set `SUMMER4_SOLVER=euler` in the test environment, or call
  `summer4.solvers.set_default_solver("euler")` in a fixture and
  `set_default_solver(None)` when it finishes;
- or pass `solver="euler"` in just those tests.

For a single CPU device, so that chains run vectorized as they did before, set
`SUMMER4_HOST_DEVICES=1`.

A model that turns parameters into Python numbers inside an older-style custom function (for example
`float(params["beta"])`) fails under the adaptive solver, because the values are traced by
JAX. Keep them as JAX arrays, or pass `solver="euler"`.
