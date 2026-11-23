# What summer4 sets up for you

When you `import summer4`, it makes five choices on your behalf so that models are accurate
and fast without any set-up. This page says what each choice is, how to check it, and how to
change it. If you never read this page, nothing here needs your attention.

You can see them all at once:

```python
import summer4

summer4.runtime_info()
```

`runtime_info()` reports whether 64-bit numbers are on (`x64`), how many CPU devices summer4
asked for and why (`host_devices_requested`, `host_devices_source`), the solver that
`run()` uses when you do not name one (`default_solver`), and which of XLA's CPU runtimes runs
the compiled model (`cpu_runtime`, `cpu_runtime_source`). It does not start JAX, so it is
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

## 4. MCMC chains run vectorized (parallel is opt-in)

**What happens.** A Bayesian fit runs several chains (independent random walks over the
parameters) so that you can check that they agree. With `chain_method="auto"`, the default,
the chains run vectorized: all together on one device. This is the default for now because
numpyro's `"parallel"` method (one chain per CPU device) fails on recent JAX (0.11 observed)
when gradients go through the adaptive default solver.

**Check.** `summer4.epi.calibration.resolve_chain_method("auto", 4)` returns `"vectorized"`,
the method the next 4-chain run would use.

**Change.** Pass `chain_method="parallel"` (or `"vectorized"`, or `"sequential"`) to
`bm.sample(...)` or `wf.run_mcmc(...)`. Parallel works with `solver="euler"` but not yet with
the adaptive default on recent JAX. With 4 chains on a 12-core machine it ran about 1.6 times
faster than vectorized when it worked (0.86 s against 1.36 s per call), with identical
convergence statistics. If you build your own numpyro `MCMC`, one line makes it follow the
same rule:

```python
MCMC(kernel, num_warmup=500, num_samples=500, num_chains=4,
     chain_method=resolve_chain_method("auto", 4))
```

Draws for a given seed can differ between the two methods. Both sample the same posterior, so
compare distributions and summaries, not individual draws. Ensemble kernels stay vectorized.

## 5. XLA's CPU runtime settings

**What happens.** JAX compiles a model through XLA. From jaxlib 0.4.32, XLA's CPU backend runs
compiled code on a newer runtime, which is much slower for the small operations inside a
solver's time-stepping loop (reported upstream in jax-ml/jax issues 26145, 29772 and 40150).
XLA kept the older runtime behind a flag up to jaxlib 0.6. On those versions summer4 switches
it back on when imported. On the 41-compartment benchmark in `benchmarks/weighted_reduce.py`
(jaxlib 0.6.2, 12-core mini) runs became 3 to 4 times faster: a 2000-step Euler run took
0.9 ms instead of 3.4 ms, and its gradient 1.6 ms instead of 3.8 ms; the adaptive default took
0.32 ms instead of 1.17 ms.

From jaxlib 0.7 the older runtime is gone (on 0.7.0 and 0.7.1 its flag would make every run
fail, and later versions ignore it). There summer4 instead turns off the newer runtime's
*concurrency-optimised scheduler*, which reorders each step's small operations to run them in
parallel and costs more than it gains on a solver step. On the same benchmark (jaxlib 0.11.1)
this made the Euler run 1.8 times faster (1.5 ms instead of 2.7 ms), its gradient 2 times
faster, and the adaptive default 1.8 times faster. summer4 sets this flag only on the jaxlib
versions checked to accept it, 0.7.0 up to 0.12.x: XLA stops the program on a flag it does not
know, so newer versions are left alone until they are checked.

**Check.** `summer4.runtime_info().cpu_runtime` is `"legacy"` or `"thunk"`, and
`cpu_runtime_source` says why: `"default"`, `"SUMMER4_CPU_RUNTIME"`, `"XLA_FLAGS"` (you chose),
`"jaxlib"` (this jaxlib has only one runtime) or `"too-late"`. `cpu_concurrency_scheduler` is
`False` when the scheduler is off, `True` when it is on, and `None` with the older runtime.

**Change.**

- `SUMMER4_CPU_RUNTIME=thunk` (also `0`, `off`, `false`, `no`) keeps XLA's own defaults: the
  newer runtime, with its concurrency-optimised scheduler.
- Setting `XLA_FLAGS=--xla_cpu_use_thunk_runtime=...` or
  `XLA_FLAGS=--xla_cpu_enable_concurrency_optimized_scheduler=...` yourself also works, and
  summer4 will not touch it.

As with the devices in section 2, JAX reads this when it starts: import summer4 before using
JAX, or the choice comes too late and summer4 warns. It affects the CPU only.

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
