# summer4

**summer4** is a library for building compartmental models of disease spread,
such as SIR and SEIR models. You describe the compartments and the flows
between them, optionally split the population into groups such as age bands,
and run the model to get time series you can plot and fit to data. It is
written on JAX, so models can be fitted to data with Bayesian calibration.

summer4 is an alpha release (`0.2.0a7`) and the API may still change. Pages
marked as being written are placeholders.

## Install

summer4 is not published to PyPI. Install it from the git tag. With pixi,
add this to your project:

```toml
[pypi-dependencies]
summer4 = { git = "https://github.com/monash-emu/summer4.git", tag = "v0.2.0a7", extras = ["calibration", "pandas", "frames"] }
```

Or with pip:

```bash
pip install "summer4[calibration,pandas,frames] @ git+https://github.com/monash-emu/summer4.git@v0.2.0a7"
```

| Extra | Packages | When you need it |
| --- | --- | --- |
| `calibration` | numpyro, optax | Sampling and MAP fits |
| `pandas` | pandas | `Output.to_pandas`, `Result.to_frame(backend="pandas")` |
| `frames` | polars, pyarrow | `Output.to_frame`, `Result.to_frame` |

To work on summer4 itself, clone the repository and run `pixi install`, then
`pixi run setup` to install the git hooks and `pixi run test` to run the
tests. The documentation builds with `pixi run -e docs docs`; every notebook
on this site is executed during the build.

## A first model

An SIR model with an infection flow, whose rate is the contact rate times the proportion
of the population that is infectious, and a recovery flow, run for 60 days:

```python
import numpy as np
from summer4 import (
    Compartments, FlowModel, Param, Property, PropertyMap,
    SavePlan, SaveRequest, TransitionFlow,
)
from summer4.epi import ForceOfInfection

state = Property("state", ("S", "I", "R"))
model = FlowModel(PropertyMap.from_property(state))
infection = ForceOfInfection("infection", infectious=state["I"], contact_rate=Param("contact_rate"))
model.add_flow(TransitionFlow("infection", state["S"], state["I"], infection))
model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.2))
model.set_initial_population({state["S"]: 990.0, state["I"]: 10.0})

plan = SavePlan(requests={"compartments": SaveRequest(Compartments())}, ts=np.arange(61.0))
result = model.compile().run({"contact_rate": 0.5}, t0=0.0, t1=60.0, dt=0.1, save=plan)
result["compartments"].to_pandas().plot()
```

## Where next

- The {doc}`Guide <guide/index>` builds a model step by step.
- The {doc}`Textbook <textbook/index>` teaches infectious-disease modelling
  with summer4.
- The {doc}`Roadmap <roadmap>` says what exists today and what is coming.
- The {doc}`API reference <api/index>` lists every public name.

```{toctree}
:maxdepth: 2
:caption: Guide

guide/index
```

```{toctree}
:maxdepth: 2
:caption: Textbook

textbook/index
```

```{toctree}
:maxdepth: 1
:caption: Roadmap

roadmap
```

```{toctree}
:maxdepth: 1
:caption: Releases

releases
```

```{toctree}
:maxdepth: 2
:caption: API

api/index
```
