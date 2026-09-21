# No easy on-ramp for arbitrary user code as a rate (summer2's `defer`)

**Status:** open, unscheduled. Prototyped against `721ad04`.
**Where:** `src/summer4/flows/rates.py` (`Transform`, `as_rate`);
`docs/cookbook/01-custom-rates.ipynb`; `docs/dev/rate-expressions.md`
(§ There is no good on-ramp).

## What is wrong today

summer2 shipped `computegraph.defer` — a two-line helper:

```python
def defer(func):
    def wrapped_maker(*args, **kwargs):
        return Function(func, args, kwargs)
    return wrapped_maker
```

A modeller writes an ordinary Python function, wraps it, and calls it with a
mix of parameters and literals. No class, no registration, no protocol. For
users who are not software developers this was the main door into the graph.

summer4 has the *capability* three times over but no comparable door:

- **`Transform` cannot occupy the rate slot.**
  `TransitionFlow(..., rate=Transform(f, Time(), Param("amp")))` raises
  `TypeError: Cannot use Transform as a flow rate` — it is an `Adjustment`, not
  a `RateOps`. The working incantation is a dummy rate plus a callable that
  discards its first argument:

  ```python
  TransitionFlow("r", state["I"], state["R"], 1.0,
                 adjust=[Transform(lambda prev, t, amp: f(t, amp),
                                   Time(), Param("amp"))])
  ```

- **`Transform` is documented only as a precedence level.** It appears in
  `docs/user/08-flows.ipynb` and `docs/user/07-from-summer2.md` solely in the
  ordering rule `Overwrite` (0) → `Multiply` (1) → `Transform` (2). No page
  presents it as the place to put arbitrary code.
- **The `prev` first parameter is unmotivated** for a user who wants a rate
  rather than an adjustment to one.
- **`derived_fn` is heavier than the job** — a `NamedTuple` schema plus a
  `(params, *, y, t)` hook — and per R3 it disables parameter hoisting
  model-wide.
- **The cookbook ladder skips the rung.** `docs/cookbook/01-custom-rates.ipynb`
  goes `Reduce` arithmetic → FOI-specific `kind=` callable → subclass
  `RateOps` + `register_rate_eval`. Nothing sits between rungs 1 and 3.

## Why a `Defer` node is the right fix, not a wrapper over `Transform`

Because `defer`'s dependencies arrive as **explicit arguments**, a `Defer` node
is fully analysable — the objection that a generic callable node defeats
staging does not apply. Prototype, ~35 lines including `__rate_stage__`,
`__rate_bytes__`, `__field_paths__` and `__flow_refs__`:

```python
@register_rate_eval(Defer)
def _eval_defer(expr, *, eval_child, **_):
    return expr.fn(*[eval_child(a) for a in expr.args])
```

with `__rate_stage__` returning `"step"` if any argument is step, else `"run"`.

Verified:

- `defer(my_code)(Time(), Param("amp"))` used as a flow rate evaluates
  identically to the `Transform` workaround above.
- `defer(f)(Param("x"), Param("y"))` classifies `run` **and hoists** —
  `build_hoist_table` gives it a slot.
- `defer(f)(Time(), Param("amp"))` classifies `step`.

So the generic node is not a staging regression. The genuine loss versus a
typed node is **sub-node** hoisting: `Interp` hoists its breakpoint stack and
value stack independently of its argument, whereas a `Defer` wrapping the same
interpolation with a `Time()` argument is all-or-nothing and rebuilds the knot
stack every step. That is a reason to keep `Interp`, not a reason to withhold
`Defer`.

## Done when

- A `Defer` rate node (and a `defer(fn)` curry) is public, usable in the rate
  slot of every flow type, with the four dunders implemented.
- The digest keys on `id(fn)` — fail-safe, a redundant recompile — with an
  optional `name=` so users who rebuild models in a loop can opt into a stable
  key. Do **not** key on `fn.__qualname__` alone: every lambda is `<lambda>`,
  which would collide and fail *unsafe*, the same trap as
  `custom-rate-node-digest-collision.md`.
- `docs/cookbook/01-custom-rates.ipynb` gains it as rung 2, between `Reduce`
  arithmetic and the FOI `kind=` callable, framed for a modeller rather than a
  package author.
- `docs/user/07-from-summer2.md` maps `computegraph.defer` → `summer4.defer`.

## Related

- `docs/dev/rate-expressions.md` — the wider design rationale; this note is the
  ergonomics half of it.
- `custom-rate-node-digest-collision.md` — the digest trap any new
  callable-carrying node must avoid.
- `derived-fn-blocks-hoisting.md` — why `derived_fn` is not the answer here.
