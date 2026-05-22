"""Package init for mjlab_microduck.

Logging backend selection
-------------------------
By default we alias `trackio` as `wandb` in `sys.modules` so every
`import wandb` in mjlab / rsl_rl / our scripts ends up using the local-
only trackio backend.  This is convenient for local dev (no API key,
offline, fast) but doesn't support the wandb features that production
training relies on: checkpoint upload, `--wandb-run-path` resume from
a previous run, etc.

To use the *real* wandb client (matching the original microduck
workflow) prefix the command with the env var:

    MJLAB_MICRODUCK_LOGGER=wandb uv run train Mjlab-Velocity-Flat-MicroDuck-Sprung ...

That skips the aliasing entirely, so `import wandb` resolves to the
actual wandb package.  You'll need to have run `wandb login` once on
that machine.

Defaults to trackio for backward compatibility with the local-dev
workflow we've been using.  Set the env var to switch.
"""

import os as _os
import sys as _sys

# Resolve preference from env.  Anything other than "wandb" (or unset)
# routes through trackio.  Explicit "trackio" works too for clarity.
_logger_pref = _os.environ.get("MJLAB_MICRODUCK_LOGGER", "trackio").lower()

if _logger_pref != "wandb":
    try:
        import trackio as _trackio
    except ImportError:
        pass
    else:
        # rsl_rl (and any other wandb caller) passes kwargs that trackio's
        # corresponding functions don't accept:
        #   wandb.init(..., entity=...)          → trackio.init has no `entity`
        #   wandb.save(path, base_path=...)      → trackio.save has no `base_path`
        # Rather than maintain a per-function shim list, wrap each function so
        # any kwarg the underlying trackio fn doesn't recognize is silently
        # dropped.  Lets new wandb-isms slip through without crashing.
        import inspect as _inspect

        def _make_kw_filter(fn):
            valid = set(_inspect.signature(fn).parameters)
            def _shim(*args, **kwargs):
                for k in set(kwargs) - valid:
                    kwargs.pop(k)
                return fn(*args, **kwargs)
            _shim.__wrapped__ = fn
            return _shim

        for _name in ("init", "save", "log", "finish"):
            _fn = getattr(_trackio, _name, None)
            if _fn is not None:
                setattr(_trackio, _name, _make_kw_filter(_fn))

        _sys.modules.setdefault("wandb", _trackio)
