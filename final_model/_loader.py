"""sys.path-isolation helper for loading the three sibling comparison projects.

All three projects (ball / court / bounce) ship identical top-level package
names (``data/``, ``models/``, ``evaluation/``, ``train/``). Importing more than
one in the same process would make Python's module cache serve whichever
project's ``data.preprocessing`` (etc.) was imported first to all of them.

``import_project`` isolates each project: it imports the requested modules with
the project root temporarily on ``sys.path``, then renames every *project-local*
module the import created (identified by its ``__file__`` living under the
project root) to a unique prefix in ``sys.modules``. Subsequent project imports
therefore miss the cache and load fresh from their own root. Third-party modules
pulled in as a side effect (torch, sklearn, ...) keep their real names and are
shared.

Because the renamed modules are no longer reachable under their plain names,
wrappers MUST capture every symbol they need at module-load time (right after
``import_project`` returns) and reference those captured symbols thereafter. A
lazy ``from data.preprocessing import ...`` inside a method would fail — the name
has been renamed away and the project root is no longer on ``sys.path``.
"""
import importlib
import os
import sys

# final_model/ sits directly under the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BALL_ROOT = os.path.join(REPO_ROOT, "tennis_ball_tracking_comparison")
COURT_ROOT = os.path.join(REPO_ROOT, "courtkeypoint_detection_comparison")
BOUNCE_ROOT = os.path.join(REPO_ROOT, "bounce_detection_comparison")


def import_project(project_root, imports, unique_prefix):
    """Import ``imports`` from ``project_root`` in isolation.

    Parameters
    ----------
    project_root : str
        Absolute path to the sibling project (its top-level packages live here).
    imports : str | list[str]
        Dotted module path(s) to import, e.g. ``"models.tracknetv4"`` or a list.
    unique_prefix : str
        Prefix applied to the project's modules in ``sys.modules`` afterwards,
        e.g. ``"_fm_ball"`` turns ``"models"`` into ``"_fm_ball.models"``.

    Returns
    -------
    dict[str, module]
        Maps each requested import name to its imported module object.
    """
    if isinstance(imports, str):
        imports = [imports]
    project_root = os.path.realpath(project_root)

    before = set(sys.modules)
    result = {}
    sys.path.insert(0, project_root)
    try:
        for name in imports:
            result[name] = importlib.import_module(name)
    finally:
        try:
            sys.path.remove(project_root)
        except ValueError:
            pass

    # Rename every module the import just added whose source lives under
    # project_root. Snapshot the new keys first (we mutate sys.modules below).
    prefix = project_root + os.sep
    for name in list(set(sys.modules) - before):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        if os.path.realpath(f).startswith(prefix):
            sys.modules[unique_prefix + "." + name] = mod
            del sys.modules[name]
    return result
