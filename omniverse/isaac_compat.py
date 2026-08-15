"""Version-tolerant Isaac Sim imports.

Isaac Sim 4.5 renamed almost every extension from ``omni.isaac.*`` to
``isaacsim.*``. Both naming schemes are still in the wild — UTP lab machines
often run an older build than a personal install — and code written against one
fails with a bare ImportError on the other, which is a confusing first
experience.

Everything Isaac-specific is funnelled through this module so the rest of the
project imports one name and does not care which build is running.

Nothing here is exercised by the test suite: it cannot be, without Isaac Sim
installed. It is deliberately kept to imports and lookups so there is little to
go wrong.
"""

from __future__ import annotations

import sys

# Populated by `require_isaac()`.
ISAAC_VERSION_STYLE: str | None = None


class IsaacNotAvailable(RuntimeError):
    """Raised with actionable guidance rather than a bare ImportError."""


def _guidance() -> str:
    return (
        "Isaac Sim modules are not importable.\n"
        "\n"
        "This script must be run by Isaac Sim's own Python, not the system one.\n"
        "\n"
        "  Windows (Isaac Sim 4.x installed via Omniverse Launcher):\n"
        '    "%LOCALAPPDATA%\\ov\\pkg\\isaac-sim-4.2.0\\python.bat" run_isaac.py\n'
        "\n"
        "  Windows (pip install, Isaac Sim 4.5+):\n"
        "    python run_isaac.py        (in the venv where you pip-installed isaacsim)\n"
        "\n"
        "  Linux:\n"
        "    ~/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh run_isaac.py\n"
        "\n"
        "If you meant to run without Isaac Sim, use the offline demo instead:\n"
        "    python services/mapper/main.py --source sim\n"
    )


def start_simulation_app(headless: bool = False, width: int = 1280, height: int = 720):
    """Create the SimulationApp. MUST be called before any other omni import.

    Isaac Sim boots a full Kit application underneath. Importing ``omni.*``
    before that application exists fails in ways that look unrelated to the
    real cause, so this is always the first call in any standalone script.
    """
    try:
        try:
            from isaacsim import SimulationApp  # Isaac Sim 4.5+
        except ImportError:
            from omni.isaac.kit import SimulationApp  # Isaac Sim 4.0-4.2
    except ImportError as exc:
        raise IsaacNotAvailable(_guidance()) from exc

    return SimulationApp(
        {"headless": headless, "width": width, "height": height}
    )


def require_isaac() -> dict:
    """Import the Isaac APIs this project uses, under either naming scheme.

    Returns a dict of the symbols so callers do not repeat the try/except.
    Call only *after* `start_simulation_app`.
    """
    global ISAAC_VERSION_STYLE

    try:
        # Isaac Sim 4.5+
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation as Articulation
        from isaacsim.core.utils.nucleus import get_assets_root_path
        from isaacsim.core.utils.stage import add_reference_to_stage

        ISAAC_VERSION_STYLE = "isaacsim"
    except ImportError:
        try:
            # Isaac Sim 4.0 - 4.2
            from omni.isaac.core import World
            from omni.isaac.core.articulations import Articulation
            from omni.isaac.core.utils.nucleus import get_assets_root_path
            from omni.isaac.core.utils.stage import add_reference_to_stage

            ISAAC_VERSION_STYLE = "omni.isaac"
        except ImportError as exc:
            raise IsaacNotAvailable(_guidance()) from exc

    return {
        "World": World,
        "Articulation": Articulation,
        "get_assets_root_path": get_assets_root_path,
        "add_reference_to_stage": add_reference_to_stage,
        "style": ISAAC_VERSION_STYLE,
    }


def kaya_asset_paths(assets_root: str) -> list[str]:
    """Candidate paths for the built-in Kaya robot, newest layout first.

    NVIDIA Kaya is a three-wheel holonomic omni robot — the same configuration
    as this project's platform — and it ships with Isaac Sim with the omni
    rollers already modelled as individual rigid bodies. Starting from it
    avoids authoring roller physics by hand, which is by far the fiddliest part
    of simulating an omni wheel.

    NVIDIA moved the asset between releases, so several paths are tried.
    """
    return [
        f"{assets_root}/Isaac/Robots/NVIDIA/Kaya/kaya.usd",
        f"{assets_root}/Isaac/Robots/Kaya/kaya.usd",
        f"{assets_root}/Isaac/Robots/Kaya/kaya_realsense.usd",
    ]


def resolve_existing_asset(candidates: list[str]) -> str | None:
    """Return the first candidate that actually resolves, or None."""
    try:
        import omni.client
    except ImportError:
        return candidates[0] if candidates else None

    for path in candidates:
        try:
            result, _ = omni.client.stat(path)
            if result == omni.client.Result.OK:
                return path
        except Exception:
            continue
    return None


def find_joint_indices(articulation, joint_names: list[str]) -> list[int]:
    """Map joint names to DOF indices, with a useful error if one is missing.

    Isaac orders DOFs by its own traversal, not by the order they appear in the
    USD, so indices must always be looked up by name rather than assumed.
    """
    try:
        dof_names = list(articulation.dof_names)
    except AttributeError as exc:
        raise IsaacNotAvailable(
            "articulation has no dof_names; is the robot initialised? "
            "Call world.reset() before reading joint information."
        ) from exc

    indices = []
    for name in joint_names:
        if name not in dof_names:
            raise ValueError(
                f"joint {name!r} not found on the robot.\n"
                f"Available joints: {dof_names}\n"
                "Update WHEEL_JOINT_NAMES in run_isaac.py to match."
            )
        indices.append(dof_names.index(name))
    return indices


def print_environment_report() -> None:
    """Print what was detected. First thing to check when something is wrong."""
    print(f"  python           : {sys.version.split()[0]}")
    print(f"  isaac api style  : {ISAAC_VERSION_STYLE or 'not yet resolved'}")
    try:
        import omni.isaac.version as version_module

        print(f"  isaac sim        : {version_module.get_version()[0]}")
    except Exception:
        pass
