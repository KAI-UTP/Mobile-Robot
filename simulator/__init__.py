"""Virtual robot and world for hardware-free development.

The exploration controllers used to live here too. They are now in `autonomy/`,
because they drive the real robot as well and a package named "simulator" was
the wrong home for the only code that decides where the hardware goes.
"""

from .virtual_robot import NoiseProfile, VirtualRobot, VirtualWorld, Wall

__all__ = [
    "NoiseProfile",
    "VirtualRobot",
    "VirtualWorld",
    "Wall",
]
