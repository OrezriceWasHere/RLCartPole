from collections import namedtuple

Observation = namedtuple("Observation", [
    "cart_position", #
    "cart_velocity",
    "pole_angle",
    "pole_ang_velocity"
])
