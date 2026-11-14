"""Warm-up input of the worker: openpi's own example observation for the robot family the policy was built for.

The family is the robot input transform among the policy's input transforms (LiberoInputs, DroidInputs or
AlohaInputs, as the openpi config defines them), so a custom config of a known family needs no extra setting. The
example is that family's make_*_example() helper from openpi, a fresh random observation on every call.
"""

from openpi.policies import aloha_policy
from openpi.policies import droid_policy
from openpi.policies import libero_policy

FAMILIES = (
    (libero_policy.LiberoInputs, "libero", libero_policy.make_libero_example),
    (droid_policy.DroidInputs, "droid", droid_policy.make_droid_example),
    (aloha_policy.AlohaInputs, "aloha", aloha_policy.make_aloha_example),
)


def family(transforms):
    """(name, make_example) for the first LIBERO, DROID or ALOHA input transform in `transforms`."""
    for transform in transforms:
        for cls, name, make_example in FAMILIES:
            if isinstance(transform, cls):
                return name, make_example
    raise ValueError(f"no LIBERO, DROID or ALOHA input transform in {[type(t).__name__ for t in transforms]}")
