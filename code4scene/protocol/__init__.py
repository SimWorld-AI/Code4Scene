"""The paper's scoring protocol, implemented exactly.

Modules
-------
``actor_f1``   Actor-level Repair F1 (image-to-scene ground-truth verifier).
``physics``    Physical Safety score (two fixed-weight leaves, 5 cm support).
``t2s``        Detailed Alignment, Overview Alignment and the T2S case score.
``i2s``        The I2S case score, 0.8 * Repair F1 + 0.2 * Physics.
``aggregate``  Setting means and the model score, 0.5 * S_T2S + 0.5 * S_I2S.
``constants``  Every frozen weight, tolerance, rounding rule and policy ID.
"""

from . import actor_f1, aggregate, constants, i2s, physics, t2s

__all__ = ["actor_f1", "aggregate", "constants", "i2s", "physics", "t2s"]
