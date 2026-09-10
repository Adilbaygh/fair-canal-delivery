"""faircanal - lexicographic max-min water delivery on irrigation canals.

The package is built in the order the model requires: plant, delivery
operator, closed loop, network balance, lexicographic programme,
certificate. Each layer is only added once the layer below it is guarded by
a permanent test.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
