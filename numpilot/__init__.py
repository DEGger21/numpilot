"""NumPilot public Python API."""

from numpilot.operator_specs import OperatorSpec, operator_spec
from numpilot.model_gateway import (
    ModelGateway,
    ModelGatewayError,
    ModelRequest,
    ModelResult,
)


__version__ = "0.3.0a1.dev0"

__all__ = [
    "ModelGateway",
    "ModelGatewayError",
    "ModelRequest",
    "ModelResult",
    "OperatorSpec",
    "operator_spec",
    "__version__",
]
