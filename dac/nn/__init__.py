from . import layers
from . import quantize

# Losses are training-only and depend on descript-audiotools. Avoid importing
# them in the inference worker, which only needs the quantizer implementation.
try:
    import audiotools  # noqa: F401
except ModuleNotFoundError:
    audiotools = None

if audiotools is not None:
    from . import loss
