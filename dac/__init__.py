__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

from . import nn

# Speech inference uses only dac.nn's quantizers. Loading the training and codec
# helpers eagerly forced the much larger descript-audiotools dependency tree into
# deployments that never construct a DAC model. Keep those legacy exports when
# audiotools is installed, while allowing the inference-only worker to stay lean.
try:
    import audiotools
except ModuleNotFoundError:
    audiotools = None

if audiotools is not None:
    audiotools.ml.BaseModel.INTERN += ["dac.**"]
    audiotools.ml.BaseModel.EXTERN += ["einops"]

    from . import model
    from . import utils
    from .model import DAC
    from .model import DACFile
