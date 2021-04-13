

from .build import META_ARCH_REGISTRY, build_model

# import all the meta_arch, so they will be registered
from .rcnn import GeneralizedRCNN

__all__ = list(globals().keys())