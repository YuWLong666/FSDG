# -*- coding:utf-8 -*-

from .model import predictor
from .model import build_model
from .backbone_bases import build_backbone
from .model import CFDG

# from backbone import build_backbone
import models.backbone_bases as bkbn_bases
from .model import build_model

# ************************************************************************************************
# fine-grained feature extractor + fine-grained label predictor


def build_models(args):
    return build_model(args)


