from .base import BaseHead, build_head, register_head
from .cls_head import ClsHead, ClsHeadV1, ClsHeadV2, ClsHeadV3, cls_head
from .reg_head import RegHead, RegHeadV1, reg_head
from .emb_head import EmbHead, emb_head

__all__ = [
    # base
    "BaseHead",
    "build_head",
    "register_head",
    # classification
    "ClsHead",
    "ClsHeadV1",
    "ClsHeadV2",
    "ClsHeadV3",
    "cls_head",
    # regression
    "RegHead",
    "RegHeadV1",
    "reg_head",
    # embedding
    "EmbHead",
    "emb_head",
]
