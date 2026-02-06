"""
Shared data types used across segmentation modules.
This module contains the common data structures to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import numpy as np


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    center: Optional[Tuple[int, int]] = None
    area: Optional[int] = None

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: np.ndarray
    logits: Optional[np.ndarray] = None

    def __post_init__(self):
        self.box.center = (
            (self.box.xmin + self.box.xmax) // 2,
            (self.box.ymin + self.box.ymax) // 2,
        )
        self.box.area = (self.box.xmax - self.box.xmin) * (
            self.box.ymax - self.box.ymin
        )

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
            mask=detection_dict.get("mask"),
            logits=detection_dict.get("logits"),
        )