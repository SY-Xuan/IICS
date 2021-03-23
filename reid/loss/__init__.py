from __future__ import absolute_import

from .oim import oim, OIM, OIMLoss
from .triplet import TripletLoss
from .label_smoothing import LabelSmoothing

__all__ = [
    'oim',
    'OIM',
    'OIMLoss',
    'TripletLoss',
    'LabelSmoothing',
]
