#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:51:40 2021

@author: ghiggi
"""
import pygsp
from typing import List, Dict
from abc import ABC, abstractmethod


##----------------------------------------------------------------------------.
class NowProject(ABC):
    """Define general NowProject model class."""

    @abstractmethod
    def forward(self, x):
        """Implement a forward pass."""
        pass


##----------------------------------------------------------------------------.
class UNetModel(NowProject):
    """Define general UNet class."""

    @abstractmethod
    def encode(self, *args, **kwargs):
        """Encode an input into a lower dimensional space."""
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        """Decode low dimensional data into a high dimensional space."""
        pass

    def forward(self, x):
        """Implement a forward pass."""
        x_encoded = self.encode(x)
        output = self.decode(*x_encoded)
        return output


##----------------------------------------------------------------------------.
class ConvNetModel(NowProject):
    """Define general ResNet class."""

    @abstractmethod
    def forward(self, x):
        """Implement a forward pass."""
        pass


##----------------------------------------------------------------------------.
class DownscalingNetModel(NowProject):
    """Define general DownscalingNet class."""

    @abstractmethod
    def decode(self, *args, **kwargs):
        """Decode low dimensional data into a high dimensional space."""
        pass

    def forward(self, x):
        """Implement a forward pass."""
        output = self.decode(x)
        return output
