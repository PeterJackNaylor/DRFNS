#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from skimage.morphology import dilation, erosion, square


def generate_wsl(ws):
    """
    Generates watershed line. In particular, useful for seperating object
    in ground thruth as they are labeled by different intergers.
    """
    se = square(3)
    ero = ws.copy()
    ero[ero == 0] = ero.max() + 1
    ero = erosion(ero, se)
    ero[ws == 0] = 0

    grad = dilation(ws, se) - ero
    grad[ws == 0] = 0
    grad[grad > 0] = 255
    grad = grad.astype(np.uint8)
    return grad