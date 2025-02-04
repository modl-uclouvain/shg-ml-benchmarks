## This module contains functions dealing with the SHG tensor 'd'=========================================================

## Import necessary modules ===============================================================================================
import inspect
from typing import Union, get_args, get_origin

import numpy as np
import scipy.constants as cst


# Part of the enforcement the type hinting of the arguments ================================================================
def enforce(x, x_type, x_annot):
    if (x_type) is x_annot or (x_type) in get_args(x_annot):
        pass
    else:
        if get_origin(x_annot) is Union:
            raise Exception(
                f"{x} is of type {x_type} instead of types {get_args(x_annot)}"
            )
        else:
            raise Exception(f"{x} is of type {x_type} instead of type {x_annot}")

    return


# Returns d (numpy array or list) as an (numpy) array =====================================================================
# Throws an error if d is neither an array nor a list
def to_array(d: np.ndarray | list) -> np.ndarray:
    # Enforce the types of the arguments from annotations (generic section)
    args = inspect.getfullargspec(to_array).args
    annotations = inspect.getfullargspec(to_array).annotations
    for x in args:
        x_type = type(locals()[x])
        x_annot = annotations[x]
        enforce(x, x_type, x_annot)

    return np.array(d)


# Converts the atomic units of NLO susceptibility (as in Abinit) to SI units in pm/V ======================================
# Throws an error if d is neither an int, a float, a np.ndarray, nor a list
def au_to_pmV(d: int | float | np.ndarray | list) -> np.ndarray:
    from abipy.abilab import units as abu

    # Enforce the types of the arguments from annotations (generic section)
    args = inspect.getfullargspec(au_to_pmV).args
    annotations = inspect.getfullargspec(au_to_pmV).annotations
    for x in args:
        x_type = type(locals()[x])
        x_annot = annotations[x]
        enforce(x, x_type, x_annot)

    coef_au_to_pmV = (
        16
        * np.pi**2
        * abu.bohr_to_ang**2
        * 1e-8
        * cst.epsilon_0
        / cst.elementary_charge
    )

    return d * coef_au_to_pmV


# Returns True if the given list or array is in Voigt notation (shape 3x6)================================================
# Throws an Exception if d is neither 3x3x3 nor 3x6
def is_voigt(d: np.ndarray | list) -> np.ndarray:
    d = to_array(d)
    d_shape = d.shape
    if d_shape == (3, 6):
        return True
    elif d_shape == (3, 3, 3):
        return False
    else:
        raise Exception(
            "The tensor d is of shape {d_shape} although it should either be 3x6 (Voigt) or 3x3x3."
        )


# Returns the d tensor (list or array) in Voigt notation (array, 3x6)======================================================
# Throws an Exception if shape different than 3x6 (Voigt) or 3x3x3
def to_voigt(d: np.ndarray | list) -> np.ndarray:
    d = to_array(d)

    if is_voigt(d):
        return d
    else:
        d_new = np.zeros([3, 6])
        voigt = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]]

        for i in range(3):
            for j in range(6):
                d_new[i, j] = d[i, voigt[j][0], voigt[j][1]]

        return d_new


# Returns the d tensor (list or array) in non-Voigt notation (array, 3x3x3)===============================================
# Throws an Exception if shape different than 3x6 (Voigt) or 3x3x3
def from_voigt(d: np.ndarray | list) -> np.ndarray:
    d = to_array(d)

    if is_voigt(d):
        d_new = np.zeros([3, 3, 3])
        voigt = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    d_new[i, j, k] = d[i, int(voigt[j, k])]

        return d_new
    else:
        return d


# THIS VERSION IS NOT CORRECT, MY MISTAKE IN READING THE ORIGINAL EQUATION...
# Returns the dKP (d_KP) as defined in [Ref.1]===========================================================================
# Throws an Exception if shape different than 3x6 (Voigt) or 3x3x3
# Prev. named get_dKP
def get_dKP_old(d: np.ndarray | list) -> float:
    d = from_voigt(d)

    first = 0
    second = 0
    third = 0
    for i in range(3):
        first += (19 / 105) * d[i, i, i] ** 2
        for j in range(3):
            if j != i:
                second += (13 / 105) * d[i, i, i] * d[i, j, j]
                third += (44 / 105) * d[i, i, j] ** 2  # 14/105 in Francesco's thesis

    fourth = 0
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        # for i, j, k in ((0,1,2), (1,2,0), (2,0,1), (0,2,1), (1,0,2), (2,1,0)): # no, only three, stated in Cyvin 1965
        fourth += (13 / 105) * (d[i, i, j] * d[j, k, k] + (5 / 7) * d[i, j, k] ** 2)

    # print(first, second, third, fourth)

    return np.sqrt(first + second + third + fourth)


# THIS VERSION IS THE CORRECT ONE
def get_dKP_weird(d: np.ndarray | list) -> float:
    d = from_voigt(d)

    first = 0
    second = 0
    third = 0
    for i in range(3):
        first += (19 / 105) * d[i, i, i] ** 2
        for j in range(3):
            if j != i:
                second += (13 / 105) * d[i, i, i] * d[i, j, j]
                third += (44 / 105) * d[i, i, j] ** 2  # 14/105 in Francesco's thesis

    fourth = 0
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        fourth += (13 / 105) * (d[i, i, j] * d[j, k, k])
    fifth = (5 / 7) * (d[0, 1, 2] ** 2)

    # print(first, second, third, fourth)

    return np.sqrt(first + second + third + fourth + fifth)


def get_dKP(d: np.ndarray | list) -> float:
    return get_dKP_weird(d)


# Returns the dRMS (sqrt(1stinv/27)) =====================================================================================
def get_dRMS(d: np.ndarray | list) -> float:
    # Converts d to its non-Voigt form, Error if neither 3x6 nor 3x3x3 initially
    d = from_voigt(d)

    # Computes the 1st invariant of the 3rd-order 3D tensor
    inv1 = np.einsum("ijk,ijk", d, d)

    return np.sqrt(inv1 / 27)
