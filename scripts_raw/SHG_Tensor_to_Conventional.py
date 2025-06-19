'''Script with functions to verify the symmetry of SHG tensors and to find its conventional form (or at least try)'''
'''
python SHG_Tensor_to_Conventional.py --file df_outputs.json.gz --action abisym
python SHG_Tensor_to_Conventional.py --file df_abisym.json.gz --action findconv
'''

from copy import copy
from pathlib import Path
from plotly.offline import iplot, init_notebook_mode
from plotly.graph_objs import Mesh3d
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure, Lattice
from pymatgen.core.tensors import Tensor, SquareTensor, get_uvec
from pymatgen.core.operations import SymmOp
from abipy.core.structure import Structure as AbiStructure

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go


import SHG_Tensor_Func as shg

# ======================================================================================================
# Various functions

def verify_kleinman(d):
    # Work with tensor in non-Voigt form
    dijk = shg.from_voigt(d)

    # Kleinman implies that all arrangements of i,j, and k are equivalent
    klmn_equival = ["ijk", "kij", "jki", "ikj", "jik", "kji"]
    # Meaning there are only 10 independent components 
    # The -1 at the end is there to treat those combination as array indices
    klmn_indpt = np.array([ [1,1,1], [1,2,2], [1,3,3], [1,2,3], [1,1,3], [1,1,2],
                            [2,2,2], [2,3,3], [2,2,3],
                            [3,3,3]])-1
    
    # Iterate over the independent components
    for idx in klmn_indpt:
        ref = dijk[idx[0], idx[1], idx[2]]
        # Iterate over the components that should be equivalent to the reference under Kleinman
        for tmp in klmn_equival:
            str = tmp.replace("i", f"{idx[0]}").replace("j", f"{idx[1]}").replace("k", f"{idx[2]}")
            idx_equival = [int(str[0]),int(str[1]),int(str[2])]
            compared = dijk[idx_equival[0], idx_equival[1], idx_equival[2]]

            if not math.isclose(ref, compared, abs_tol=1e-09) or (np.sign(ref*compared)==-1 and ref>=1e-09):
                print(f"The SHG tensor does not respect Kleinman symmetry due to the following components\
                      being different: d{idx}={ref} and d{idx_equival}={compared}")
                return False

    return True


def create_kleinman_tensor(d11=0, d12=0, d13=0, d14=0, d15=0, d16=0, d22=0, d23=0, d24=0, d33=0, as_voigt=True):
    d = np.array([  [d11, d12, d13, d14, d15, d16],
                    [d16, d22, d23, d24, d14, d12],
                    [d15, d24, d33, d23, d13, d14]
                  ])

    if as_voigt:
        return d

    return shg.from_voigt(d)


def pg_nb_to_symbol(pg_symbol=None, pg_nb=None):
    pg_symbols_HM = ["1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm", "3", 
                     "-3", "32", "3m", "-3m", "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm", "23", "m-3", "432", "-43m", 
                     "m-3m"]
    cs_pg_symbols_HM = ["-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m"]

    if not pg_nb and not pg_symbol:
        raise Exception("No argument has been provided.")
    if pg_nb and pg_symbol:
        raise Exception("Please provide either the point group number or its symbol, not both.")
    if pg_symbol and pg_symbol not in pg_symbols_HM:
        raise Exception(f"The point group symbol {pg_symbol} provided as argument is not valid.")

    if pg_nb:
        pg = pg_symbols_HM[pg_nb-1]
    else:
        pg = pg_symbol

    if pg in cs_pg_symbols_HM:
        pg = "cs_pg"

    return pg


def get_mask_shg_kleinman_by_pg(pg_nb=None, pg_symbol=None, keep_null=False, as_voigt=True, spg_symbol=None):
    pg = pg_nb_to_symbol(pg_symbol=pg_symbol, pg_nb=pg_nb)

    shg_shape_by_pg = {
        "1":    {"d11": 1, "d12": 1, "d13": 1, "d14": 1, "d15": 1, "d16": 1, "d22": 1, "d23": 1, "d24": 1, "d33": 1},
        "2":    {"d11": 0, "d12": 0, "d13": 0, "d14": 1, "d15": 0, "d16": 1, "d22": 1, "d23": 1, "d24": 0, "d33": 0},
        "m":    {"d11": 1, "d12": 1, "d13": 1, "d14": 0, "d15": 1, "d16": 0, "d22": 0, "d23": 0, "d24": 1, "d33": 1},
        "222":  {"d11": 0, "d12": 0, "d13": 0, "d14": 1, "d15": 0, "d16": 0, "d22": 0, "d23": 0, "d24": 0, "d33": 0},
        "mm2":  {"d11": 0, "d12": 0, "d13": 0, "d14": 0, "d15": 1, "d16": 0, "d22": 0, "d23": 0, "d24": 1, "d33": 1},
        "4":    {"d11": 0, "d12": 0, "d13": 0, "d14": 0, "d15": 1, "d16": 0, "d22": 0, "d23": 0, "d24": 1, "d33": 1},
        "-4":   {"d11": 0, "d12": 0, "d13": 0, "d14": 1, "d15": 1, "d16": 0, "d22": 0, "d23": 0, "d24": 1, "d33": 0},
        "422":  {"d11": 0, "d12": 0, "d13": 0, "d14": 0, "d15": 0, "d16": 0, "d22": 0, "d23": 0, "d24": 0, "d33": 0},
        "4mm":  {"d11": 0, "d12": 0, "d13": 0, "d14": 0, "d15": 1, "d16": 0, "d22": 0, "d23": 0, "d24": 1, "d33": 1},
        "-42m": {"d11": 0, "d12": 0, "d13": 0, "d14": 1, "d15": 0, "d16": 0, "d22": 0, "d23": 0, "d24": 0, "d33": 0},
        "3":    {"d11": 1, "d12": 1, "d13": 0, "d14": 0, "d15": 1, "d16": 1, "d22": 1, "d23": 0, "d24": 1, "d33": 1},
        "32":   {"d11": 1, "d12": 1, "d13": 0, "d14": 0, "d15": 0, "d16": 0, "d22": 0, "d23": 0, "d24": 0, "d33": 0},
        "3m":   {"d11": 0, "d12": 0, "d13": 0, "d14": 0, "d15": 1, "d16": 1, "d22": 1, "d23": 0, "d24": 1, "d33": 1},
        "6":    {"d11": 0, "d12": 0, "d13": 0, "d14": 0, "d15": 1, "d16": 0, "d22": 0, "d23": 0, "d24": 1, "d33": 1},
        "-6":   {"d11": 1, "d12": 1, "d13": 0, "d14": 0, "d15": 0, "d16": 1, "d22": 1, "d23": 0, "d24": 0, "d33": 0},
        "622":  {"d11": 0, "d12": 0, "d13": 0, "d14": 0, "d15": 0, "d16": 0, "d22": 0, "d23": 0, "d24": 0, "d33": 0},
        "6mm":  {"d11": 0, "d12": 0, "d13": 0, "d14": 0, "d15": 1, "d16": 0, "d22": 0, "d23": 0, "d24": 1, "d33": 1},
        "-6m2": {"d11": 0, "d12": 0, "d13": 0, "d14": 0, "d15": 0, "d16": 1, "d22": 1, "d23": 0, "d24": 0, "d33": 0},
        "23":   {"d11": 0, "d12": 0, "d13": 0, "d14": 1, "d15": 0, "d16": 0, "d22": 0, "d23": 0, "d24": 0, "d33": 0},
        "432":  {"d11": 0, "d12": 0, "d13": 0, "d14": 0, "d15": 0, "d16": 0, "d22": 0, "d23": 0, "d24": 0, "d33": 0},
        "-43m": {"d11": 0, "d12": 0, "d13": 0, "d14": 1, "d15": 0, "d16": 0, "d22": 0, "d23": 0, "d24": 0, "d33": 0},
        "cs_pg": {"d11": 0, "d12": 0, "d13": 0, "d14": 0, "d15": 0, "d16": 0, "d22": 0, "d23": 0, "d24": 0, "d33": 0},
    }

    mask = create_kleinman_tensor(**shg_shape_by_pg[pg], as_voigt=as_voigt)

    # Handle spg which have axes inverted wrt usual pg (to fill when I encounter entries like this)
    if spg_symbol and pg=='32' and spg_symbol in ["P312", "P3_112", "P3_212"]:
        mask = np.abs(np.round(shg.to_voigt(shg.apply_rot(mask, 0, 0, 90))))
    elif spg_symbol and pg=='-42m' and spg_symbol in ["I-4m2", "P-4m2", "I-4c2",]:
        mask = np.abs(np.round(shg.to_voigt(shg.apply_rot(mask, 0, 0, 45))))
    elif spg_symbol and pg=='3m' and spg_symbol in ["P31m", "P31c"]:
        mask = np.abs(np.round(shg.to_voigt(shg.apply_rot(mask, 0, 0, 90))))
    elif spg_symbol and pg=='-6m2' and spg_symbol in ["P-62m", "P-62c"]:
        mask = np.abs(np.round(shg.to_voigt(shg.apply_rot(mask, 0, 0, 90))))

    # If one wants a mask to keep the components that are supposed to be zero
    if keep_null:
        # mask =  ~create_kleinman_tensor(**shg_shape_by_pg[pg], as_voigt=as_voigt)+2
        mask = ~mask.astype(np.int32) + 2

    return mask

# Return True if the non-null independent components are properly related by neumann principle for each point group
# This requires the SHG tensor to adopt the conventional form.
def verify_shg_neumann_by_pg(d, pg_nb=None, pg_symbol=None, abs_tol=1e-1, rel_tol_coef=0.05, spg_symbol=None):
    pg = pg_nb_to_symbol(pg_symbol=pg_symbol, pg_nb=pg_nb)

    dijk = shg.to_voigt(d)

    # Handle spg which have axes inverted wrt usual pg (to fill when I encounter entries like this)
    if spg_symbol in ["P312", "P3_112", "P3_212"]:
        dijk = shg.to_voigt(shg.apply_rot(dijk, 0, 0, 90))
    elif spg_symbol in ["I-4m2", "P-4m2", "I-4c2",]:
        dijk = shg.to_voigt(shg.apply_rot(dijk, 0, 0, 45))
    elif spg_symbol in ["P31m", "P31c"]:
        dijk = shg.to_voigt(shg.apply_rot(dijk, 0, 0, 90))
    elif spg_symbol in ["P-62m", "P-62c"]:
        dijk = shg.to_voigt(shg.apply_rot(dijk, 0, 0, 90))

    d11 =  dijk[1-1,1-1] ; d12 =  dijk[1-1,2-1] ; d13 =  dijk[1-1,3-1] ; d14 =  dijk[1-1,4-1] ; d15 =  dijk[1-1,5-1]
    d16 =  dijk[1-1,6-1] ; d22 =  dijk[2-1,2-1] ; d23 =  dijk[2-1,3-1] ; d24 =  dijk[2-1,4-1] ; d33 =  dijk[3-1,3-1]

    if pg in ["1", "2", "m", "222", "mm2", "422", "-42m", "622", "23", "432", "-43m", "cs_pg"]:
        # No constraints are imposed on the independent components for these pg
        tmp = [True]
    elif pg in ["4", "4mm", "6", "6mm"]:
        if rel_tol_coef:
            rel_tol = rel_tol_coef*np.max([np.abs(d15), np.abs(d24)])
        else:
            rel_tol = 1e-9 # default value of math.isclose
        tmp = [math.isclose(np.abs(d15), np.abs(d24), abs_tol=abs_tol, rel_tol=rel_tol)]
    elif pg=="-4":
        if rel_tol_coef:
            rel_tol = rel_tol_coef*np.max([np.abs(d15), np.abs(d24)])
        else:
            rel_tol = 1e-9 # default value of math.isclose
        tmp = [math.isclose(d15, -d24, abs_tol=abs_tol, rel_tol=rel_tol),]
    elif pg in ["3"]:
        if rel_tol_coef:
            rel_tol_1 = rel_tol_coef*np.max([np.abs(d16), np.abs(d22)])
            rel_tol_2 = rel_tol_coef*np.max([np.abs(d11), np.abs(d12)])
            rel_tol_3 = rel_tol_coef*np.max([np.abs(d15), np.abs(d24)])
        else:
            rel_tol_1 = 1e-9 # default value of math.isclose
            rel_tol_2 = 1e-9 # default value of math.isclose
            rel_tol_3 = 1e-9 # default value of math.isclose
        tmp = [math.isclose(d16, -d22, abs_tol=abs_tol, rel_tol=rel_tol_1),
               math.isclose(d11, -d12, abs_tol=abs_tol, rel_tol=rel_tol_2),
               math.isclose(np.abs(d15), np.abs(d24), abs_tol=abs_tol, rel_tol=rel_tol_3)]
    elif pg in ["-6"]:
        if rel_tol_coef:
            rel_tol_1 = rel_tol_coef*np.max([np.abs(d16), np.abs(d22)])
            rel_tol_2 = rel_tol_coef*np.max([np.abs(d11), np.abs(d12)])
        else:
            rel_tol_1 = 1e-9 # default value of math.isclose
            rel_tol_2 = 1e-9 # default value of math.isclose
        tmp = [math.isclose(d16, -d22, abs_tol=abs_tol, rel_tol=rel_tol_1),
               math.isclose(d11, -d12, abs_tol=abs_tol, rel_tol=rel_tol_2)]
    elif pg=="32":
        if rel_tol_coef:
            rel_tol = rel_tol_coef*np.max([np.abs(d11), np.abs(d12)])
        else:
            rel_tol = 1e-9 # default value of math.isclose
        tmp = [math.isclose(d11, -d12, abs_tol=abs_tol, rel_tol=rel_tol),]
    elif pg=="3m":
        if rel_tol_coef:
            rel_tol_1 = rel_tol_coef*np.max([np.abs(d16), np.abs(d22)])
            rel_tol_2 = rel_tol_coef*np.max([np.abs(d15), np.abs(d24)])
        else:
            rel_tol_1 = 1e-9 # default value of math.isclose
            rel_tol_2 = 1e-9 # default value of math.isclose
        tmp = [math.isclose(d16, -d22, abs_tol=abs_tol, rel_tol=rel_tol_1),
               math.isclose(np.abs(d15), np.abs(d24), abs_tol=abs_tol, rel_tol=rel_tol_2)]
    elif pg=="-6m2":
        if rel_tol_coef:
            rel_tol = rel_tol_coef*np.max([np.abs(d16), np.abs(d22)])
        else:
            rel_tol = 1e-9 # default value of math.isclose
        tmp = [math.isclose(d16, -d22, abs_tol=abs_tol, rel_tol=rel_tol),]
    else:
        raise Exception(f"The point group symbol {pg} is unknown...")

    return all(tmp)


def equate_coef(d1, d2):
    d_com = np.mean([np.abs(d1), np.abs(d2)])
    return np.sign(d1)*d_com, np.sign(d2)*d_com

def correct_shg_neumann_by_pg(d, pg_nb=None, pg_symbol=None, spg_symbol=None):
    pg = pg_nb_to_symbol(pg_symbol=pg_symbol, pg_nb=pg_nb)

    dijk = shg.to_voigt(d)

    # Handle spg which have axes inverted wrt usual pg (to fill when I encounter entries like this)
    if spg_symbol in ["P312", "P3_112", "P3_212"]:
        dijk = shg.to_voigt(shg.apply_rot(dijk, 0, 0, 90))
    elif spg_symbol in ["I-4m2", "P-4m2", "I-4c2",]:
        dijk = shg.to_voigt(shg.apply_rot(dijk, 0, 0, 45))
    elif spg_symbol in ["P31m", "P31c"]:
        dijk = shg.to_voigt(shg.apply_rot(dijk, 0, 0, 90))
    elif spg_symbol in ["P-62m", "P-62c"]:
        dijk = shg.to_voigt(shg.apply_rot(dijk, 0, 0, 90))

    d11 =  dijk[1-1,1-1] ; d12 =  dijk[1-1,2-1] ; d13 =  dijk[1-1,3-1] ; d14 =  dijk[1-1,4-1] ; d15 =  dijk[1-1,5-1]
    d16 =  dijk[1-1,6-1] ; d22 =  dijk[2-1,2-1] ; d23 =  dijk[2-1,3-1] ; d24 =  dijk[2-1,4-1] ; d33 =  dijk[3-1,3-1]

    if pg in ["1", "2", "m", "222", "mm2", "422", "-42m", "622", "23", "432", "-43m", "cs_pg"]:
        # No constraints are imposed on the independent components for these pg
        tmp = [True]
    elif pg in ["4", "4mm", "6", "6mm"]:
        d15, d24 = equate_coef(d15, d24)
    elif pg=="-4":
        d15, d24 = equate_coef(d15, d24)
    elif pg in ["3"]:
        d16, d22 = equate_coef(d16, d22)
        d11, d12 = equate_coef(d11, d12)
        d15, d24 = equate_coef(d15, d24)
    elif pg in ["-6"]:
        d16, d22 = equate_coef(d16, d22)
        d11, d12 = equate_coef(d11, d12)
    elif pg=="32":
        d11, d12 = equate_coef(d11, d12)
    elif pg=="3m":
        d16, d22 = equate_coef(d16, d22)
        d15, d24 = equate_coef(d15, d24)
    elif pg=="-6m2":
        d16, d22 = equate_coef(d16, d22)
    else:
        raise Exception(f"The point group symbol {pg} is unknown...")

    d_constrained = np.array([
        [d11, d12, d13, d14, d15, d16],
        [d16, d22, d23, d24, d14, d12],
        [d15, d24, d33, d23, d13, d14],
    ])

    # Handle spg which have axes inverted wrt usual pg (to fill when I encounter entries like this)
    if spg_symbol in ["P312", "P3_112", "P3_212"]:
        d_constrained = shg.to_voigt(shg.apply_rot(d_constrained, 0, 0, -90))
    elif spg_symbol in ["I-4m2", "P-4m2", "I-4c2",]:
        d_constrained = shg.to_voigt(shg.apply_rot(d_constrained, 0, 0, -45))
    elif spg_symbol in ["P31m", "P31c"]:
        d_constrained = shg.to_voigt(shg.apply_rot(d_constrained, 0, 0, -90))
    elif spg_symbol in ["P-62m", "P-62c"]:
        d_constrained = shg.to_voigt(shg.apply_rot(d_constrained, 0, 0, -90))

    return d_constrained


def get_max_null_component(d, pg_symbol=None, pg_nb=None, spg_symbol=None):
    mask_null = get_mask_shg_kleinman_by_pg(pg_nb=pg_nb, pg_symbol=pg_symbol, keep_null=True, spg_symbol=spg_symbol)
    d_null = shg.to_voigt(d) * mask_null
    return np.max(np.abs(d_null))

def get_sum_full_component(d, pg_symbol=None, pg_nb=None, spg_symbol=None):
    mask_full = get_mask_shg_kleinman_by_pg(pg_nb=pg_nb, pg_symbol=pg_symbol, keep_null=False, spg_symbol=spg_symbol)
    d_full = shg.to_voigt(d) * mask_full
    return np.sum(np.abs(d_full))

def get_full_d(d, pg_symbol=None, pg_nb=None, spg_symbol=None):
    mask_full = get_mask_shg_kleinman_by_pg(pg_nb=pg_nb, pg_symbol=pg_symbol, keep_null=False, spg_symbol=spg_symbol)
    return shg.to_voigt(d) * mask_full

def get_min_full_component(d, pg_symbol=None, pg_nb=None, spg_symbol=None):
    mask_full = get_mask_shg_kleinman_by_pg(pg_nb=pg_nb, pg_symbol=pg_symbol, keep_null=False, spg_symbol=spg_symbol)
    d_full = shg.to_voigt(d) * mask_full
    min_d_full = np.ma.masked_equal(np.abs(d_full), 0.0, copy=False).min()
    if isinstance(min_d_full, np.ma.core.MaskedConstant):
        return 0.0
    return min_d_full

def get_max_full_component(d, pg_symbol=None, pg_nb=None, spg_symbol=None):
    mask_full = get_mask_shg_kleinman_by_pg(pg_nb=pg_nb, pg_symbol=pg_symbol, keep_null=False, spg_symbol=spg_symbol)
    d_full = shg.to_voigt(d) * mask_full
    idx_arg_max = np.unravel_index(np.argmax(np.abs(d_full)), d_full.shape)
    max_d_full = d_full[idx_arg_max]
    return idx_arg_max, max_d_full

# https://stackoverflow.com/questions/58065055/floor-and-ceil-with-number-of-decimals
def my_floor(a, precision=2):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

def verify_shg_conventional_shape_by_pg(d, pg_nb=None, pg_symbol=None, tol_max_null = 1e-09, tol_rel_null_full=1e-04, verify_neumann=True, spg_symbol=None):
    is_kleinman = verify_kleinman(d)
    if not is_kleinman:
        # print("bc of kleinman")
        return is_kleinman, "Kleinman"

    min_d_full = get_min_full_component(d=d, pg_symbol=pg_symbol, pg_nb=pg_nb, spg_symbol=spg_symbol)
    if math.isclose(min_d_full,0):
        # print(f"bc of min_d_full: {min_d_full}")
        return False, f"{min_d_full = }"

    max_d_null = get_max_null_component(d, pg_symbol=pg_symbol, pg_nb=pg_nb, spg_symbol=spg_symbol)
    
    if max_d_null>=tol_max_null and my_floor(max_d_null/min_d_full, precision=str(tol_rel_null_full)[::-1].find('.'))>tol_rel_null_full:
        # print(f"The largest of the components supposed to be null is {max_d_null} instead of being close to 0 and is greater than 1% of the minimum value of the non-null components.")
        return False, f"{max_d_null/min_d_full = }"

    if verify_neumann:
        is_neumann = verify_shg_neumann_by_pg(d, pg_nb=pg_nb, pg_symbol=pg_symbol, spg_symbol=spg_symbol)
        if not is_neumann:
            # print(f"The given SHG tensor does not respect neumann symmetry.")
            return is_neumann, "neumann"

    return True, None


def apply_all_rot_get_max_null_and_min_full( d, 
                                step      = 10, 
                                minangle_a  = 0,
                                maxangle_a  = 120,
                                minangle_b  = 0,
                                maxangle_b  = 120,
                                minangle_g  = 0,
                                maxangle_g  = 120,
                                pg_symbol = None,
                                pg_nb     = None
                        ):

    # Create the arrays of rotational angles around each axis, from 0 to maxangle included with a step of step
    valpha    = np.arange(minangle_a,maxangle_a+0.5,step)
    vbeta     = np.arange(minangle_b,maxangle_b+0.5,step)
    vgamma    = np.arange(minangle_g,maxangle_g+0.5,step)
    l = len(valpha)

    # Instantiate two arrays with the dKP and dijk for each rotation respectively
    max_null_ar  = np.zeros([l,l,l])
    min_full_ar  = np.zeros([l,l,l])
    
    for ia, alpha in enumerate(valpha):
        for ib, beta in enumerate(vbeta):
            for ig, gamma in enumerate(vgamma):
                d_rot = shg.apply_rot(d, alpha, beta, gamma)
                max_null_ar[ia, ib, ig] = get_max_null_component(d_rot, pg_symbol=pg_symbol, pg_nb=pg_nb)
                min_full_ar[ia, ib, ig] = get_min_full_component(d_rot, pg_symbol=pg_symbol, pg_nb=pg_nb)
    
    
    return valpha,vbeta,vgamma,max_null_ar, min_full_ar

def apply_all_rot_get_max_null( d, 
                                step      = 10, 
                                minangle_a  = 0,
                                maxangle_a  = 120,
                                minangle_b  = 0,
                                maxangle_b  = 120,
                                minangle_g  = 0,
                                maxangle_g  = 120,
                                pg_symbol = None,
                                pg_nb     = None
                        ):

    # Create the arrays of rotational angles around each axis, from 0 to maxangle included with a step of step
    valpha    = np.arange(minangle_a,maxangle_a+0.5,step)
    vbeta     = np.arange(minangle_b,maxangle_b+0.5,step)
    vgamma    = np.arange(minangle_g,maxangle_g+0.5,step)
    l = len(valpha)

    # Instantiate two arrays with the dKP and dijk for each rotation respectively
    max_null_ar  = np.zeros([l,l,l])
    
    for ia, alpha in enumerate(valpha):
        for ib, beta in enumerate(vbeta):
            for ig, gamma in enumerate(vgamma):
                d_rot = shg.apply_rot(d, alpha, beta, gamma)
                max_null_ar[ia, ib, ig] = get_max_null_component(d_rot, pg_symbol=pg_symbol, pg_nb=pg_nb)
    
    
    return valpha,vbeta,vgamma,max_null_ar


def minimize_max_null(  d,
                        pg_symbol = None,
                        pg_nb     = None,
                        step=5,
                        minangle_a  = 0,
                        maxangle_a  = 120,
                        minangle_b  = 0,
                        maxangle_b  = 120,
                        minangle_g  = 0,
                        maxangle_g  = 120,
                        ):

    valpha,vbeta,vgamma,max_null_ar = apply_all_rot_get_max_null(   d,
                                                                    step=step,
                                                                    minangle_a  = minangle_a,
                                                                    maxangle_a  = maxangle_a,
                                                                    minangle_b  = minangle_b,
                                                                    maxangle_b  = maxangle_b,
                                                                    minangle_g  = minangle_g,
                                                                    maxangle_g  = maxangle_g,
                                                                    pg_symbol=pg_symbol,
                                                                    pg_nb=pg_nb)
    idx_arg_min = np.unravel_index(np.argmin(max_null_ar), max_null_ar.shape)
    min_a = valpha[idx_arg_min[0]]
    min_b = vbeta[idx_arg_min[1]]
    min_g = vgamma[idx_arg_min[2]]
    min_max_null = max_null_ar.min()

    return min_a, min_b, min_g, min_max_null


def switch_sign_max_full(d, pg_symbol=None, pg_nb=None):
    pg = pg_nb_to_symbol(pg_symbol=pg_symbol, pg_nb=pg_nb)
    if pg == "cs_pg":
        raise Exception(f"The point group {pg} is centrosymmetric... d_ijk should all be zero.")
    idx_arg_max, max_d_full = get_max_full_component(d, pg_symbol=pg)
    max_d_null = get_max_null_component(d, pg_symbol=pg)

    if np.sign(max_d_full)==1:
        return (0.0, 0.0, 0.0, max_d_full)

    for a, b, g in [(180.0, 0.0, 0.0),      (0.0, 180.0, 0.0),      (0.0, 0.0, 180.0),
                    (180.0, 180.0, 0.0),    (180.0, 0.0, 180.0),    (0.0, 180.0, 180.0),
                    (90.0, 0.0, 0.0),      (0.0, 90.0, 0.0),      (0.0, 0.0, 90.0),
                    (90.0, 90.0, 0.0),    (90.0, 0.0, 90.0),    (0.0, 90.0, 90.0),
                    ]:
        d_rot = shg.apply_rot(d, a, b, g)
        max_d_full_rot = shg.to_voigt(d_rot)[idx_arg_max]
        if np.sign(max_d_full_rot)==1:
            if not math.isclose(max_d_full_rot, np.abs(max_d_full), abs_tol=1e-06):
                raise Exception(f"Once rotated to switch the sign of the max. of the abs. values of dijk to +, the latter is different than originally: {max_d_full_rot}!={max_d_full}")
            max_d_rot_null = get_max_null_component(d_rot, pg_symbol=pg)
            if math.isclose(max_d_null, max_d_rot_null, abs_tol=1e-3):
                return (a, b, g, max_d_full_rot)

    print(pg_symbol)
    print(d)
    print(max_d_full)
    raise Exception(f"The sign of the maximum of the absolute values of the dijk could not be switched to positive...")


def apply_rot_by_R(d, R):

    # Define the order of the tensor to rotate
    ndim = len(d.shape)
        
    # Equivalent to R @ d @ R.T but generalized to any order of tensor
    # Rotate d: use np.einsum to multiply d by R in a for-loop over the order of tensor to target each dimension
    subs = 'ijklmnop'[:ndim]
    eins = 'Zz,{}->{}'
    for n in range(ndim): # or reversed(range(ndim)), does not matter
        eins_n = eins.format(subs.replace(subs[n],'z'),subs.replace(subs[n],'Z')) # 'Zz,ijz->ijZ' then 'Zz,izk->iZk' then 'Zz,zjk->Zjk' for ndim=3
        d = np.einsum(eins_n,R,d)
    
    return d

# Applies a Euler rotation to a system of vectors such as rprim e.g. (each row of syst is a vector to be rotated) ===============
# Rotation of alpha around x, then beta around new y, then gamma around new z (angles in degrees!)
# Throws an Exception/Error if the arguments are not of the same type as the annotations
# syst can have any number of row but each row must be 3 elements long
def apply_rot_syst_by_R(syst,
                        R
                        ):

    # Decomposes the rotation by rotating each vector (1st order tensor) individually 
    syst = shg.to_array(syst)
    syst_rot = np.zeros(np.shape(syst))
    for i in range(len(syst)):
        syst_rot[i] = apply_rot_by_R(syst[i],R)

    return syst_rot


# Plot an ellipsoid with semi-axis length a along x, b (y), and c (z) and then rotate this ellispoid by (alpha, beta, gamma)
def plot_ellipsoid(diag_indicatrix, t=None, alpha=0, beta=0, gamma=0, add_alpha=0, add_beta=0, add_gamma=0):

    # some math: generate points on the surface of the ellipsoid
    
    # Indicatrix
    a = diag_indicatrix[0]
    b = diag_indicatrix[1]
    c = diag_indicatrix[2]

    phi = np.linspace(0, 2*np.pi)
    theta = np.linspace(-np.pi/2, np.pi/2)
    phi, theta=np.meshgrid(phi, theta)
    
    x = np.cos(theta) * np.sin(phi) * a
    y = np.cos(theta) * np.cos(phi) * b
    z = np.sin(theta) * c

    lim_range = np.max([a,b,c])*1.25

    points_ellips = np.array([x.flatten(), y.flatten(), z.flatten()]).T # row = 1 point, column=x,y,z // rotation lattice structure
    points_ellips_rotated_bis = shg.apply_rot_syst(points_ellips, alpha, beta, gamma)
    points_ellips_rotated = shg.apply_rot_syst(points_ellips_rotated_bis, add_alpha, add_beta, add_gamma)
    
    fig = go.Figure(data=[Mesh3d({
                                   'x': points_ellips_rotated[:,0],
                                   'y': points_ellips_rotated[:,1],
                                   'z': points_ellips_rotated[:,2],
                                   'alphahull': 0
                                   })
                        ]
                    )                  
    

    fig.add_trace(go.Scatter3d(x=[-lim_range, lim_range], 
                               y=[0, 0],
                               z=[0, 0],
                               mode='lines', line_width = 15, line_color = "black"))
    fig.add_trace(go.Scatter3d(x=[0, 0], 
                               y=[-lim_range, lim_range],
                               z=[0, 0],
                               mode='lines', line_width = 15, line_color = "black"))
    fig.add_trace(go.Scatter3d(x=[0, 0], 
                               y=[0, 0],
                               z=[-lim_range, lim_range],
                               mode='lines', line_width = 15, line_color = "black"))

    # Optical axis
    if t!=None:
        coord_line_z = 3.0*np.max(z)*np.cos(np.deg2rad(t)) # *1.5 to make it visible despite the ellipsoid
        coord_line_x = 3.0*np.max(z)*np.sin(np.deg2rad(t))
        coords_line = np.array(
            [
                [-coord_line_x,     0,      -coord_line_z],
                [coord_line_x,      0,       coord_line_z]
            ]
        )
        coords_line_rotated_bis = shg.apply_rot_syst(coords_line, alpha, beta, gamma)
        coords_line_rotated = shg.apply_rot_syst(coords_line_rotated_bis, add_alpha, add_beta, add_gamma)
        print(f"{coords_line_rotated = }")

        fig.add_trace(go.Scatter3d(x=coords_line_rotated[:,0], 
                                   y=coords_line_rotated[:,1],
                                   z=coords_line_rotated[:,2],
                                   mode='lines', line_width = 15, line_color = "red"))
        coords_line = np.array(
            [
                [coord_line_x,      0,      -coord_line_z],
                [-coord_line_x,     0,      coord_line_z]
            ]
        )
        coords_line_rotated_bis = shg.apply_rot_syst(coords_line, alpha, beta, gamma)
        coords_line_rotated = shg.apply_rot_syst(coords_line_rotated_bis, add_alpha, add_beta, add_gamma)
        print(f"{coords_line_rotated = }")

        fig.add_trace(go.Scatter3d(x=coords_line_rotated[:,0], 
                                   y=coords_line_rotated[:,1],
                                   z=coords_line_rotated[:,2],
                                   mode='lines', line_width = 15, line_color = "red"))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-lim_range, lim_range]), # nticks=4
            yaxis=dict(range=[-lim_range, lim_range]),
            zaxis=dict(range=[-lim_range, lim_range])
        ),
    )
    fig.update_layout(scene_aspectmode='cube')
    fig.update_layout(scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",),
                    yaxis = dict(
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    zaxis = dict(
                        backgroundcolor="rgb(230, 230,200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),),
                    width=700,
                    margin=dict(
                    r=10, l=10,
                    b=10, t=10)
                  )
    
    
    init_notebook_mode()
    iplot(fig)


class SHGTensor(Tensor):
    @staticmethod
    def get_ieee_rotation(
        structure: Structure,
        refine_rotation: bool = True,
    ) -> SquareTensor:
        """Given a structure associated with a tensor, determines
        the rotation matrix for IEEE conversion according to
        the 1987 IEEE standards.

        Args:
            structure (Structure): a structure associated with the
                tensor to be converted to the IEEE standard
            refine_rotation (bool): whether to refine the rotation
                using SquareTensor.refine_rotation
        """
        # Check conventional setting:
        sga = SpacegroupAnalyzer(structure, symprec=0.001)
        dataset = sga.get_symmetry_dataset()
        trans_mat = dataset.transformation_matrix
        conv_latt = Lattice(np.transpose(np.dot(np.transpose(structure.lattice.matrix), np.linalg.inv(trans_mat))))
        xtal_sys = sga.get_crystal_system()

        vecs = conv_latt.matrix
        lengths = np.array(conv_latt.abc)
        angles = np.array(conv_latt.angles)
        rotation = np.zeros((3, 3))

        # IEEE rules: a,b,c || x1,x2,x3
        if xtal_sys == "cubic":
            rotation = [vecs[i] / lengths[i] for i in range(3)]

        # IEEE rules: a=b in length; c,a || x3, x1
        elif xtal_sys == "tetragonal":
            rotation = np.array(
                [vec / mag for (mag, vec) in sorted(zip(lengths, vecs, strict=True), key=lambda x: x[0])]
            )
            if abs(lengths[2] - lengths[1]) < abs(lengths[1] - lengths[0]):
                rotation[0], rotation[2] = rotation[2], rotation[0].copy()
            rotation[1] = get_uvec(np.cross(rotation[2], rotation[0]))

        # IEEE rules: c<a<b; c,a || x3,x1
        elif xtal_sys == "orthorhombic":
            try:
                rotation = [vec / mag for (mag, vec) in sorted(zip(lengths, vecs, strict=True))]
            except ValueError:
                rotation = [vec / mag for (mag, vec) in sorted(zip(lengths, vecs, strict=True), key=lambda x: x[0])]
            rotation = np.roll(rotation, 2, axis=0)

        # IEEE rules: c,a || x3,x1, c is threefold axis
        # Note this also includes rhombohedral crystal systems
        elif xtal_sys in ("trigonal", "hexagonal"):
            # find threefold axis:
            tf_index = np.argmin(abs(angles - 120.0))
            non_tf_mask = np.logical_not(angles == angles[tf_index])
            rotation[2] = get_uvec(vecs[tf_index])
            rotation[0] = get_uvec(vecs[non_tf_mask][0])
            rotation[1] = get_uvec(np.cross(rotation[2], rotation[0]))

        # IEEE rules: b,c || x2,x3; alpha=beta=90, c<a
        elif xtal_sys == "monoclinic":
            # Find unique axis
            u_index = np.argmax(abs(angles - 90.0))
            n_umask = np.logical_not(angles == angles[u_index])
            rotation[1] = get_uvec(vecs[u_index])
            # Shorter of remaining lattice vectors for c axis
            c = next(vec / mag for (mag, vec) in sorted(zip(lengths[n_umask], vecs[n_umask], strict=True)))
            rotation[2] = np.array(c)
            rotation[0] = np.cross(rotation[1], rotation[2])

        # IEEE rules: c || x3, x2 normal to ac plane
        elif xtal_sys == "triclinic":
            try:
                rotation = [vec / mag for (mag, vec) in sorted(zip(lengths, vecs, strict=True))]
            except ValueError:
                rotation = [vec / mag for (mag, vec) in sorted(zip(lengths, vecs, strict=True), key=lambda x: x[0])]
            rotation[1] = get_uvec(np.cross(rotation[2], rotation[0]))
            rotation[0] = np.cross(rotation[1], rotation[2])

        rotation = SquareTensor(rotation)
        if refine_rotation:
            rotation = rotation.refine_rotation()

        return rotation

R_z = np.array([ # rotation of +90 around z
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1],
])
R_x = np.array([ # rotation of +90 around x
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0],
])
R_y = np.array([ # rotation of +90 around y
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
])


# ====================================================================================

# Get the ABINIT symmetries (spg, pg, and crystal system) for a df 
# with a 'structure' column containing the dict of pmg Structure
def get_abisym(
        df_nosym, 
        path_df_abisym = "df_abisym.json.gz"
):
    # Load the df at the path_df_abisym if it exists
    if path_df_abisym is not None and Path(path_df_abisym).exists():
        print("A df with the abisym already exists, loading...")
        df = pd.read_json(path_df_abisym)
    else:
        # Copy for safety
        df = df_nosym.copy()

        # to use PointGroup.from_space_group I had to update my pymatgen version... before I was using the following:
        # $ pip freeze | grep pymatgen
        # pymatgen==2024.3.1
        from pymatgen.symmetry.groups import PointGroup
        from abipy.core.structure import Structure as AbiStructure
        list_abi_spg_symbol = []
        list_abi_spg_number = []
        list_abi_pg_symbol = []
        list_abi_crystal_system = []
        for ir, r in df.iterrows():
            structure = AbiStructure.from_dict(r['structure'])
            dict_spginfo = structure.abiget_spginfo()
            list_abi_spg_symbol.append(dict_spginfo['spg_symbol'].split("(#")[0])
            list_abi_spg_number.append(dict_spginfo['spg_number'])
            list_abi_crystal_system.append(dict_spginfo['bravais'].split("(")[1][:-1])
            try:
                list_abi_pg_symbol.append(PointGroup.from_space_group(sg_symbol=list_abi_spg_symbol[-1]).symbol)
            except ValueError:
                # Some matching between spg and pg are unknown to pymatgen, so need to do this matching manually via 
                # Wikipedia list of spg
                if list_abi_spg_symbol[-1] in ["P3_221", "P3_121", "P321"]:
                    list_abi_pg_symbol.append(PointGroup("32").symbol)
                elif list_abi_spg_symbol[-1] in ["P3m1"]:
                    list_abi_pg_symbol.append(PointGroup("3m").symbol)
                else:
                    print(list_abi_spg_symbol[-1])
                    raise ValueError
    
        df['abi_spg_symbol'] = list_abi_spg_symbol
        df['abi_spg_number'] = list_abi_spg_number
        df['abi_pg_symbol'] = list_abi_pg_symbol
        df['abi_crystal_system'] = list_abi_crystal_system
    
        if path_df_abisym is not None:
            df.to_json(path_df_abisym, orient="columns")

        return df
    
# ===================================================================================================================

# Takes a df with dijk, epsij, structure (dict of pmg Structure), and a pg and spg as input
# Returns a copy of the input df with new columns related to the conventional form of the SHG tensor
def get_conv(
        df,
        path_df_conv = "df_conv.json.gz",
        name_pg = "abi_pg_symbol",
        name_spg = "abi_spg_symbol",
        tol_rel_null_full = 5e-2,
        force_zero = True,
        force_neumann = True,
):

    # Load the new df if it already exists
    if path_df_conv is not None and Path(path_df_conv).exists():
        df_rot_ieee = pd.read_json(path_df_conv)
    else:
        list_dijk_rot               = []
        list_epsij_rot              = []
        list_structure_rot          = []
        list_rot_is_conventional    = []
        list_pg_match               = []

        for ir, r in df.iterrows():
            d = r['dijk']
            eps = r['epsij']
            structure = Structure.from_dict(r['structure'])
            pg = r[name_pg]
            spg = r[name_spg]

            spga = SpacegroupAnalyzer(structure=structure, symprec=0.001) # dflt 0.01
            if pg!=spga.get_point_group_symbol():
                list_pg_match.append(False)
            else:
                list_pg_match.append(True)

            if  verify_shg_conventional_shape_by_pg(d, pg_symbol=pg, tol_max_null=1e-6, tol_rel_null_full=tol_rel_null_full, 
                                                        spg_symbol=spg)[0]:
                d_rot = copy(d)
                eps_rot = copy(eps)
                structure_new = structure.copy()

            # not a CS per se but zero response in static limit by symmetry
            elif pg=="422":
                d_rot = np.zeros_like(np.array(d))
                eps_rot = copy(eps)
                structure_new = structure.copy()

            else:
                try:
                    # d_rot = np.array(SHGTensor(np.array(d)).convert_to_ieee(structure=structure))
                    # eps_rot = np.array(SHGTensor(np.array(eps)).convert_to_ieee(structure=structure))
                    # if np.allclose(d_rot, 0.0):
                    d_rot = np.array(SHGTensor(np.array(d)).convert_to_ieee(structure=structure, initial_fit=False))
                    eps_rot = np.array(SHGTensor(np.array(eps)).convert_to_ieee(structure=structure, initial_fit=False))

                    R_ieee = np.array(SHGTensor(np.array(d)).get_ieee_rotation(structure=structure))
                    structure_new = structure.copy().apply_operation(SymmOp.from_rotation_and_translation(rotation_matrix=R_ieee))
                    rot_pmg_ieee = True
                except StopIteration:
                    d_rot = copy(d)
                    eps_rot = copy(eps)
                    structure_new = structure.copy()
                    rot_pmg_ieee = False

                if pg=="mm2": # bc IEEE conventions not adapted for mm2 (see Roberts 1992)
                    sga = SpacegroupAnalyzer(structure_new, symprec=0.001) # dflt 0.01
                    for symmop in sga.get_point_group_operations(cartesian=True)[1:]: # [1:] to avoid identity
                        if math.isclose(np.prod(np.diag(symmop.affine_matrix[:3,:3])), 1): # only operation 2 is True
                            symmop_2 = symmop.affine_matrix[:3,:3]
                            symmop_2 = np.round(symmop_2) # to avoid dealing with 0.9999999999
                            break
                    else:
                        raise Exception(f"Symmop 2 not found for pg mm2 in material {ir}")
                    dataset = sga.get_symmetry_dataset()
                    trans_mat = dataset.transformation_matrix
                    conv_latt = Lattice(np.transpose(np.dot(np.transpose(structure_new.lattice.matrix), np.linalg.inv(trans_mat))))
                    arg_polar_axis = np.where(symmop_2==1)[0][0]
                    vec_polar_axis = conv_latt.matrix[arg_polar_axis]
                    t = [0,1,2]
                    t.remove(arg_polar_axis)
                    vec0 = conv_latt.matrix[t,:][0]
                    vec1 = conv_latt.matrix[t,:][1]
                    if np.linalg.norm(vec0)<np.linalg.norm(vec1):
                        vec_a = vec0
                    else:
                        vec_a = vec1

                    while   not math.isclose(vec_polar_axis[2]/np.linalg.norm(vec_polar_axis), 1) and \
                            not math.isclose(vec_a[0]/np.linalg.norm(vec_a), 1):
                        if math.isclose(symmop_2[0,0], 1):
                            angles_for_mm2 = [0, 90, 0]
                            R_axis2 = R_y
                        elif math.isclose(symmop_2[1,1], 1):
                            angles_for_mm2 = [90, 0, 0]
                            R_axis2 = R_x
                        elif math.isclose(symmop_2[2,2], 1) and math.isclose(vec_polar_axis[2]/np.linalg.norm(vec_polar_axis), -1):
                            angles_for_mm2 = [180, 0, 0]
                            R_axis2 = np.array([ # rotation of +180 around x
                                [1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1],
                            ])
                        elif math.isclose(symmop_2[2,2], 1) and math.isclose(vec_polar_axis[2]/np.linalg.norm(vec_polar_axis), 1):
                            angles_for_mm2 = [0, 0, 90]
                            R_axis2 = R_z
                        else:
                            raise Exception(f"Something is wrong with the symmop_2: {symmop_2} ")
                        d_rot = shg.apply_rot(d_rot, *angles_for_mm2)
                        eps_rot = shg.apply_rot(eps_rot, *angles_for_mm2)
                        structure_new = structure_new.copy().apply_operation(SymmOp.from_rotation_and_translation(rotation_matrix=R_axis2))

                        sga = SpacegroupAnalyzer(structure_new, symprec=0.001) # dflt 0.01
                        for symmop in sga.get_point_group_operations(cartesian=True)[1:]: # [1:] to avoid identity
                            if math.isclose(np.prod(np.diag(symmop.affine_matrix[:3,:3])), 1): # only operation 2 is True
                                symmop_2 = symmop.affine_matrix[:3,:3]
                                symmop_2 = np.round(symmop_2)
                                break
                        arg_polar_axis = np.where(symmop_2==1)[0][0]
                        dataset = sga.get_symmetry_dataset()
                        trans_mat = dataset.transformation_matrix
                        conv_latt = Lattice(np.transpose(np.dot(np.transpose(structure_new.lattice.matrix), np.linalg.inv(trans_mat))))
                        arg_polar_axis = np.where(symmop_2==1)[0][0]
                        vec_polar_axis = conv_latt.matrix[arg_polar_axis]
                        t = [0,1,2]
                        t.remove(arg_polar_axis)
                        vec0 = conv_latt.matrix[t,:][0]
                        vec1 = conv_latt.matrix[t,:][1]
                        if np.linalg.norm(vec0)<np.linalg.norm(vec1):
                            vec_a = vec0
                        else:
                            vec_a = vec1

                # Align mirror plan with x-z plan (-1 at symmop[1,1]), if b//-y, then rotate by 180 around x, 
                # then align the smallest lattice parameter (conventional setting for all this) with z
                # Needed bc some monoclinic are highly symmetrical with three conventional angles of 90 degrees 
                # such that the algorithm throws errors
                elif pg=="m" and not rot_pmg_ieee:
                    sga = SpacegroupAnalyzer(structure_new, symprec=0.001) # dflt 0.01
                    for symmop in sga.get_point_group_operations(cartesian=True)[1:]: # [1:] to avoid identity
                        symmop_m = np.round(symmop.affine_matrix[:3,:3])
                        if -1 in np.diag(symmop_m) and 1 in np.diag(symmop_m) and np.prod(np.diag(symmop_m))==-1:
                            break
                    dataset = sga.get_symmetry_dataset()
                    trans_mat = dataset.transformation_matrix
                    conv_latt = Lattice(np.transpose(np.dot(np.transpose(structure_new.lattice.matrix), np.linalg.inv(trans_mat))))
                    arg_axis_normal_m = np.where(symmop_m==-1)[0][0]
                    vec_axis_normal_m = conv_latt.matrix[arg_axis_normal_m]
                    t = [0,1,2]
                    t.remove(arg_axis_normal_m)
                    vec0 = conv_latt.matrix[t,:][0]
                    vec1 = conv_latt.matrix[t,:][1]
                    if np.linalg.norm(vec0)<np.linalg.norm(vec1):
                        vec_c = vec0
                    else:
                        vec_c = vec1

                    while   not math.isclose(vec_axis_normal_m[1]/np.linalg.norm(vec_axis_normal_m), 1) and \
                            not math.isclose(vec_c[2]/np.linalg.norm(vec_c), 1):
                        if math.isclose(symmop_m[0,0], -1):
                            angles_for_m = [0, 0, 90]
                            R_axis2 = R_z
                        elif math.isclose(symmop_m[1,1], -1) and math.isclose(vec_axis_normal_m[1]/np.linalg.norm(vec_axis_normal_m), -1):
                            angles_for_m = [180, 0, 0]
                            R_axis2 = np.array([ # rotation of +180 around x
                                [1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1],
                            ])
                        elif math.isclose(symmop_m[1,1], -1) and math.isclose(vec_axis_normal_m[1]/np.linalg.norm(vec_axis_normal_m), 1):
                            angles_for_m = [0, 90, 0]
                            R_axis2 = R_y
                        elif math.isclose(symmop_m[2,2], -1):
                            angles_for_m = [90, 0, 0]
                            R_axis2 = R_x
                        else:
                            raise Exception(f"Something is wrong with the symmop_m: {symmop_m} ")
                        d_rot = shg.apply_rot(d_rot, *angles_for_m)
                        eps_rot = shg.apply_rot(eps_rot, *angles_for_m)
                        structure_new = structure_new.copy().apply_operation(SymmOp.from_rotation_and_translation(rotation_matrix=R_axis2))

                        sga = SpacegroupAnalyzer(structure_new, symprec=0.001) # dflt 0.01
                        for symmop in sga.get_point_group_operations(cartesian=True)[1:]: # [1:] to avoid identity
                            symmop_m = np.round(symmop.affine_matrix[:3,:3])
                            if -1 in np.diag(symmop_m) and 1 in np.diag(symmop_m) and np.prod(np.diag(symmop_m))==-1:
                                break
                        dataset = sga.get_symmetry_dataset()
                        trans_mat = dataset.transformation_matrix
                        conv_latt = Lattice(np.transpose(np.dot(np.transpose(structure_new.lattice.matrix), np.linalg.inv(trans_mat))))
                        arg_axis_normal_m = np.where(symmop_m==-1)[0][0]
                        vec_axis_normal_m = conv_latt.matrix[arg_axis_normal_m]
                        t = [0,1,2]
                        t.remove(arg_axis_normal_m)
                        vec0 = conv_latt.matrix[t,:][0]
                        vec1 = conv_latt.matrix[t,:][1]
                        if np.linalg.norm(vec0)<np.linalg.norm(vec1):
                            vec_c = vec0
                        else:
                            vec_c = vec1


            list_dijk_rot.append(list(d_rot))
            list_epsij_rot.append(list(eps_rot))
            list_structure_rot.append(structure_new.as_dict())
            if pg == "422":
                list_rot_is_conventional.append(True)
            elif not verify_shg_conventional_shape_by_pg(d_rot, pg_symbol=pg, tol_max_null=1e-6, tol_rel_null_full=tol_rel_null_full, 
                                                        spg_symbol=spg)[0]:
                list_rot_is_conventional.append(False)
            else:
                list_rot_is_conventional.append(True)

        df_rot_ieee = df.copy()

        df_rot_ieee['pg_match_abi_spga'] = list_pg_match
        df_rot_ieee['dijk_rot'] = list_dijk_rot
        df_rot_ieee['epsij_rot'] = list_epsij_rot
        df_rot_ieee['structure_rot'] = list_structure_rot
        df_rot_ieee['rot_is_conventional'] = list_rot_is_conventional

        # for ir, r in df_rot_ieee[(df_rot_ieee['abi_pg_symbol'].isin(["mm2", "-6m2", "P31m"])) & (df_rot_ieee['rot_is_conventional']==False)].iterrows():
        for ir, r in df_rot_ieee[df_rot_ieee['rot_is_conventional']==False].iterrows():
            for angles in [[90, 0, 0], [0, 90, 0], [0, 0, 90], [90, 90, 0], [90, 0, 90], [0, 90, 90], [90, 90, 90]]:
                d_rot = shg.apply_rot(r['dijk_rot'], *angles)
                eps_rot = shg.apply_rot(r['epsij_rot'], *angles)
                structure = Structure.from_dict(r['structure_rot'])
                structure_new = structure.copy()
                structure_new.lattice = Lattice(matrix=shg.apply_rot_syst(structure.lattice.matrix, *angles))
                if verify_shg_conventional_shape_by_pg(d_rot, pg_symbol=r[name_pg], tol_max_null=1e-6, tol_rel_null_full=tol_rel_null_full, 
                                                        spg_symbol=r[name_spg])[0]:
                    df_rot_ieee.at[ir, 'dijk_rot'] = list(d_rot)
                    df_rot_ieee.at[ir, 'epsij_rot'] = list(eps_rot)
                    df_rot_ieee.at[ir, 'structure_rot'] = structure_new.as_dict()
                    df_rot_ieee.at[ir, 'rot_is_conventional'] = True
                    break
        for ir, r in df_rot_ieee[df_rot_ieee['rot_is_conventional']==False].iterrows():
            for angles in [[90, 0, 0], [0, 90, 0], [0, 0, 90], [90, 90, 0], [90, 0, 90], [0, 90, 90], [90, 90, 90]]:
                d_rot = shg.apply_rot(r['dijk'], *angles)
                eps_rot = shg.apply_rot(r['epsij'], *angles)
                structure = Structure.from_dict(r['structure'])
                structure_new = structure.copy()
                structure_new.lattice = Lattice(matrix=shg.apply_rot_syst(structure.lattice.matrix, *angles))
                if verify_shg_conventional_shape_by_pg(d_rot, pg_symbol=r[name_pg], tol_max_null=1e-6, tol_rel_null_full=tol_rel_null_full, 
                                                        spg_symbol=r[name_spg])[0]:
                    df_rot_ieee.at[ir, 'dijk_rot'] = list(d_rot)
                    df_rot_ieee.at[ir, 'epsij_rot'] = list(eps_rot)
                    df_rot_ieee.at[ir, 'structure_rot'] = structure_new.as_dict()
                    df_rot_ieee.at[ir, 'rot_is_conventional'] = True
                    break


        for ir, r in df_rot_ieee[(df_rot_ieee[name_pg]=="-42m") & (df_rot_ieee['rot_is_conventional']==False)].iterrows():
            angles = [0, 0, 45]
            d_rot = shg.apply_rot(r['dijk'], *angles)
            eps_rot = shg.apply_rot(r['epsij'], *angles)
            structure = Structure.from_dict(r['structure'])
            structure_new = structure.copy()
            structure_new.lattice = Lattice(matrix=shg.apply_rot_syst(structure.lattice.matrix, *angles))
            if verify_shg_conventional_shape_by_pg(d_rot, pg_symbol=r[name_pg], tol_max_null=1e-6, tol_rel_null_full=tol_rel_null_full, 
                                                    spg_symbol=r[name_spg])[0]:
                df_rot_ieee.at[ir, 'dijk_rot'] = list(d_rot)
                df_rot_ieee.at[ir, 'epsij_rot'] = list(eps_rot)
                df_rot_ieee.at[ir, 'structure_rot'] = structure_new.as_dict()
                df_rot_ieee.at[ir, 'rot_is_conventional'] = True



        df_rot_ieee['dKP_full_wrt_dKP'] = [0.0]*len(df_rot_ieee)
        for ir, r in df_rot_ieee[df_rot_ieee['rot_is_conventional']==False].iterrows():
            pg = r[name_pg]
            spg = r[name_spg]
            dKP_start = shg.get_dKP(r['dijk'])
            dKP_full = shg.get_dKP(get_full_d(r['dijk'], pg_symbol=pg, spg_symbol=spg))
            if verify_shg_conventional_shape_by_pg(np.array(r['dijk']), pg_symbol=pg, spg_symbol=spg, tol_rel_null_full=1.5)[0] and \
            np.abs(dKP_start-dKP_full)/dKP_start <= 0.1:
                df_rot_ieee.at[ir, 'dijk_rot'] = list(r['dijk'])
                df_rot_ieee.at[ir, 'epsij_rot'] = list(r['epsij'])
                df_rot_ieee.at[ir, 'structure_rot'] = r['structure']
                df_rot_ieee.at[ir, 'rot_is_conventional'] = True
                df_rot_ieee.at[ir, 'dKP_full_wrt_dKP'] = np.abs(dKP_start-dKP_full)/dKP_start
        for ir, r in df_rot_ieee[df_rot_ieee['rot_is_conventional']==False].iterrows():
            pg = r[name_pg]
            spg = r[name_spg]
            dKP_start = shg.get_dKP(r['dijk_rot'])
            dKP_full = shg.get_dKP(get_full_d(r['dijk_rot'], pg_symbol=pg, spg_symbol=spg))
            if verify_shg_conventional_shape_by_pg(np.array(r['dijk_rot']), pg_symbol=pg, spg_symbol=spg, tol_rel_null_full=1.50)[0] and \
            np.abs(dKP_start-dKP_full)/dKP_start <= 0.1:
                df_rot_ieee.at[ir, 'rot_is_conventional'] = True
                df_rot_ieee.at[ir, 'dKP_full_wrt_dKP'] = np.abs(dKP_start-dKP_full)/dKP_start



        df_rot_ieee['spga_001_spg_for_conventional'] = ["abi_spg_ok_when_conventional"]*len(df_rot_ieee)
        df_rot_ieee['spga_001_pg_for_conventional'] = ["abi_pg_ok_when_conventional"]*len(df_rot_ieee)
        for ir, r in df_rot_ieee[df_rot_ieee['rot_is_conventional']==False].iterrows():
            for angles in [[0, 0, 0], [90, 0, 0], [0, 90, 0], [0, 0, 90], [90, 90, 0], [90, 0, 90], [0, 90, 90], [90, 90, 90]]:
                d_rot = shg.apply_rot(r['dijk_rot'], *angles)
                eps_rot = shg.apply_rot(r['epsij_rot'], *angles)
                structure = Structure.from_dict(r['structure_rot'])
                structure_new = structure.copy()
                structure_new.lattice = Lattice(matrix=shg.apply_rot_syst(structure.lattice.matrix, *angles))
                spga = SpacegroupAnalyzer(structure=structure, symprec=0.01)
                if verify_shg_conventional_shape_by_pg(d_rot, pg_symbol=spga.get_point_group_symbol(), tol_max_null=1e-6, tol_rel_null_full=tol_rel_null_full, 
                                                        spg_symbol=spga.get_space_group_symbol())[0]:
                    df_rot_ieee.at[ir, 'dijk_rot'] = list(d_rot)
                    df_rot_ieee.at[ir, 'epsij_rot'] = list(eps_rot)
                    df_rot_ieee.at[ir, 'structure_rot'] = structure_new.as_dict()
                    df_rot_ieee.at[ir, 'rot_is_conventional'] = True
                    df_rot_ieee.at[ir, 'spga_001_pg_for_conventional'] = spga.get_point_group_symbol()
                    df_rot_ieee.at[ir, 'spga_001_spg_for_conventional'] = spga.get_space_group_symbol()
                    break
        for ir, r in df_rot_ieee[df_rot_ieee['rot_is_conventional']==False].iterrows():
            for angles in [[0, 0, 0], [90, 0, 0], [0, 90, 0], [0, 0, 90], [90, 90, 0], [90, 0, 90], [0, 90, 90], [90, 90, 90]]:
                d_rot = shg.apply_rot(r['dijk'], *angles)
                eps_rot = shg.apply_rot(r['epsij'], *angles)
                structure = Structure.from_dict(r['structure'])
                structure_new = structure.copy()
                structure_new.lattice = Lattice(matrix=shg.apply_rot_syst(structure.lattice.matrix, *angles))
                spga = SpacegroupAnalyzer(structure=structure, symprec=0.01)
                if verify_shg_conventional_shape_by_pg(d_rot, pg_symbol=spga.get_point_group_symbol(), tol_max_null=1e-6, tol_rel_null_full=tol_rel_null_full, 
                                                        spg_symbol=spga.get_space_group_symbol())[0]:
                    df_rot_ieee.at[ir, 'dijk_rot'] = list(d_rot)
                    df_rot_ieee.at[ir, 'epsij_rot'] = list(eps_rot)
                    df_rot_ieee.at[ir, 'structure_rot'] = structure_new.as_dict()
                    df_rot_ieee.at[ir, 'rot_is_conventional'] = True
                    df_rot_ieee.at[ir, 'spga_001_pg_for_conventional'] = spga.get_point_group_symbol()
                    df_rot_ieee.at[ir, 'spga_001_spg_for_conventional'] = spga.get_space_group_symbol()
                    break


        df_rot_ieee['rot_was_symmetrized'] = [False]*len(df_rot_ieee)
        # For the materials still not okay, let's allow a symmetrization of the tensor based on the structure given
        # BUT only if the new dKP is within 5% of the original one
        for ir, r in df_rot_ieee[df_rot_ieee['rot_is_conventional']==False].iterrows():
            try:
                d = r['dijk']
                structure = Structure.from_dict(r['structure'])
                dKP_start = shg.get_dKP(d)
                d_rot = np.array(SHGTensor(np.array(d)).convert_to_ieee(structure=structure, initial_fit=True))
                dKP_end = shg.get_dKP(d_rot)
                if np.abs(dKP_end-dKP_start)/dKP_start > 0.05:
                    continue
                eps_rot = np.array(SHGTensor(np.array(eps)).convert_to_ieee(structure=structure, initial_fit=True))
                R_ieee = np.array(SHGTensor(np.array(d)).get_ieee_rotation(structure=structure))
                structure_new = structure.copy().apply_operation(SymmOp.from_rotation_and_translation(rotation_matrix=R_ieee))
                if verify_shg_conventional_shape_by_pg(d_rot, pg_symbol=r[name_pg], tol_max_null=1e-6, tol_rel_null_full=tol_rel_null_full, 
                                                        spg_symbol=r[name_spg])[0]:
                    df_rot_ieee.at[ir, 'dijk_rot'] = list(d_rot)
                    df_rot_ieee.at[ir, 'epsij_rot'] = list(eps_rot)
                    df_rot_ieee.at[ir, 'structure_rot'] = structure_new.as_dict()
                    df_rot_ieee.at[ir, 'rot_is_conventional'] = True
                    df_rot_ieee.at[ir, 'rot_was_symmetrized'] = True
            except StopIteration:
                continue

        df_rot_ieee['why_not_conventional_original'] = ["is_conventional"]*len(df_rot_ieee)
        df_rot_ieee['why_not_conventional_rot'] = ["is_conventional"]*len(df_rot_ieee)
        for ir, r in df_rot_ieee[df_rot_ieee['rot_is_conventional']==False].iterrows():
            pg = r[name_pg]
            spg = r[name_spg]
            df_rot_ieee.at[ir, "why_not_conventional_original"] = verify_shg_conventional_shape_by_pg(np.array(r['dijk']), pg_symbol=pg, spg_symbol=spg, tol_rel_null_full=0.50)[1]
            df_rot_ieee.at[ir, "why_not_conventional_rot"] = verify_shg_conventional_shape_by_pg(np.array(r['dijk_rot']), pg_symbol=pg, spg_symbol=spg, tol_rel_null_full=0.50)[1]


        # recompute dKP
        list_dKP_rot = []
        for ir, r in df_rot_ieee.iterrows():
            list_dKP_rot.append(shg.get_dKP(r['dijk_rot']))
        df_rot_ieee['dKP_rot'] = list_dKP_rot

        # Force components that should be 0 to perfect 0 (instead of 1e-...)
        if force_zero:
            list_dKP_full = []
            list_dijk_full = []
            for ir, r in df_rot_ieee.iterrows():
                d_rot = r['dijk_rot']
                if r['spga_001_spg_for_conventional']!="abi_spg_ok_when_conventional":
                    pg = r['spga_001_pg_for_conventional']
                    spg = r['spga_001_spg_for_conventional']
                else:
                    pg = r['abi_pg_symbol']
                    spg = r['abi_spg_symbol']
    
                list_dijk_full.append(shg.from_voigt(get_full_d(r['dijk_rot'], pg_symbol=pg, spg_symbol=spg)))
                list_dKP_full.append(shg.get_dKP(list_dijk_full[-1]))
    
            df_rot_ieee['dijk_full'] = list_dijk_full
            df_rot_ieee['dKP_full'] = list_dKP_full


        # Enforce Neumann's principles (equality between certain components)
        if force_neumann:
            list_dijk_full_neum = []
            list_dKP_full_neum = []
            for ir, r in df_rot_ieee.iterrows():
                d_rot = r['dijk_full']
                if r['spga_001_spg_for_conventional']!="abi_spg_ok_when_conventional":
                    pg = r['spga_001_pg_for_conventional']
                    spg = r['spga_001_spg_for_conventional']
                else:
                    pg = r['abi_pg_symbol']
                    spg = r['abi_spg_symbol']

                if not verify_shg_neumann_by_pg(d=d_rot, pg_symbol=pg, spg_symbol=spg, abs_tol=1e-9, rel_tol_coef=None):
                    d_constr = correct_shg_neumann_by_pg(d_rot, pg_symbol=pg, spg_symbol=spg)
                else:
                     d_constr = d_rot

                if   verify_shg_neumann_by_pg(d_constr, pg_symbol=pg, spg_symbol=spg, abs_tol=1e-9, rel_tol_coef=None) and\
                    (verify_shg_conventional_shape_by_pg(d_constr, pg_symbol=pg, spg_symbol=spg,
                                                            tol_max_null=1e-9, tol_rel_null_full=1e-4)[0] or \
                     pg=="422" # bc dijk=0 so impossible to verify shg_conventional_shape
                    ):
                    list_dijk_full_neum.append(list(shg.from_voigt(d_constr)))
                    list_dKP_full_neum.append(shg.get_dKP(d_constr))
                else:
                    raise Exception(f"{ir}")

            df_rot_ieee['dijk_full_neum'] = list_dijk_full_neum
            df_rot_ieee['dKP_full_neum'] = list_dKP_full_neum

        # Save final df
        if path_df_conv is not None:
            df_rot_ieee.to_json(path_df_conv)
    
    return df_rot_ieee


# ==================================================================================================================

def main(file, action, force_zero, force_neum):
    df = pd.read_json(file)

    if action=="abisym":
        _ = get_abisym(
            df_nosym = df,
        )
    elif action=="findconv":
        _ = get_conv(
            df = df,
            force_zero = force_zero,
            force_neumann = force_neum,
        )
    else:
        raise ValueError(f"The argument 'action' is unknown as it is neither 'abisym' nor 'findconv'.")


if __name__ == "__main__":
    import argparse

    # Set up the argument parser
    # Add required "optional" argument 
    parser = argparse.ArgumentParser(description="Process a file to either get the abisym or find the conventional SHG form.")
    parser.add_argument('--file',     type=str, required=True, help="Path to the df as a json file to be processed")
    parser.add_argument('--action',   type=str, required=True, help="Type of action to perform: abisym or findconv")
    parser.add_argument('--no_force_zero', action='store_false', help="When findconv, avoid enforcing the symmetry into the null components.")
    parser.add_argument('--no_force_neum', action='store_false', help="When findconv, avoid enforcing Neumann's principle into equal components.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.file, args.action, args.no_force_zero, args.no_force_neum) 
