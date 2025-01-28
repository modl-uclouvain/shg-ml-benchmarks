## This module contains functions dealing with the SHG tensor 'd'=========================================================

## Import necessary modules ===============================================================================================
import numpy as np
import inspect
import plotly.graph_objects as go
import scipy.constants as cst

from typing import Optional, Union, get_origin, get_args
from scipy.spatial.transform import Rotation
from abipy.abilab import Structure
from abipy.abilab import units as abu

# Part of the enforcement the type hinting of the arguments ================================================================
def enforce(x,x_type,x_annot):

    if (x_type) is x_annot or (x_type) in get_args(x_annot):
        pass
    else:
        if get_origin(x_annot) is Union:
            raise Exception(f"{x} is of type {x_type} instead of types {get_args(x_annot)}")
        else:
            raise Exception(f"{x} is of type {x_type} instead of type {x_annot}")

    return

# Returns d (numpy array or list) as an (numpy) array =====================================================================
# Throws an error if d is neither an array nor a list
def to_array(   d:      Union[np.ndarray,list]
                ) ->    np.ndarray:

    # Enforce the types of the arguments from annotations (generic section)
    args        = inspect.getfullargspec(to_array).args
    annotations = inspect.getfullargspec(to_array).annotations
    for x in args:
        x_type  = type(locals()[x])
        x_annot = annotations[x]
        enforce(x,x_type,x_annot)

    return np.array(d)

# Converts the atomic units of NLO susceptibility (as in Abinit) to SI units in pm/V ======================================
# Throws an error if d is neither an int, a float, a np.ndarray, nor a list
def au_to_pmV(  d: Union[int,float,np.ndarray,list]
                ) -> np.ndarray:

    # Enforce the types of the arguments from annotations (generic section)
    args        = inspect.getfullargspec(au_to_pmV).args
    annotations = inspect.getfullargspec(au_to_pmV).annotations
    for x in args:
        x_type  = type(locals()[x])
        x_annot = annotations[x]
        enforce(x,x_type,x_annot)

    coef_au_to_pmV = 16 * np.pi**2 * abu.bohr_to_ang**2 * 1e-8 * cst.epsilon_0 / cst.elementary_charge

    return d*coef_au_to_pmV

# Returns True if the given list or array is in Voigt notation (shape 3x6)================================================
# Throws an Exception if d is neither 3x3x3 nor 3x6
def is_voigt(d: Union[np.ndarray,list]) -> np.ndarray:

    d       = to_array(d)
    d_shape = d.shape
    if d_shape == (3,6):
        return True
    elif d_shape == (3,3,3):
        return False
    else:
        raise Exception("The tensor d is of shape {d_shape} although it should either be 3x6 (Voigt) or 3x3x3.")


# Returns the d tensor (list or array) in Voigt notation (array, 3x6)======================================================
# Throws an Exception if shape different than 3x6 (Voigt) or 3x3x3
def to_voigt(d: Union[np.ndarray,list]) -> np.ndarray:

    d = to_array(d)

    if is_voigt(d):
        return d
    else:
        d_new = np.zeros([3,6])
        voigt = [[0,0], [1,1], [2,2], [1,2], [0,2], [0,1]]

        for i in range(3):
            for j in range(6):
                d_new[i,j] = d[i,voigt[j][0], voigt[j][1]]

        return d_new


# Returns the d tensor (list or array) in non-Voigt notation (array, 3x3x3)===============================================
# Throws an Exception if shape different than 3x6 (Voigt) or 3x3x3
def from_voigt(d: Union[np.ndarray,list]) -> np.ndarray:

    d = to_array(d)

    if is_voigt(d):
        d_new = np.zeros([3,3,3])
        voigt = np.array([[0, 5, 4],
                        [5, 1, 3],
                        [4, 3, 2]])

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    d_new[i,j,k] = d[i, int(voigt[j, k])]

        return d_new
    else:
        return d


# THIS VERSION IS NOT CORRECT, MY MISTAKE IN READING THE ORIGINAL EQUATION...
# Returns the dKP (d_KP) as defined in [Ref.1]===========================================================================
# Throws an Exception if shape different than 3x6 (Voigt) or 3x3x3
# Prev. named get_dKP
def get_dKP_old(d: Union[np.ndarray,list]) -> float:

    d = from_voigt(d)

    first = 0
    second = 0
    third = 0
    for i in range(3):
        first += (19/105) * d[i,i,i]**2
        for j in range(3):
            if j != i:
                second += (13/105) * d[i,i,i] * d[i,j,j]
                third += (44/105) * d[i,i,j]**2 # 14/105 in Francesco's thesis


    fourth = 0
    for i, j, k in ((0,1,2), (1,2,0), (2,0,1)):
    #for i, j, k in ((0,1,2), (1,2,0), (2,0,1), (0,2,1), (1,0,2), (2,1,0)): # no, only three, stated in Cyvin 1965
        fourth += (13/105) * (d[i,i,j]*d[j,k,k] + (5/7)*d[i,j,k]**2)

    #print(first, second, third, fourth)

    return np.sqrt(first + second + third + fourth)


# THIS VERSION IS THE CORRECT ONE
def get_dKP_weird(d: Union[np.ndarray,list]) -> float:

    d = from_voigt(d)

    first = 0
    second = 0
    third = 0
    for i in range(3):
        first += (19/105) * d[i,i,i]**2
        for j in range(3):
            if j != i:
                second += (13/105) * d[i,i,i] * d[i,j,j]
                third += (44/105) * d[i,i,j]**2 # 14/105 in Francesco's thesis


    fourth = 0
    for i, j, k in ((0,1,2), (1,2,0), (2,0,1)):
        fourth += (13/105) * (d[i,i,j]*d[j,k,k])
    fifth = (5/7)*(d[0,1,2]**2)

    #print(first, second, third, fourth)

    return np.sqrt(first + second + third + fourth + fifth)

def get_dKP(d: Union[np.ndarray,list]) -> float:
    return get_dKP_weird(d)

# Returns the dRMS (sqrt(1stinv/27)) =====================================================================================
def get_dRMS(d: Union[np.ndarray,list]) -> float:

    # Converts d to its non-Voigt form, Error if neither 3x6 nor 3x3x3 initially
    d = from_voigt(d)

    # Computes the 1st invariant of the 3rd-order 3D tensor
    inv1 = np.einsum('ijk,ijk',d,d)

    return np.sqrt(inv1/27)


# Applies a Euler rotation of the crystal basis in 3D and return the updated d tensor or arbitrary order (<4) ============
# Rotation of alpha around x, then beta around new y, then gamma around new z (angles in degrees!)
# Throws an Exception/Error if the arguments are not of the same type as the annotations
# d is a tensor of arbitrary order but must be "3D" or in Voigt notation
def apply_rot(  d:      Union[np.ndarray,list],
                alpha:  Union[int,float,np.float64,np.int64],
                beta:   Union[int,float,np.float64,np.int64],
                gamma:  Union[int,float,np.float64,np.int64],
                ) ->    np.ndarray:


    # Enforce the types of the arguments from annotations (generic section)
    args        = inspect.getfullargspec(apply_rot).args
    annotations = inspect.getfullargspec(apply_rot).annotations
    for x in args:
        x_type  = type(locals()[x])
        x_annot = annotations[x]
        enforce(x,x_type,x_annot)

    # Convert d into a np.ndarray and into a non-Voigt format if necessary and checks that d is "3D" for a rotation in 3D
    d = to_array(d)
    if d.shape == (3,6):
        d = from_voigt(d)
    assert np.all(np.array(d.shape) == 3), "The tensor d is not expressed in the 3 spatial dimensions."

    # Convert the angles from degrees to radians
    alpha *= np.pi/180
    beta  *= np.pi/180
    gamma *= np.pi/180

    # Define variables for cosinus and sinus of the Euler angles
    ca = np.cos(alpha) ; sa = np.sin(alpha)
    cb = np.cos(beta)  ; sb = np.sin(beta)
    cg = np.cos(gamma) ; sg = np.sin(gamma)

    # Rotation matrix Rz*Ry*Rx = https://en.wikipedia.org//wiki/Rotation_matrix#General_rotations
    # When discussing a rotation, there are two possible conventions: rotation of the axes, and
    # rotation of the object relative to fixed axes. The latter is adopted here with the right-handed rule for positive angles,
    # i.e., counter-clockwise rotation when viewed from above the rotational axis
    R = np.array([[cb*cg, sa*sb*cg-ca*sg, ca*sb*cg+sa*sg],
                  [cb*sg, sa*sb*sg+ca*cg, ca*sb*sg-sa*cg],
                  [-sb,   sa*cb,          ca*cb]])


    # Define the order of the tensor to rotate
    ndim = len(d.shape)

    # Rotate d: use np.einsum to multiply d by R in a for-loop over the order of tensor to target each dimension
    subs = 'ijklmnop'[:ndim]
    eins = 'Zz,{}->{}'
    for n in range(ndim): # or reversed(range(ndim)), does not matter
        eins_n = eins.format(subs.replace(subs[n],'z'),subs.replace(subs[n],'Z')) # 'Zz,ijz->ijZ' then 'Zz,izk->iZk' then 'Zz,zjk->Zjk' for ndim=3
        d = np.einsum(eins_n,R,d)

    return d # return the rotated tensor


# Applies a Euler rotation to a system of vectors such as rprim e.g. (each row of syst is a vector to be rotated) ===============
# Rotation of alpha around x, then beta around new y, then gamma around new z (angles in degrees!)
# Throws an Exception/Error if the arguments are not of the same type as the annotations
# syst can have any number of row but each row must be 3 elements long
def apply_rot_syst( syst:   Union[np.ndarray,list],
                    alpha:  Union[int,float,np.float64,np.int64],
                    beta:   Union[int,float,np.float64,np.int64],
                    gamma:  Union[int,float,np.float64,np.int64],
                    ) ->    np.ndarray:

    # Enforce the types of the arguments from annotations (generic section)
    args        = inspect.getfullargspec(apply_rot_syst).args
    annotations = inspect.getfullargspec(apply_rot_syst).annotations
    for x in args:
        x_type  = type(locals()[x])
        x_annot = annotations[x]
        enforce(x,x_type,x_annot)

    # Decomposes the rotation by rotating each vector (1st order tensor) individually
    syst = to_array(syst)
    syst_rot = np.zeros(np.shape(syst))
    for i in range(len(syst)):
        syst_rot[i] = apply_rot(syst[i],alpha,beta,gamma)

    return syst_rot


# Applies rotations of step from 0 to maxangle along alpha, beta and gamma to the 3x3x3 or 3x6 tensor d  ==================
# Computes d_KP and prints some information on the distribution.
def apply_rot_get_dKP_old(  d:          Union[np.ndarray,list],
                        step:       int                     =10,
                        maxangle:   Union[int,float]        =180
                        ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,float,float,float,float,float]:

    # Enforce the types of the arguments from annotations (generic section)
    args        = inspect.getfullargspec(apply_rot_get_dKP_old).args
    annotations = inspect.getfullargspec(apply_rot_get_dKP_old).annotations
    for x in args:
        x_type  = type(locals()[x])
        x_annot = annotations[x]
        enforce(x,x_type,x_annot)

    # Create the arrays of rotational angles around each axis, from 0 to maxangle included with a step of step
    valpha    = np.arange(0,maxangle+0.5,step)
    vbeta     = np.arange(0,maxangle+0.5,step)
    vgamma    = np.arange(0,maxangle+0.5,step)
    l = len(valpha)

    # Instantiate two arrays with the dKP and dijk for each rotation respectively
    dKP_ar  = np.zeros([l,l,l])
    dijk_ar = np.zeros([l,l,l,3,3,3])

    for ia, alpha in enumerate(valpha):
        for ib, beta in enumerate(vbeta):
            for ig, gamma in enumerate(vgamma):
                d_rot               = apply_rot(d, alpha, beta, gamma)
                dKP_ar[ia, ib, ig]  = get_dKP_old(d_rot)
                dijk_ar[ia, ib, ig] = d_rot

    #print(f"Angles around each axis: {valpha}")
    #print("Minimum d_KP :{0}".format(np.min(dKP_ar)))
    #print("Maximum d_KP: {0}".format(np.max(dKP_ar)))
    #print("Average d_KP: {0}".format(np.average(dKP_ar)))
    #print("Minimum value of dij : {0}".format(np.min(dijk_ar)))
    #print("Maximum value of dij : {0}".format(np.max(dijk_ar)))

    return valpha,vbeta,vgamma,dijk_ar,dKP_ar,np.min(dKP_ar),np.max(dKP_ar),np.average(dKP_ar),np.min(dijk_ar),np.max(dijk_ar)


def apply_rot_get_dKP_weird(  d:          Union[np.ndarray,list],
                        step:       int                     =10,
                        maxangle:   Union[int,float]        =180
                        ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,float,float,float,float,float]:

    # Enforce the types of the arguments from annotations (generic section)
    args        = inspect.getfullargspec(apply_rot_get_dKP_weird).args
    annotations = inspect.getfullargspec(apply_rot_get_dKP_weird).annotations
    for x in args:
        x_type  = type(locals()[x])
        x_annot = annotations[x]
        enforce(x,x_type,x_annot)

    # Create the arrays of rotational angles around each axis, from 0 to maxangle included with a step of step
    valpha    = np.arange(0,maxangle+0.5,step)
    vbeta     = np.arange(0,maxangle+0.5,step)
    vgamma    = np.arange(0,maxangle+0.5,step)
    l = len(valpha)

    # Instantiate two arrays with the dKP and dijk for each rotation respectively
    # dKP_ar  = np.zeros([l,l,l])
    dijk_ar = np.zeros([l,l,l,3,3,3])

    for ia, alpha in enumerate(valpha):
        for ib, beta in enumerate(vbeta):
            for ig, gamma in enumerate(vgamma):
                d_rot               = apply_rot(d, alpha, beta, gamma)
                # dKP_ar[ia, ib, ig]  = get_dKP_weird(d_rot)
                dijk_ar[ia, ib, ig] = d_rot

    #print(f"Angles around each axis: {valpha}")
    #print("Minimum d_KP :{0}".format(np.min(dKP_ar)))
    #print("Maximum d_KP: {0}".format(np.max(dKP_ar)))
    #print("Average d_KP: {0}".format(np.average(dKP_ar)))
    #print("Minimum value of dij : {0}".format(np.min(dijk_ar)))
    #print("Maximum value of dij : {0}".format(np.max(dijk_ar)))

    return valpha,vbeta,vgamma,dijk_ar

def apply_rot_get_dKP(  d:          Union[np.ndarray,list],
                        step:       int                     =10,
                        maxangle:   Union[int,float]        =180
                        ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,float,float,float,float,float]:
    return apply_rot_get_dKP_weird( d,
                                    step,
                                    maxangle
    )

# Find the Euler angles that allow to go from syst1 to syst2 via a rotation ===============================================
# Each row of syst1 and syst2 should be a new vector of length 3, Error otherwise
def find_euler( syst1: Union[np.ndarray,list],
                syst2: Union[np.ndarray,list]
                ) -> np.ndarray :

    # Enforce the types of the arguments from annotations (generic section)
    args        = inspect.getfullargspec(find_euler).args
    annotations = inspect.getfullargspec(find_euler).annotations
    for x in args:
        x_type  = type(locals()[x])
        x_annot = annotations[x]
        enforce(x,x_type,x_annot)

    syst1 = np.array(syst1) ; syst2 = np.array(syst2)

    assert len(np.shape(syst1)) ==2 and  len(np.shape(syst1)) == 2, 'The two systems must be 2D arrays (shape: N x 3).'
    assert np.shape(syst1)[1] == 3 and np.shape(syst2)[1] == 3, 'The length of each vector (row) of the systems must be 3 elements (3D).'

    R = Rotation.align_vectors(syst2,syst1)[0]

    euler = R.as_euler('xyz',degrees=True)

    return euler

# Plot two sets of lattice vectors in 3D ==================================================================================
def plot_3D_axes(latt1: Union[np.ndarray,list],
                 latt2: Union[np.ndarray,list],
                 name1: str                     = 'Lattice 1',
                 name2: str                     = 'Lattice 2'
                 ) -> go.Figure :

    # Enforce the types of the arguments from annotations (generic section)
    args        = inspect.getfullargspec(plot_3D_axes).args
    annotations = inspect.getfullargspec(plot_3D_axes).annotations
    for x in args:
        x_type  = type(locals()[x])
        x_annot = annotations[x]
        enforce(x,x_type,x_annot)

    latt1 = np.array(latt1) ; latt2 = np.array(latt2)

    assert len(np.shape(latt1)) == 2 and len(np.shape(latt2)) == 2, 'Either latt1 or latt2 is not a 2D array.'
    assert np.shape(latt1) == (3, 3) and np.shape(latt2) == (3, 3), 'Either latt1 or latt2 is not of shape 3 x 3.'

    fig = go.Figure()

    fig.add_traces(data=[go.Scatter3d(  x=[0,latt1[0,0]], y=[0,latt1[0,1]], z=[0,latt1[0,2]],
                                        mode='lines+markers',
                                        line=dict(width=5,color='red'),
                                        name=f'{name1} A')])
    fig.add_traces(data=[go.Scatter3d(  x=[0,latt1[1,0]], y=[0,latt1[1,1]], z=[0,latt1[1,2]],
                                        mode='lines+markers',
                                        line=dict(width=5,color='green'),
                                        name='B')])
    fig.add_traces(data=[go.Scatter3d(  x=[0,latt1[2,0]], y=[0,latt1[2,1]], z=[0,latt1[2,2]],
                                        mode='lines+markers',
                                        line=dict(width=5,color='blue'),
                                        name='C')])

    fig.add_traces(data=[go.Scatter3d(  x=[0,latt2[0,0]], y=[0,latt2[0,1]], z=[0,latt2[0,2]],
                                        mode='lines+markers',
                                        line=dict(width=5,color='red',dash='dash'),
                                        name=f'{name2} A')])
    fig.add_traces(data=[go.Scatter3d(  x=[0,latt2[1,0]], y=[0,latt2[1,1]], z=[0,latt2[1,2]],
                                        mode='lines+markers',
                                        line=dict(width=5,color='green',dash='dash'),\
                                        name='B')])
    fig.add_traces(data=[go.Scatter3d(  x=[0,latt2[2,0]], y=[0,latt2[2,1]], z=[0,latt2[2,2]],
                                        mode='lines+markers',
                                        line=dict(width=5,color='blue',dash='dash'),
                                        name='C')])

    # xaxis.backgroundcolor is used to set background color
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

    return fig

# Plot the lattice vectors and atoms positions of an abipy Structure with its rotated counterpart in 3D ===================
def plot_3D_struc(struc:        Structure,
                  name:         str                                     = 'NLDS',
                  alpha:        Union[int,float,np.float64,np.int64]    = 0,
                  beta:         Union[int,float,np.float64,np.int64]    = 0,
                  gamma:        Union[int,float,np.float64,np.int64]    = 0,
                  sites_orig:   bool                                    = True,
                  sites_rot:    bool                                    = True
                 ) -> go.Figure :

    # Enforce the types of the arguments from annotations (generic section)
    args        = inspect.getfullargspec(plot_3D_struc).args
    annotations = inspect.getfullargspec(plot_3D_struc).annotations
    for x in args:
        x_type  = type(locals()[x])
        x_annot = annotations[x]
        enforce(x,x_type,x_annot)

    # Instantiates the lattice vectors and atoms positions of the original structure
    latt    = struc.lattice_vectors()
    sites   = struc.sites
    PosA    = [site.coords[0] for site in sites]
    PosB    = [site.coords[1] for site in sites]
    PosC    = [site.coords[2] for site in sites]

    lst_species     = [str(site.specie) for site in struc.sites]
    uniqs,idcs,counts  = np.unique(lst_species,return_index=True,return_counts=True)
    uniqs = uniqs[np.argsort(idcs)] ; counts = counts[np.argsort(idcs)] ; idcs = np.sort(idcs)

    # Rotates the lattice vectors and atoms positions
    need_rot = np.all(np.array([alpha,beta,gamma])==0)
    if not need_rot:
        latt_rot = apply_rot_syst(latt,alpha,beta,gamma)
        PosA_rot = [apply_rot(site.coords,alpha,beta,gamma)[0] for site in sites]
        PosB_rot = [apply_rot(site.coords,alpha,beta,gamma)[1] for site in sites]
        PosC_rot = [apply_rot(site.coords,alpha,beta,gamma)[2] for site in sites]


    fig = go.Figure()

    fig.add_traces(data=[go.Scatter3d(  x=[0,latt[0,0]], y=[0,latt[0,1]], z=[0,latt[0,2]],
                                        mode='lines+markers',
                                        line=dict(width=5,color='red'),
                                        marker=dict(size=5),
                                        name=f'{name} A')])
    fig.add_traces(data=[go.Scatter3d(  x=[0,latt[1,0]], y=[0,latt[1,1]], z=[0,latt[1,2]],
                                        mode='lines+markers',
                                        line=dict(width=5,color='green'),
                                        marker=dict(size=5),
                                        name='B')])
    fig.add_traces(data=[go.Scatter3d(  x=[0,latt[2,0]], y=[0,latt[2,1]], z=[0,latt[2,2]],
                                        mode='lines+markers',
                                        line=dict(width=5,color='blue'),
                                        marker=dict(size=5),
                                        name='C')])

    if sites_orig:
        for iuniq in range(len(uniqs)):
            uniq = uniqs[iuniq] ; idx = idcs[iuniq] ; count = counts[iuniq]
            fig.add_traces(data=[go.Scatter3d(  x=PosA[idx:idx+count], y=PosB[idx:idx+count], z=PosC[idx:idx+count],
                                        mode='markers+text',
                                        hovertext=uniq,
                                        text=np.arange(idx,idx+count),
                                        textposition='middle center',
                                        name=f'{name}: {uniq}')])

    if not need_rot:
        fig.add_traces(data=[go.Scatter3d(  x=[0,latt_rot[0,0]], y=[0,latt_rot[0,1]], z=[0,latt_rot[0,2]],
                                            mode='lines+markers',
                                            line=dict(width=5,color='red',dash='dash'),
                                            marker=dict(size=5,line=dict(width=2,color='black')),
                                            name=f'Rot A')])
        fig.add_traces(data=[go.Scatter3d(  x=[0,latt_rot[1,0]], y=[0,latt_rot[1,1]], z=[0,latt_rot[1,2]],
                                            mode='lines+markers',
                                            line=dict(width=5,color='green',dash='dash'),\
                                            marker=dict(size=5,line=dict(width=2,color='black')),
                                            name='B')])
        fig.add_traces(data=[go.Scatter3d(  x=[0,latt_rot[2,0]], y=[0,latt_rot[2,1]], z=[0,latt_rot[2,2]],
                                            mode='lines+markers',
                                            line=dict(width=5,color='blue',dash='dash'),
                                            marker=dict(size=5,line=dict(width=2,color='black')),
                                            name='C')])

        if sites_rot:
            for iuniq in range(len(uniqs)):
                uniq = uniqs[iuniq] ; idx = idcs[iuniq] ; count = counts[iuniq]
                fig.add_traces(data=[go.Scatter3d(  x=PosA_rot[idx:idx+count], y=PosB_rot[idx:idx+count], z=PosC_rot[idx:idx+count],
                                                    mode='markers+text',
                                                    marker=dict(line=dict(width=2,color='black')),
                                                    hovertext=uniq,
                                                    text=np.arange(idx,idx+count),
                                                    textposition='middle center',
                                                    name=f'Rot: {uniq}')])

    # xaxis.backgroundcolor is used to set background color
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

    return fig

# TO CLEAN WHAT FOLLOWS
def get_kernel(d):
    # See https://arxiv.org/pdf/1704.01327.pdf
    # We compute the kernel U of d

    U = np.zeros([3,3])

    for i in range(3):
        for l in range(3):
            for j in range(3):
                for k in range(3):
                    U[i,l] += d[i, j, k] * d[j, k, l]

    return U


def get_zeigenvalue(d, guessV=[1/3**0.5, 1/3**0.5, 1/3**0.5], guessZ=1, iter=1):
    # Compute the Z-eigenvalue nu, as defined by
    # d_ijk x_j x_k = nu x_i with x_i x_i = 1
    #
    # The equation is solved using an initial guess for x (guessV) and nu (guessZ)
    # Feel free to change them !

    from scipy.optimize import fsolve

    def equations(vars, D):
        # Equations to be solved in the form eqs = 0
        x1, x2, x3, nu = vars
        x = [x1, x2, x3]
        eq1 = 0
        eq2 = 0
        eq3 = 0
        for j in range(3):
            for k in range(3):
                eq1 += D[0, j, k] * x[j] * x[k] - nu * x1
                eq2 += D[1, j, k] * x[j] * x[k] - nu * x2
                eq3 += D[2, j, k] * x[j] * x[k] - nu * x3

        eq4 = x1**2 + x2**2 + x3**2 - 1

        return (eq1, eq2, eq3, eq4)

    x1, x2, x3, nu = fsolve(equations, (guessV[0], guessV[1], guessV[2], guessZ), args=(d))
    zeros = np.array(equations((x1, x2, x3, nu), d))

    if np.any(np.abs(zeros[0:-1]) >= 1e-8):
        print("The Z-eigenvalue might not be correct : check the result !")

    elif nu <= 0:

        if iter > 2:
            print("The Z-eigenvalue is negative, even with an opposite vector")
        else:
            x1, x2, x3, nu = get_zeigenvalue(d, guessV=-np.array(guessV), guessZ=1, iter=2)

    return x1, x2, x3, nu


def get_invariants(d):
    # See https://arxiv.org/pdf/1704.01327.pdf, page 14
    # First we get U = Ubar = Uhat
    # Three invariants, related to the eigenvalues of U: a, b and c
    # a, b and c could be considered as the invariants
    # Inv1 = a + b + c = sum_{i,j,k} d_ijk**2
    # Inv2 = a**2 + b**2 + c**2
    # Inv3 = a**3 + b**3 + c**3
    # Inv4 = Z-eigenvalue of d

    U = get_kernel(d) # The eigenvalues of U are lambda_i

    Inv1 = np.trace(U) # Same as lambda1 + lambda2 + lambda3
    Inv2 = np.trace(np.matmul(U,U)) # Same as lambda1**2 + lambda2**2 + lambda3**2
    Inv3 = np.trace(np.matmul(U, np.matmul(U,U))) # Same as lambda1**3 + lambda2**3 + lambda3**3

    # Inv4 = get_zeigenvalue(d)[3]

    return Inv1, Inv2, Inv3#, Inv4


def apply_rot_get_invariants(  d:          Union[np.ndarray,list],
                        step:       int                     =20,
                        maxangle:   Union[int,float]        =360,
                        inv: int = 2
                        ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,float,float,float,float,float]:

    # Enforce the types of the arguments from annotations (generic section)
    args        = inspect.getfullargspec(apply_rot_get_dinv2).args
    annotations = inspect.getfullargspec(apply_rot_get_dinv2).annotations
    for x in args:
        x_type  = type(locals()[x])
        x_annot = annotations[x]
        enforce(x,x_type,x_annot)

    # Create the arrays of rotational angles around each axis, from 0 to maxangle included with a step of step
    valpha    = np.arange(0,maxangle+0.5,step)
    vbeta     = np.arange(0,maxangle+0.5,step)
    vgamma    = np.arange(0,maxangle+0.5,step)
    l = len(valpha)

    # Instantiate two arrays with the dKP and dijk for each rotation respectively
    dKP_ar  = np.zeros([l,l,l])
    dijk_ar = np.zeros([l,l,l,3,3,3])

    for ia, alpha in enumerate(valpha):
        for ib, beta in enumerate(vbeta):
            for ig, gamma in enumerate(vgamma):
                d_rot               = apply_rot(d, alpha, beta, gamma)
                dKP_ar[ia, ib, ig]  = get_invariants(d_rot)[inv-1]
                dijk_ar[ia, ib, ig] = d_rot

    #print(f"Angles around each axis: {valpha}")
    #print("Minimum d_KP :{0}".format(np.min(dKP_ar)))
    #print("Maximum d_KP: {0}".format(np.max(dKP_ar)))
    #print("Average d_KP: {0}".format(np.average(dKP_ar)))
    #print("Minimum value of dij : {0}".format(np.min(dijk_ar)))
    #print("Maximum value of dij : {0}".format(np.max(dijk_ar)))

    return valpha,vbeta,vgamma,dijk_ar,dKP_ar,np.min(dKP_ar),np.max(dKP_ar),np.average(dKP_ar),np.min(dijk_ar),np.max(dijk_ar)
