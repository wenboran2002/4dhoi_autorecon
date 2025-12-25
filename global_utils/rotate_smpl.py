import torch
import smplx
import pickle
import trimesh
import os
import numpy as np
import functools

def _copysign(a, b):
    """
    Return an array where each element has the absolute value taken from a,
    with the sign taken from b. This is like the standard copysign
    floating-point operation.

    Args:
        a: source array.
        b: array whose signs will be used, of the same shape as a.

    Returns:
        Array of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return np.where(signs_differ, -a, a)

def _sqrt_positive_part(x):
    """
    Returns np.sqrt(np.maximum(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = np.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = np.sqrt(x[positive_mask])
    return ret

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = np.split(quaternions, 4, axis=-1)
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def _axis_angle_rotation(axis, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape array of Euler angles in radians

    Returns:
        Rotation matrices as array of shape (..., 3, 3).
    """

    cos = np.cos(angle)
    sin = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return np.stack(R_flat, axis=-1).reshape(angle.shape + (3, 3))

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = np.linalg.norm(axis_angle, ord=2, axis=-1, keepdims=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = np.concatenate(
        [np.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1
    )
    return quaternions

def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as numpy array of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as numpy array of shape (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return np.stack((o0, o1, o2, o3), -1)

def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as numpy array of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as numpy array
        of shape (..., 3), where the magnitude is the angle
        turned anticlockwise in radians around the vector's
        direction.
    """
    norms = np.linalg.norm(quaternions[..., 1:], axis=-1, keepdims=True)
    half_angles = np.arctan2(norms, quaternions[..., 0:1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def _angle_from_tan(axis, other_axis, data, horizontal, tait_bryan):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as numpy array of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a numpy array
        of shape (...).
    """
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return np.arctan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return np.arctan2(-data[..., i2], data[..., i1])
    return np.arctan2(data[..., i2], -data[..., i1])

def _index_from_letter(letter):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2

def matrix_to_euler_angles(matrix, convention):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as numpy array of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as numpy array of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = np.arcsin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = np.arccos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return np.stack(o, axis=-1)

def euler_angles_to_matrix(euler_angles, convention):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as numpy array of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as numpy array of shape (..., 3, 3).
    """
    if euler_angles.ndim == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, np.split(euler_angles,3, axis=-1))
    return functools.reduce(np.matmul, matrices)

def rotate(vertices,orient_path,obj_pcd,scale,dist):
    smpl_vertices = vertices.copy()
    params = np.load(orient_path)
    global_orient = params['global_orient']
    global_orient_org = global_orient.copy()
    
    joint = params['joint']
    pelvis = joint * scale + dist
    print(global_orient)
    theta = np.linalg.norm(global_orient) * 360 / (2 * np.pi)
    global_orient[0,[0,2]] = [0,0]
    global_orient[0,1] *= -1
    print(global_orient)
    transform_matrix = axis_angle_to_matrix(global_orient)
    print('transform_matrix',transform_matrix)
    print('pelvis',pelvis)
      
    m = axis_angle_to_matrix(global_orient_org)
    
    s = smpl_vertices-pelvis
    obj_pcd = obj_pcd-pelvis
    out = np.matmul(transform_matrix[0],s[0].T).T.reshape(1,-1,3)
    obj_pcd_ = np.matmul(transform_matrix[0],obj_pcd[0].T).T.reshape(1,-1,3)

    axis_angle = matrix_to_axis_angle(np.matmul(transform_matrix[0],m[0]))
    print('axis_anglge',axis_angle)
    return out,transform_matrix,obj_pcd_,pelvis

def rotate_after(vertices,obj_pcd,orient,pv):
    smpl_vertices = vertices.copy()
    pelvis = np.asarray(pv.points)
    global_orient = orient
    global_orient_org = orient.copy()
    print(global_orient)
    theta = np.linalg.norm(global_orient) * 360 / (2 * np.pi)
    global_orient[0,[0,2]] = [0,0]
    global_orient[0,1] *= -1
    print(global_orient)
    transform_matrix = axis_angle_to_matrix(global_orient)
    print('transform_matrix', transform_matrix)
    print('pelvis',pelvis)
    
    m = axis_angle_to_matrix(global_orient_org)
    
    s = smpl_vertices-pelvis
    obj_pcd = obj_pcd-pelvis
    out=np.matmul(transform_matrix[0],s[0].T).T.reshape(1,-1,3)
    obj_pcd_ = np.matmul(transform_matrix[0],obj_pcd[0].T).T.reshape(1,-1,3)

    axis_angle = matrix_to_axis_angle(np.matmul(transform_matrix[0],m[0]))
    print('axis_anglge',axis_angle)
    return out, transform_matrix, obj_pcd_, pelvis

