import numpy as np
import torch


def cvt_local2global(local_point, src_point):
    """
    Convert points from local frame to global
    :param local_point: A local point or array of local points that must be converted 1-D np.array or 2-D np.array
    :param src_point: A
    :return:
    """
    size = local_point.shape[-1]
    x, y, a = 0, 0, 0
    if size == 3:
        x, y, a = local_point.T
    elif size == 2:
        x, y = local_point.T
    X, Y, A = src_point.T
    x1 = x * np.cos(A) - y * np.sin(A) + X
    y1 = x * np.sin(A) + y * np.cos(A) + Y
    a1 = (a + A + np.pi) % (2 * np.pi) - np.pi
    if size == 3:
        return np.array([x1, y1, a1]).T
    elif size == 2:
        return np.array([x1, y1]).T
    else:
        return


def torch_local2global(local_points, source_points):
    X, Y, A = source_points.T
    x, y, a = local_points.T
    x1 = x * np.cos(A) - y * np.sin(A) + X
    y1 = x * np.sin(A) + y * np.cos(A) + Y
    a1 = (a + A + np.pi) % (2 * np.pi) - np.pi
    return torch.stack([x1, y1, a1]).T


def cvt_global2local(global_point, src_point):
    size = global_point.shape[-1]
    x1, y1, a1 = 0, 0, 0
    if size == 3:
        x1, y1, a1 = global_point.T
    elif size == 2:
        x1, y1 = global_point.T
    X, Y, A = src_point.T
    x = x1 * np.cos(A) + y1 * np.sin(A) - X * np.cos(A) - Y * np.sin(A)
    y = -x1 * np.sin(A) + y1 * np.cos(A) + X * np.sin(A) - Y * np.cos(A)
    a = (a1 - A + np.pi) % (2 * np.pi) - np.pi
    if size == 3:
        return np.array([x, y, a]).T
    elif size == 2:
        return np.array([x, y]).T
    else:
        return

def torch_global2local(global_point, src_point):
    x1, y1 = global_point.T
    X, Y, A = src_point.T
    x = x1 * torch.cos(A) + y1 * torch.sin(A) - X * torch.cos(A) - Y * torch.sin(A)
    y = -x1 * torch.sin(A) + y1 * torch.cos(A) + X * torch.sin(A) - Y * torch.cos(A)
    return torch.stack([x, y]).T


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
