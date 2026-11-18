"""Tests for the PyTorch3D-backed 3D scene renderer."""

import numpy as np
import pytest
import trimesh

pytest.importorskip("pytorch3d")

from scenic.core.pytorch3d_renderer import (  # noqa: E402
    renderMeshTensor,
    renderScene,
    renderSceneTensor,
)


def makeScene():
    scene = trimesh.scene.Scene()
    box = trimesh.creation.box((1, 2, 3))
    box.apply_transform(trimesh.transformations.euler_matrix(0.6, 0.3, 0.2))
    box.visual.face_colors = [30, 179, 0, 255]
    scene.add_geometry(box, geom_name="box")

    sphere = trimesh.creation.icosphere(radius=0.7)
    sphere.apply_translation([1.8, 0, 0])
    sphere.visual.face_colors = [30, 179, 0, 255]
    scene.add_geometry(sphere, geom_name="sphere")

    # A wireframe path, as used for workspace maps and visibility regions.
    edges = box.face_adjacency_edges[box.face_adjacency_angles > np.radians(0.1)]
    path = trimesh.path.Path3D(
        **trimesh.path.exchange.misc.edges_to_path(edges, box.vertices)
    )
    scene.add_geometry(path, geom_name="wireframe")
    return scene


def test_renderScene_returnsPNG():
    png = renderScene(makeScene(), resolution=128)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 100


def test_renderSceneTensor_shapeAndColors():
    image = renderSceneTensor(makeScene(), resolution=(64, 96))
    assert tuple(image.shape) == (64, 96, 3)
    array = image.detach().cpu().numpy()
    assert array.min() >= 0.0 and array.max() <= 1.0
    # Background is white and the object is green.
    assert np.allclose(array[0, 0], 1.0, atol=1e-4)
    assert np.any(array[..., 1] > array[..., 0])


def test_renderScene_empty():
    image = renderSceneTensor(trimesh.scene.Scene(), resolution=32)
    assert tuple(image.shape) == (32, 32, 3)
    assert np.allclose(image.detach().cpu().numpy(), 1.0)


def test_renderMeshTensor_isDifferentiable():
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    verts = torch.tensor(
        mesh.vertices, dtype=torch.float32, device=device, requires_grad=True
    )
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    colors = torch.full((len(verts), 3), 0.5, dtype=torch.float32, device=device)

    image = renderMeshTensor(verts, faces, colors, resolution=64, device=device)
    assert image.requires_grad
    (image**2).mean().backward()
    assert verts.grad is not None
    assert torch.isfinite(verts.grad).all()
    assert verts.grad.abs().sum() > 0.0
