"""Differentiable 3D rendering of Scenic scenes using PyTorch3D.

Scenic's built-in schematic viewer renders a :class:`trimesh.scene.Scene` with
trimesh's OpenGL/pyglet backend.  That backend is convenient but it is not
differentiable and it needs a display (or Xvfb).  This module converts the same
trimesh scene into a PyTorch3D :class:`~pytorch3d.structures.Meshes` object and
renders it with PyTorch3D's differentiable mesh rasterizer and Phong shader.

The entry point used by Scenic is :func:`renderScene`, which returns PNG bytes.
:func:`renderSceneTensor` and :func:`renderMeshTensor` return image tensors and
keep the autograd graph, so gradients can be propagated back to mesh vertices
(see the ``verts`` argument of :func:`renderMeshTensor`).
"""

from __future__ import annotations

import io
import os

import numpy as np
import trimesh

DEFAULT_RESOLUTION = 1080
DEFAULT_FOV = 45.0
DEFAULT_ELEVATION = 90.0  # degrees above the XY plane; 90 is a top-down view
DEFAULT_AZIMUTH = 0.0  # degrees around Z; 0 looks along -Y (with +Y up on screen)

_DEFAULT_MESH_COLOR = (0.55, 0.55, 0.55)
_DEFAULT_PATH_COLOR = (0.25, 0.25, 0.30)
_DEFAULT_POINT_COLOR = (0.1, 0.1, 0.1)

# Fraction of the scene's bounding-box diagonal used as the radius of the
# tubes which represent line paths (workspace wireframes, visible regions).
_PATH_RADIUS_FRACTION = 0.0015


# ---------------------------------------------------------------------------
# Small colour/geometry helpers
# ---------------------------------------------------------------------------


def _rgbaToRgb(rgba):
    """Convert uint8/float RGBA colours to float RGB, compositing over white."""
    colors = np.asarray(rgba)
    if colors.ndim == 1:
        colors = colors[np.newaxis, :]
    if colors.shape[1] < 3:
        raise ValueError(f"expected RGB(A) colours, got shape {colors.shape}")
    if np.issubdtype(colors.dtype, np.integer):
        colors = colors.astype(np.float64) / 255.0
    else:
        colors = colors.astype(np.float64)
        if colors.max(initial=0.0) > 1.0:
            colors /= 255.0
    rgb = colors[:, :3]
    if colors.shape[1] > 3:
        alpha = colors[:, 3:4]
        rgb = rgb * alpha + (1.0 - alpha)
    return np.clip(rgb, 0.0, 1.0)


def _vertexColors(mesh, count):
    """Best-effort extraction of per-vertex RGB colours from a trimesh mesh."""
    visual = mesh.visual
    if visual is not None:
        try:
            colors = visual.vertex_colors
        except Exception:  # textured visuals may not expose vertex colours
            colors = None
        if colors is not None and len(colors) == count:
            return _rgbaToRgb(colors)
        try:
            color = visual.main_color
        except Exception:
            color = None
        if color is not None:
            return np.tile(_rgbaToRgb(color)[0], (count, 1))
    return np.tile(np.asarray(_DEFAULT_MESH_COLOR), (count, 1))


def _meshData(mesh, transform=None):
    """Convert a trimesh mesh to (vertices, faces, colours, normals) arrays."""
    if transform is not None:
        mesh = mesh.copy()
        mesh.vertices = trimesh.transform_points(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(transform, dtype=np.float64),
        )
    if len(mesh.faces) == 0:
        return None

    # Disconnect faces across sharp creases while keeping smooth patches, then
    # use the resulting vertex normals.  This matches trimesh's OpenGL viewer
    # (which also smooth-shades small meshes) and avoids round-looking boxes.
    try:
        smooth = mesh.smooth_shaded
    except Exception:
        smooth = mesh

    verts = np.asarray(smooth.vertices, dtype=np.float64)
    faces = np.asarray(smooth.faces, dtype=np.int64)
    if len(verts) == 0 or len(faces) == 0:
        return None
    try:
        normals = np.asarray(smooth.vertex_normals, dtype=np.float64)
    except Exception:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    colors = _vertexColors(smooth, len(verts))
    return verts, faces, colors, normals


def _segmentTubes(segments, radius, colors):
    """Build a thin 4-sided prism for each line segment.

    Args:
        segments: (N, 2, 3) array of line segment endpoints.
        radius: Radius of the tubes.
        colors: Either a single RGB colour or (N, 3) per-segment RGB colours.

    Returns:
        (vertices, faces, colours, normals) arrays, or None if no segments.
    """
    seg = np.asarray(segments, dtype=np.float64)
    if seg.size == 0:
        return None
    if seg.ndim != 3 or seg.shape[1] != 2:
        raise ValueError(f"expected segments of shape (N, 2, 3), got {seg.shape}")
    lengths = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
    keep = lengths > 1e-12
    seg, lengths = seg[keep], lengths[keep]
    if len(seg) == 0:
        return None

    colors = np.asarray(colors, dtype=np.float64)
    if colors.ndim == 1:
        colors = np.tile(colors, (len(seg), 1))
    else:
        colors = colors[keep]
    if len(colors) != len(seg):
        colors = np.tile(colors[0], (len(seg), 1))

    directions = (seg[:, 1] - seg[:, 0]) / lengths[:, None]

    # Pick a helper axis which is not nearly parallel to each segment, then
    # build two unit vectors spanning the plane perpendicular to the segment.
    helper = np.zeros_like(directions)
    horizontal = np.abs(directions[:, 0]) < 0.9
    helper[horizontal, 0] = 1.0
    helper[~horizontal, 1] = 1.0
    u = np.cross(directions, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(directions, u)

    # Four corners of a square cross-section.
    radial = np.stack((u + v, u - v, -u + v, -u - v), axis=1) / np.sqrt(2.0)
    bottom = seg[:, 0, None, :] + radius * radial
    top = seg[:, 1, None, :] + radius * radial
    verts = np.stack((bottom, top), axis=1).reshape(-1, 3)

    base = (np.arange(len(seg)) * 8)[:, None]
    bottomIdx = base + np.arange(4)[None, :]
    topIdx = base + 4 + np.arange(4)[None, :]
    faces = []
    for j in range(4):
        jn = (j + 1) % 4
        b0, b1 = bottomIdx[:, j], bottomIdx[:, jn]
        t0, t1 = topIdx[:, j], topIdx[:, jn]
        faces.append(np.stack((b0, b1, t1), axis=1))
        faces.append(np.stack((b0, t1, t0), axis=1))
    # Caps (winding does not matter here because explicit normals are supplied).
    faces.append(np.stack((bottomIdx[:, 0], bottomIdx[:, 2], bottomIdx[:, 1]), axis=1))
    faces.append(np.stack((bottomIdx[:, 0], bottomIdx[:, 3], bottomIdx[:, 2]), axis=1))
    faces.append(np.stack((topIdx[:, 0], topIdx[:, 1], topIdx[:, 2]), axis=1))
    faces.append(np.stack((topIdx[:, 0], topIdx[:, 2], topIdx[:, 3]), axis=1))
    faces = np.concatenate(faces, axis=0)

    normals = np.stack((radial, radial), axis=1).reshape(-1, 3)
    colors = np.repeat(colors, 8, axis=0)
    return verts, faces, colors, normals


def _pointMarkers(points, radius, colors):
    """Build a small octahedron at each point."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return None
    colors = np.asarray(colors, dtype=np.float64)
    if colors.ndim == 1:
        colors = np.tile(colors, (len(points), 1))
    offsets = radius * np.array(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    verts = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    octahedron = np.array(
        (
            (0, 2, 4),
            (2, 1, 4),
            (1, 3, 4),
            (3, 0, 4),
            (2, 0, 5),
            (1, 2, 5),
            (3, 1, 5),
            (0, 3, 5),
        )
    )
    faces = octahedron[None, :, :] + (np.arange(len(points)) * 6)[:, None, None]
    faces = faces.reshape(-1, 3)
    normals = np.tile(offsets, (len(points), 1))
    colors = np.repeat(colors, 6, axis=0)
    return verts, faces, colors, normals


def _combine(parts):
    """Combine a list of (verts, faces, colours, normals) arrays."""
    parts = [part for part in parts if part is not None]
    if not parts:
        return None
    verts, faces, colors, normals = [], [], [], []
    offset = 0
    for v, f, c, n in parts:
        verts.append(np.asarray(v, dtype=np.float64))
        faces.append(np.asarray(f, dtype=np.int64) + offset)
        colors.append(np.asarray(c, dtype=np.float64))
        normals.append(np.asarray(n, dtype=np.float64))
        offset += len(v)
    return (
        np.concatenate(verts),
        np.concatenate(faces),
        np.concatenate(colors),
        np.concatenate(normals),
    )


# ---------------------------------------------------------------------------
# Scene conversion and camera framing
# ---------------------------------------------------------------------------


def _sceneTransform(scene, name):
    try:
        transform = scene.graph.get(name)[0]
    except Exception:
        transform = None
    return None if transform is None else np.asarray(transform, dtype=np.float64)


def _collectGeometry(scene):
    """Convert a trimesh scene to a single indexed mesh."""
    meshParts = []
    pathParts = []
    pointParts = []
    boundsPoints = []

    for name, geometry in scene.geometry.items():
        transform = _sceneTransform(scene, name)
        if isinstance(geometry, trimesh.primitives.Primitive):
            geometry = geometry.to_mesh()
        if isinstance(geometry, trimesh.Trimesh):
            data = _meshData(geometry, transform)
            if data is not None:
                meshParts.append(data)
                boundsPoints.append(data[0])
        elif isinstance(geometry, trimesh.path.Path3D):
            pathColors = getattr(geometry, "colors", None)
            if pathColors is not None:
                pathColors = np.asarray(pathColors)
                if pathColors.ndim == 1:
                    pathColors = pathColors[None, :]
            for index, entity in enumerate(geometry.entities):
                if pathColors is not None:
                    color = pathColors[index % len(pathColors)]
                else:
                    color = _DEFAULT_PATH_COLOR
                color = _rgbaToRgb(color)[0]
                try:
                    polyline = np.asarray(
                        entity.discrete(geometry.vertices), dtype=np.float64
                    )
                except Exception:
                    continue
                if polyline.ndim != 2 or polyline.shape[1] != 3 or len(polyline) < 2:
                    continue
                if transform is not None:
                    polyline = trimesh.transform_points(polyline, transform)
                segments = np.stack((polyline[:-1], polyline[1:]), axis=1)
                pathParts.append((segments, color))
                boundsPoints.append(polyline)
        elif isinstance(geometry, trimesh.PointCloud):
            vertices = np.asarray(geometry.vertices, dtype=np.float64)
            if transform is not None:
                vertices = trimesh.transform_points(vertices, transform)
            colors = getattr(geometry, "colors", None)
            if colors is None:
                colors = np.tile(np.asarray(_DEFAULT_POINT_COLOR), (len(vertices), 1))
            else:
                colors = _rgbaToRgb(colors)
            pointParts.append((vertices, colors))
            boundsPoints.append(vertices)

    if boundsPoints:
        allPoints = np.concatenate(
            [np.asarray(points).reshape(-1, 3) for points in boundsPoints]
        )
    else:
        allPoints = np.zeros((0, 3), dtype=np.float64)

    if len(allPoints) > 0:
        low, high = allPoints.min(axis=0), allPoints.max(axis=0)
    else:
        low = high = np.zeros(3)
    diagonal = float(np.linalg.norm(high - low))
    if diagonal <= 0.0:
        diagonal = 1.0
    lineRadius = max(_PATH_RADIUS_FRACTION * diagonal, 1e-6)

    parts = list(meshParts)
    for segments, color in pathParts:
        parts.append(_segmentTubes(segments, lineRadius, color))
    for points, colors in pointParts:
        parts.append(_pointMarkers(points, 2.0 * lineRadius, colors))

    combined = _combine(parts)
    if combined is None:
        return None
    verts, faces, colors, normals = combined
    # Normals may be slightly non-unit after operations such as smoothing.
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    normals = normals / lengths
    return verts, faces, colors, normals


def _computeCamera(verts, resolution, fov, elevation, azimuth, margin=1.05):
    """Fit a perspective camera so that all vertices are inside the frame."""
    height, width = resolution
    low, high = verts.min(axis=0), verts.max(axis=0)
    center = (low + high) / 2.0

    el = np.radians(elevation)
    az = np.radians(azimuth)
    direction = np.array(
        (np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)),
        dtype=np.float64,
    )
    forward = -direction
    if abs(elevation) >= 89.5:
        up = np.array((0.0, 1.0, 0.0))
    else:
        up = np.array((0.0, 0.0, 1.0))
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cameraUp = np.cross(right, forward)
    cameraUp /= np.linalg.norm(cameraUp)

    relative = verts - center
    x = relative @ right
    y = relative @ cameraUp
    z = relative @ forward

    aspect = width / float(height)
    tanY = np.tan(np.radians(fov) / 2.0)
    tanX = tanY * aspect
    distance = max(
        float(np.max(np.abs(x) / tanX - z)),
        float(np.max(np.abs(y) / tanY - z)),
    )
    distance = max(distance, 1e-4) * margin
    eye = center + direction * distance

    znear = max(1e-4, float(distance + z.min()) * 0.5)
    zfar = float(distance + z.max()) * 1.5 + 1e-3
    return eye, center, up, znear, zfar, aspect


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _getDevice(device=None):
    import torch

    if isinstance(device, torch.device):
        return device
    if device is None or str(device).lower() in ("", "auto"):
        device = os.environ.get("SCENIC_PYTORCH3D_DEVICE")
    if device is None or str(device).lower() in ("", "auto"):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _renderMesh(
    verts,
    faces,
    colors,
    normals,
    resolution,
    device,
    fov,
    elevation,
    azimuth,
):
    """Rasterize and shade one indexed mesh, returning an (H, W, 3) tensor."""
    import torch
    from pytorch3d.renderer import (
        BlendParams,
        FoVPerspectiveCameras,
        Materials,
        MeshRasterizer,
        MeshRenderer,
        PointLights,
        RasterizationSettings,
        SoftPhongShader,
        TexturesVertex,
        look_at_view_transform,
    )
    from pytorch3d.structures import Meshes

    height, width = resolution
    verts = torch.as_tensor(verts, dtype=torch.float32, device=device)
    faces = torch.as_tensor(faces, dtype=torch.int64, device=device)
    if len(verts) == 0 or len(faces) == 0:
        return _whiteImage(resolution, device)
    colors = torch.as_tensor(colors, dtype=torch.float32, device=device)
    if normals is not None:
        normals = torch.as_tensor(normals, dtype=torch.float32, device=device)

    # Camera fitting is not part of the differentiable computation.
    cameraData = _computeCamera(
        verts.detach().cpu().numpy().astype(np.float64),
        resolution,
        fov,
        elevation,
        azimuth,
    )
    eye, at, up, znear, zfar, aspect = cameraData
    R, T = look_at_view_transform(
        eye=eye[None, :], at=at[None, :], up=up[None, :], device=device
    )
    cameras = FoVPerspectiveCameras(
        device=device,
        R=R,
        T=T,
        fov=fov,
        aspect_ratio=aspect,
        znear=znear,
        zfar=zfar,
    )

    if normals is None:
        mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=TexturesVertex([colors]),
        )
    else:
        mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=TexturesVertex([colors]),
            verts_normals=[normals],
        )

    lights = PointLights(
        device=device,
        location=eye[None, :],
        ambient_color=((0.45, 0.45, 0.45),),
        diffuse_color=((0.55, 0.55, 0.55),),
        specular_color=((0.10, 0.10, 0.10),),
    )
    materials = Materials(
        device=device,
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((1.0, 1.0, 1.0),),
        specular_color=((1.0, 1.0, 1.0),),
        shininess=32.0,
    )
    blend = BlendParams(background_color=(1.0, 1.0, 1.0), sigma=1e-4, gamma=1e-4)
    rasterSettings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=1e-5,
        faces_per_pixel=8,
        perspective_correct=True,
        clip_barycentric_coords=True,
        cull_backfaces=False,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=rasterSettings),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials,
            blend_params=blend,
        ),
    )
    images = renderer(mesh)
    return images[0, ..., :3].clamp(0.0, 1.0)


def _whiteImage(resolution, device):
    import torch

    height, width = resolution
    return torch.ones((height, width, 3), dtype=torch.float32, device=device)


def _toPNG(image):
    from PIL import Image

    array = (image.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _parseResolution(resolution):
    if isinstance(resolution, (int, np.integer)):
        return int(resolution), int(resolution)
    height, width = resolution
    return int(height), int(width)


def renderMeshTensor(
    verts,
    faces,
    colors=None,
    verts_normals=None,
    *,
    resolution=DEFAULT_RESOLUTION,
    device=None,
    fov=DEFAULT_FOV,
    elevation=DEFAULT_ELEVATION,
    azimuth=DEFAULT_AZIMUTH,
):
    """Render a single mesh with PyTorch3D's differentiable rasterizer.

    Args:
        verts: (V, 3) tensor/array of vertex positions.  If this is a tensor
            requiring gradients, the returned image keeps the autograd graph.
        faces: (F, 3) triangle indices.
        colors: (V, 3) RGB colours in [0, 1]; defaults to mid-grey.
        verts_normals: (V, 3) vertex normals; computed from ``faces`` if absent.
        resolution: Output size (int or ``(height, width)``).
        device: Torch device (default: CUDA if available, else CPU).
        fov, elevation, azimuth: Camera parameters.

    Returns:
        An (H, W, 3) float tensor of rendered RGB values.
    """
    import torch

    device = _getDevice(device)
    resolution = _parseResolution(resolution)
    verts = torch.as_tensor(verts, dtype=torch.float32, device=device)
    faces = torch.as_tensor(faces, dtype=torch.int64, device=device)
    if len(verts) == 0 or len(faces) == 0:
        return _whiteImage(resolution, device)
    if colors is None:
        colors = torch.full((len(verts), 3), 0.5, dtype=torch.float32, device=device)
    else:
        colors = torch.as_tensor(colors, dtype=torch.float32, device=device)
        if colors.ndim == 2 and colors.shape[1] == 4:
            colors = colors[:, :3]
    normals = (
        None
        if verts_normals is None
        else torch.as_tensor(verts_normals, dtype=torch.float32, device=device)
    )
    return _renderMesh(
        verts, faces, colors, normals, resolution, device, fov, elevation, azimuth
    )


def renderSceneTensor(
    scene,
    *,
    resolution=DEFAULT_RESOLUTION,
    axes=False,
    device=None,
    fov=DEFAULT_FOV,
    elevation=DEFAULT_ELEVATION,
    azimuth=DEFAULT_AZIMUTH,
):
    """Render a trimesh scene and return the RGB image tensor.

    ``axes`` is accepted for compatibility with Scenic's existing viewer API;
    the PyTorch3D renderer does not currently draw an axis overlay.
    """
    device = _getDevice(device)
    resolution = _parseResolution(resolution)
    # The axis overlay is not reproduced by the PyTorch3D renderer.
    geometry = _collectGeometry(scene)
    if geometry is None:
        return _whiteImage(resolution, device)
    verts, faces, colors, normals = geometry
    return _renderMesh(
        verts,
        faces,
        colors,
        normals,
        resolution,
        device,
        fov,
        elevation,
        azimuth,
    )


def renderScene(
    scene,
    *,
    resolution=DEFAULT_RESOLUTION,
    axes=False,
    device=None,
    fov=DEFAULT_FOV,
    elevation=DEFAULT_ELEVATION,
    azimuth=DEFAULT_AZIMUTH,
):
    """Render a trimesh scene with PyTorch3D and return PNG bytes."""
    image = renderSceneTensor(
        scene,
        resolution=resolution,
        axes=axes,
        device=device,
        fov=fov,
        elevation=elevation,
        azimuth=azimuth,
    )
    return _toPNG(image)
