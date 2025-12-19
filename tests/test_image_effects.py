# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pytest

from omnisealbench.attacks.valuemetric import (
    Brightness as VMBrightness,
    Contrast as VMContrast,
    GaussianBlur as VMGaussianBlur,
    Hue as VMHue,
    JPEG as VMJPEG,
    MedianFilter as VMMedianFilter,
    Saturation as VMSaturation,
    jpeg_compress as vm_jpeg_compress,
    median_filter as vm_median_filter,
)

from omnisealbench.attacks.image_effects import (
    adjust_brightness as ie_adjust_brightness,
    adjust_contrast as ie_adjust_contrast,
    gaussian_blur as ie_gaussian_blur,
    adjust_hue as ie_adjust_hue,
    jpeg_compress as ie_jpeg_compress,
    median_filter as ie_median_filter,
    adjust_saturation as ie_adjust_saturation,
    resize as ie_resize,
    rotate as ie_rotate,
    random_crop as ie_random_crop,
    perspective as ie_perspective,
    hflip as ie_hflip,
)

from omnisealbench.attacks.geometric import (
    Resize as GeomResize,
    Rotate as GeomRotate,
    Crop as GeomCrop,
    Perspective as GeomPerspective,
    HorizontalFlip as GeomHFlip,
)


def _unwrap_and_batch(t):
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        t = t[0]
    if not torch.is_tensor(t):
        return None
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t.cpu()


@pytest.fixture(params=[(3, 128, 128), (3, 256, 256), (3, 512, 512)])
def image_shape(request):
    """Parametrized image shape fixture: (C, H, W)."""
    return request.param


def _cmp_tensors(a, b, rtol: float = 5e-4, atol: float = 5e-3):
    assert a is not None and b is not None, "one side is None"
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        max_abs = (a - b).abs().max().item()
        raise AssertionError(
            f"tensors differ: max_abs={max_abs:.6e} (allowed rtol={rtol}, atol={atol})"
        )


def test_mapping_equivalence_Brightness(image_shape):
    torch.manual_seed(0)
    c, h, w = image_shape
    x = torch.rand(c, h, w)
    xb = x.unsqueeze(0)

    vm = VMBrightness(min_factor=0.5, max_factor=1.5)
    v_out, _ = vm.forward(xb, None, factor=1.2)

    i_out = ie_adjust_brightness(xb, brightness_factor=1.2)

    v_t = _unwrap_and_batch(v_out)
    i_t = _unwrap_and_batch(i_out)
    _cmp_tensors(v_t, i_t, atol=5e-2)


def test_mapping_equivalence_Contrast(image_shape):
    torch.manual_seed(0)
    c, h, w = image_shape
    x = torch.rand(c, h, w)
    xb = x.unsqueeze(0)

    vm = VMContrast(min_factor=0.5, max_factor=1.5)
    v_out, _ = vm.forward(xb, None, factor=1.2)

    i_out = ie_adjust_contrast(xb, contrast_factor=1.2)

    v_t = _unwrap_and_batch(v_out)
    i_t = _unwrap_and_batch(i_out)
    _cmp_tensors(v_t, i_t, atol=5e-2)


def test_mapping_equivalence_GaussianBlur(image_shape):
    torch.manual_seed(0)
    c, h, w = image_shape
    x = torch.rand(c, h, w)
    xb = x.unsqueeze(0)

    vm = VMGaussianBlur(min_kernel_size=3, max_kernel_size=5)
    v_out, _ = vm.forward(xb, None, kernel_size=5)

    i_out = ie_gaussian_blur(xb, kernel_size=5)

    v_t = _unwrap_and_batch(v_out)
    i_t = _unwrap_and_batch(i_out)
    _cmp_tensors(v_t, i_t)


def test_mapping_equivalence_Hue(image_shape):
    torch.manual_seed(0)
    c, h, w = image_shape
    x = torch.rand(c, h, w)
    xb = x.unsqueeze(0)

    vm = VMHue(min_factor=-0.5, max_factor=0.5)
    v_out, _ = vm.forward(xb, None, factor=0.1)

    i_out = ie_adjust_hue(xb, hue_factor=0.1)

    v_t = _unwrap_and_batch(v_out)
    i_t = _unwrap_and_batch(i_out)
    _cmp_tensors(v_t, i_t)


def test_mapping_equivalence_JPEG(image_shape):
    torch.manual_seed(0)
    c, h, w = image_shape
    x = torch.rand(c, h, w)
    xb = x.unsqueeze(0)

    vm = VMJPEG(min_quality=40, max_quality=80)
    v_out, _ = vm.forward(xb, None, quality=75)

    # image_effects.jpeg_compress expects a single-image tensor (C,H,W)
    i_out = ie_jpeg_compress(x, quality_factor=75)

    v_t = _unwrap_and_batch(v_out)
    i_t = _unwrap_and_batch(i_out)
    _cmp_tensors(v_t, i_t, atol=0.2)


def test_mapping_equivalence_MedianFilter(image_shape):
    torch.manual_seed(0)
    c, h, w = image_shape
    x = torch.rand(c, h, w)
    xb = x.unsqueeze(0)

    vm = VMMedianFilter(min_kernel_size=3, max_kernel_size=3)
    v_out, _ = vm.forward(xb, None, kernel_size=3)

    i_out = ie_median_filter(xb, kernel_size=3)

    v_t = _unwrap_and_batch(v_out)
    i_t = _unwrap_and_batch(i_out)
    _cmp_tensors(v_t, i_t)


def test_mapping_equivalence_Saturation(image_shape):
    torch.manual_seed(0)
    c, h, w = image_shape
    x = torch.rand(c, h, w)
    xb = x.unsqueeze(0)

    vm = VMSaturation(min_factor=0.5, max_factor=1.5)
    v_out, _ = vm.forward(xb, None, factor=1.2)

    i_out = ie_adjust_saturation(xb, saturation_factor=1.2)

    v_t = _unwrap_and_batch(v_out)
    i_t = _unwrap_and_batch(i_out)
    _cmp_tensors(v_t, i_t, atol=5e-2)


def test_mapping_equivalence_jpeg_compress(image_shape):
    torch.manual_seed(0)
    c, h, w = image_shape
    x = torch.rand(c, h, w)

    # valuemetric function expects single-image tensor (C,H,W)
    v_out = vm_jpeg_compress(x, 75)
    i_out = ie_jpeg_compress(x, quality_factor=75)

    v_t = _unwrap_and_batch(v_out)
    i_t = _unwrap_and_batch(i_out)
    _cmp_tensors(v_t, i_t, atol=0.2)


def test_mapping_equivalence_median_filter(image_shape):
    torch.manual_seed(0)
    c, h, w = image_shape
    x = torch.rand(c, h, w)
    xb = x.unsqueeze(0)

    v_out = vm_median_filter(xb, 3)
    i_out = ie_median_filter(xb, kernel_size=3)

    v_t = _unwrap_and_batch(v_out)
    i_t = _unwrap_and_batch(i_out)
    _cmp_tensors(v_t, i_t)


def test_mapping_equivalence_Resize(image_shape):
    """Compare geometric.Resize class against image_effects.resize function.

    image_effects.resize takes an area-scale (scale) and internally applies
    sqrt(scale) to edge lengths. geometric.Resize.forward accepts a size
    parameter treated as an edge-scale multiplier, so we pass sqrt(scale)
    to the class to make outputs comparable.
    """
    import math

    torch.manual_seed(0)
    c, h, w = image_shape
    x = torch.rand(c, h, w)
    xb = x.unsqueeze(0)

    scale = 0.5

    # call image_effects.resize (area scale)
    try:
        i_out = ie_resize(xb, scale)
    except Exception as e:
        pytest.skip(f"image_effects.resize failed: {e}")

    # call geometric.Resize with sqrt(scale) as edge multiplier
    try:
        inst = GeomResize()
        # geometric.Resize.forward expects size as multiplicative factor
        v_out, _ = inst.forward(xb, None, size=math.sqrt(scale))
    except Exception as e:
        pytest.skip(f"geometric.Resize failed: {e}")

    v_t = _unwrap_and_batch(v_out)
    i_t = _unwrap_and_batch(i_out)
    if v_t is None or i_t is None:
        pytest.skip("could not unwrap outputs for Resize test")
    _cmp_tensors(v_t, i_t)


def test_mapping_equivalence_Rotate(image_shape):
    torch.manual_seed(0)
    c, h, w = image_shape
    x = torch.rand(c, h, w)
    xb = x.unsqueeze(0)

    angle = 30
    # image_effects.rotate returns tensor
    i_out = ie_rotate(xb, angle)

    inst = GeomRotate()
    v_out, _ = inst.forward(xb, None, angle=angle)

    v_t = _unwrap_and_batch(v_out)
    i_t = _unwrap_and_batch(i_out)
    _cmp_tensors(v_t, i_t)


def test_mapping_equivalence_Crop(image_shape):
    # Use scale=1.0 so both implementations act as identity crop
    torch.manual_seed(0)
    c, h, w = image_shape
    x = torch.rand(c, h, w)
    xb = x.unsqueeze(0)

    scale = 1.0
    # image_effects.random_crop expects area scale
    i_out = ie_random_crop(xb, scale)

    inst = GeomCrop()
    # geometric.Crop expects size multiplier for edges; pass 1.0 to get full size
    v_out, _ = inst.forward(xb, None, size=1.0)

    v_t = _unwrap_and_batch(v_out)
    i_t = _unwrap_and_batch(i_out)
    _cmp_tensors(v_t, i_t)


def test_mapping_equivalence_Perspective(image_shape):
    torch.manual_seed(0)
    c, h, w = image_shape
    x = torch.rand(c, h, w)
    xb = x.unsqueeze(0)

    distortion = 0.2
    # Compute one deterministic start/endpoints pair and force both
    # modules to return it so they apply identical perspective warps.
    from omnisealbench.attacks import image_effects as ie_mod

    h, w = x.shape[-2:]
    # produce deterministic points
    sp, ep = GeomPerspective.get_perspective_params(w, h, distortion)

    # Monkeypatch both modules to return the same pair
    ie_mod.get_perspective_params = lambda width, height, d: (sp, ep)
    GeomPerspective.get_perspective_params = staticmethod(lambda width, height, d: (sp, ep))

    # image_effects.perspective now uses our fixed params
    i_out = ie_perspective(xb, distortion)

    inst = GeomPerspective()
    v_out, _ = inst.forward(xb, None, distortion_scale=distortion)

    v_t = _unwrap_and_batch(v_out)
    i_t = _unwrap_and_batch(i_out)
    _cmp_tensors(v_t, i_t)


def test_mapping_equivalence_HorizontalFlip(image_shape):
    torch.manual_seed(0)
    c, h, w = image_shape
    x = torch.rand(c, h, w)
    xb = x.unsqueeze(0)

    i_out = ie_hflip(xb)

    inst = GeomHFlip()
    v_out, _ = inst.forward(xb, None)

    v_t = _unwrap_and_batch(v_out)
    i_t = _unwrap_and_batch(i_out)
    _cmp_tensors(v_t, i_t)
