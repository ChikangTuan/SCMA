from __future__ import annotations

import numpy as np


def interpolate_trace(measurement: np.ndarray, metal_trace: np.ndarray) -> np.ndarray:
    values = np.asarray(measurement, dtype=np.float32)
    trace = np.asarray(metal_trace)
    if values.ndim != 2 or values.shape != trace.shape or not np.isfinite(values).all():
        raise ValueError("Expected a finite 2-D measurement and a same-shaped trace")
    if not np.isin(trace, (0, 1)).all():
        raise ValueError("Expected a binary metal trace")
    out = values.copy()
    detectors = np.arange(values.shape[1])
    for view, missing in enumerate(trace.astype(bool)):
        if not missing.any():
            continue
        valid = ~missing
        if not valid.any():
            raise ValueError(f"View {view} contains no reliable detector bins for LI")
        out[view, missing] = np.interp(detectors[missing], detectors[valid], values[view, valid])
    return out


class FanBeam640:
    image_shape = (416, 416)
    sinogram_shape = (640, 641)

    def __init__(self):
        import astra

        self._astra = astra
        side = 512 * 0.03
        pixel_spacing = side / self.image_shape[0]
        source_to_origin = 1075 * pixel_spacing
        detector_pitch = 2 * np.sqrt(2) * side / self.sinogram_shape[1]
        n_views, n_detectors = self.sinogram_shape
        angles = (np.arange(n_views, dtype=np.float32) + 0.5) * (2 * np.pi / n_views)
        self.vol_geom = astra.create_vol_geom(*self.image_shape, -side/2, side/2, -side/2, side/2)
        self.proj_geom = astra.create_proj_geom(
            "fanflat", detector_pitch, n_detectors, angles, source_to_origin, source_to_origin
        )
        self._projector_id = astra.create_projector("line_fanflat", self.proj_geom, self.vol_geom)
        try:
            self._operator = astra.OpTomo(self._projector_id)
        except Exception:
            astra.projector.delete(self._projector_id)
            raise

    def close(self):
        if self._projector_id is not None:
            self._astra.projector.delete(self._projector_id)
            self._projector_id = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _array(self, value, shape):
        if self._projector_id is None:
            raise RuntimeError("The ASTRA projector has been closed")
        value = np.ascontiguousarray(value, dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Expected a finite array with shape {shape}")
        return value

    def forward(self, image):
        image = self._array(image, self.image_shape)
        return self._operator.FP(np.ascontiguousarray(np.rot90(image, 1)))

    def adjoint(self, sinogram):
        sinogram = self._array(sinogram, self.sinogram_shape)
        return np.ascontiguousarray(np.rot90(self._operator.BP(sinogram), -1))

    def fbp(self, sinogram):
        sinogram = self._array(sinogram, self.sinogram_shape)
        astra = self._astra
        if not astra.use_cuda():
            raise RuntimeError("Generating fan-beam LI images requires ASTRA with CUDA support")
        sino_id = astra.data2d.create("-sino", self.proj_geom, sinogram)
        rec_id = alg_id = None
        try:
            rec_id = astra.data2d.create("-vol", self.vol_geom)
            cfg = astra.astra_dict("FBP_CUDA")
            cfg.update(ProjectionDataId=sino_id, ReconstructionDataId=rec_id,
                       option={"FilterType": "ram-lak"})
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            return np.ascontiguousarray(np.rot90(astra.data2d.get(rec_id), -1))
        finally:
            if alg_id is not None:
                astra.algorithm.delete(alg_id)
            if rec_id is not None:
                astra.data2d.delete(rec_id)
            astra.data2d.delete(sino_id)

    def prepare_li(self, measurement, metal_mask):
        metal_mask = self._array(metal_mask, self.image_shape)
        if not np.isin(metal_mask, (0, 1)).all():
            raise ValueError("Expected a binary image-domain mask")
        trace = self.forward(metal_mask) > 0
        return self.fbp(interpolate_trace(measurement, trace)), trace
