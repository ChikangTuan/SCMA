import numpy as np
import torch

from .geometry import FanBeam640


def _device_csr(matrix, device):
    values = torch.from_numpy(matrix.data.astype(np.float32, copy=False)).to(device)
    rows = torch.from_numpy(matrix.indptr.astype(np.int32, copy=False)).to(device)
    columns = torch.from_numpy(matrix.indices.astype(np.int32, copy=False)).to(device)
    return torch.sparse_csr_tensor(rows, columns, values, size=matrix.shape)


class CudaFanBeam640:
    image_shape = FanBeam640.image_shape
    sinogram_shape = FanBeam640.sinogram_shape

    def __init__(self, device):
        import astra

        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("CUDA PCC requires a CUDA device")
        self.blocks = []
        views, detectors = self.sinogram_shape
        with FanBeam640() as native:
            for start in range(0, views, 64):
                stop = min(start + 64, views)
                geometry = dict(native.proj_geom)
                geometry["ProjectionAngles"] = native.proj_geom["ProjectionAngles"][start:stop]
                pid = astra.create_projector("line_fanflat", geometry, native.vol_geom)
                try:
                    mid = astra.projector.matrix(pid)
                    try:
                        matrix = astra.matrix.get(mid).astype(np.float32, copy=False)
                    finally:
                        astra.matrix.delete(mid)
                finally:
                    astra.projector.delete(pid)
                matrix.sort_indices()
                self.blocks.append((slice(start * detectors, stop * detectors),
                                    _device_csr(matrix, self.device),
                                    _device_csr(matrix.T.tocsr(), self.device)))

    def forward(self, image):
        if tuple(image.shape) != self.image_shape:
            raise ValueError("expected a single 416 by 416 attenuation image")
        vector = torch.rot90(image, 1, (0, 1)).reshape(-1).float()
        return torch.cat([torch.mv(matrix, vector) for _, matrix, _ in self.blocks]).reshape(self.sinogram_shape)

    def adjoint(self, sinogram):
        if tuple(sinogram.shape) != self.sinogram_shape:
            raise ValueError("expected a 640-view, 641-detector sinogram")
        vector = sinogram.reshape(-1).float()
        result = vector.new_zeros(self.image_shape).reshape(-1)
        for section, _, transpose in self.blocks:
            result.add_(torch.mv(transpose, vector[section]))
        return torch.rot90(result.reshape(self.image_shape), -1, (0, 1)).contiguous()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.blocks.clear()
