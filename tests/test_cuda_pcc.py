import unittest

import numpy as np
import torch

from scma.pcc_torch import solve_pcc_torch


class MatrixPair:
    def __init__(self, matrix, device):
        self.matrix = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        self.forward_calls = 0

    def forward(self, image):
        self.forward_calls += 1
        return (self.matrix @ image.flatten()).reshape(1, -1)

    def adjoint(self, sinogram):
        return (self.matrix.T @ sinogram.flatten()).reshape(1, -1)


class TensorPCCTests(unittest.TestCase):
    def check_device(self, device):
        rng = np.random.default_rng(93)
        matrix = rng.normal(size=(11, 5)).astype(np.float32)
        prior = rng.normal(size=5).astype(np.float32)
        rays = rng.normal(size=11).astype(np.float32)
        metal = np.array([False, True, False, False, False, True, False, True, False, False, False])
        rays[metal] = np.nan
        valid = matrix[~metal].astype(np.float64)
        rho = 0.7
        expected = np.linalg.solve(valid.T @ valid + rho * np.eye(5),
                                   valid.T @ rays[~metal] + rho * prior)
        pair = MatrixPair(matrix, device)
        image = torch.as_tensor(prior, device=device)[None, :]
        measurements = torch.as_tensor(rays, device=device)[None, :]
        trace = torch.as_tensor(metal, device=device)[None, :]
        result = solve_pcc_torch(image, measurements, trace, pair, rho=rho,
                                 relative_tolerance=1e-7)
        self.assertEqual(result.device, image.device)
        self.assertEqual(result.dtype, image.dtype)
        np.testing.assert_allclose(result.cpu().numpy()[0], expected, rtol=2e-5, atol=3e-6)
        before = pair.forward_calls
        result = solve_pcc_torch(image, measurements, torch.ones_like(trace), pair)
        torch.testing.assert_close(result, image, atol=0, rtol=0)
        self.assertEqual(pair.forward_calls, before)

        pair = MatrixPair(np.diag(np.geomspace(1, 100, 30)), device)
        image = torch.zeros((1, 30), device=device)
        y = torch.ones_like(image)
        clear = torch.zeros_like(image, dtype=torch.bool)
        solve_pcc_torch(image, y, clear, pair)
        self.assertEqual(pair.forward_calls, 11)

        pair = MatrixPair(np.eye(30), device)
        torch.testing.assert_close(solve_pcc_torch(image, y, clear, pair), y / 2)
        self.assertEqual(pair.forward_calls, 2)

    def test_tensor_solver_on_cpu(self):
        self.check_device("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_tensor_solver_on_gpu(self):
        self.check_device("cuda")


if __name__ == "__main__":
    unittest.main()
