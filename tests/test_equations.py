import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scma.flow import distance_to_metal, dynamic_weights, sample, training_loss
from scma.model import VelocityUNet
from scma.pcc import solve_pcc


class MatrixProjector:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.forward_calls = 0

    def forward(self, image):
        self.forward_calls += 1
        return (self.matrix @ image.ravel())[None, :]

    def adjoint(self, sinogram):
        return (self.matrix.T @ sinogram.ravel())[None, :]


class FlowTests(unittest.TestCase):
    def test_pixel_distances_and_gaussian_widths(self):
        mask = np.array([[1, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(distance_to_metal(mask),
                                   [[0, 1, 2], [1, math.sqrt(2), math.sqrt(5)]])
        self.assertTrue(np.isposinf(distance_to_metal(np.zeros_like(mask))).all())

        distance = torch.tensor([[0.0, 128.0], [0.0, 72.0], [0.0, 16.0]]).view(3, 1, 1, 2)
        metal = (distance == 0).float()
        t = torch.tensor([0.0, 0.5, 1.0])
        weights = dynamic_weights(metal, distance, t)
        expected = torch.tensor([2.0, 1.0 + math.exp(-0.5)]).expand(3, 2)
        torch.testing.assert_close(weights.reshape(3, 2), expected)
        empty = dynamic_weights(torch.zeros_like(metal), torch.full_like(distance, math.inf), t)
        torch.testing.assert_close(empty, torch.ones_like(empty))

    def test_training_time_direction_and_velocity_target(self):
        noise = torch.full((3, 1, 2, 2), -2.0)
        clean = torch.full_like(noise, 6.0)
        condition = torch.full_like(clean, 0.5)
        time = torch.tensor([0.0, 0.25, 1.0])

        def exact_field(state, li, t):
            self.assertIs(li, condition)
            torch.testing.assert_close(t, time)
            torch.testing.assert_close(state[:, 0, 0, 0], torch.tensor([-2., 0., 6.]))
            return torch.full_like(state, 8.0)

        loss = training_loss(exact_field, clean, condition, torch.ones_like(clean),
                             torch.zeros_like(clean), t=time, noise=noise)
        self.assertEqual(loss.item(), 0.0)

    def test_loss_normalizes_each_slice_before_averaging(self):
        clean = torch.zeros(2, 1, 1, 2)
        mask = torch.ones_like(clean)
        mask[1] = 0
        distance = torch.zeros_like(clean)
        distance[1] = math.inf
        predicted = torch.tensor([0., 2., 3., 3.]).view_as(clean)
        loss = training_loss(lambda *_: predicted, clean, clean, mask, distance,
                             t=torch.tensor([0.2, 0.8]), noise=clean)
        self.assertAlmostEqual(loss.item(), (2.0 + 9.0) / 2)

    def test_network_input_order_and_continuous_time_encoding(self):
        state = torch.zeros(2, 1, 32, 32)
        condition = torch.ones_like(state)
        time = torch.tensor([0.12345, 0.98765])

        class Capture(nn.Module):
            def forward(self, inputs, encoded_time):
                self.inputs, self.encoded_time = inputs, encoded_time
                return inputs[:, :1]

        with patch("scma.model.UNet2DModel", Capture):
            model = VelocityUNet()
        model(state, condition, time)
        torch.testing.assert_close(model.unet.inputs, torch.cat([state, condition], 1))
        torch.testing.assert_close(model.unet.encoded_time, torch.tensor([123.45, 987.65]))

    def test_default_50_stage_euler_and_endpoint_feedback(self):
        li = torch.ones(1, 1, 2, 2)
        times, endpoints = [], []

        def velocity(state, condition, t):
            self.assertIs(condition, li)
            times.append(t.item())
            return 2.0 * condition

        def correct(x):
            endpoints.append(x.flatten()[0].item())
            return x + 1.0

        result = sample(velocity, li, noise=torch.zeros_like(li), correct_endpoint=correct)
        np.testing.assert_allclose(times, np.arange(50) / 50, atol=1e-7)
        np.testing.assert_allclose(endpoints, 2 + 0.5 * np.arange(50), atol=3e-5)
        torch.testing.assert_close(result, torch.full_like(li, 27.0))
        pure_fm = sample(velocity, li, steps=2, noise=torch.zeros_like(li))
        torch.testing.assert_close(pure_fm, torch.full_like(li, 2.0))

    def test_pcc_feedback_strength(self):
        prior = torch.zeros(1, 1, 2, 2)

        def zero_velocity(state, condition, time):
            return torch.zeros_like(state)

        def correct(endpoint):
            return endpoint + 1.0

        for omega in (0.0, 1.0, 1.25, 1.5, 2.0):
            with self.subTest(omega=omega):
                result = sample(zero_velocity, prior, noise=prior, steps=2,
                                correct_endpoint=correct, omega=omega)
                torch.testing.assert_close(result, torch.full_like(prior, 2 * omega))

        for omega in (-0.5, math.nan, math.inf, -math.inf):
            with self.subTest(omega=omega), self.assertRaises(ValueError):
                sample(zero_velocity, prior, noise=prior, steps=2,
                       correct_endpoint=correct, omega=omega)


class ProjectionTests(unittest.TestCase):
    def test_masked_cg_matches_direct_solution(self):
        matrix = np.array([[1., 2., -1.], [2., 1., 0.], [3., -1., 0.], [1., 0., 2.]])
        projector = MatrixProjector(matrix)
        prior = np.array([[0.2, 0.3, 0.1]])
        y = np.array([[2., np.nan, 3., 4.]])
        trace = np.array([[0, 1, 0, 0]])
        rho = 0.75
        keep = trace.ravel() == 0
        reliable_matrix = matrix[keep]
        expected = np.linalg.solve(reliable_matrix.T @ reliable_matrix + rho * np.eye(3),
                                   reliable_matrix.T @ y.ravel()[keep] + rho * prior.ravel())
        actual = solve_pcc(prior, y, trace, projector, rho=rho)
        np.testing.assert_allclose(actual.ravel(), expected, rtol=1e-10, atol=1e-12)

    def test_zero_initial_residual(self):
        matrix = np.array([[2., 1.], [1., 3.]])
        projector = MatrixProjector(matrix)
        prior = np.array([[4., -1.]])
        y = (matrix @ prior.ravel())[None, :]
        actual = solve_pcc(prior, y, np.zeros_like(y), projector)
        np.testing.assert_array_equal(prior, actual)
        self.assertEqual(projector.forward_calls, 1)

    def test_stops_at_relative_initial_residual(self):
        matrix = np.diag([1., 1.0001])
        for amplitude in (1e-6, 1e4):
            with self.subTest(amplitude=amplitude):
                projector = MatrixProjector(matrix)
                y = np.full((1, 2), amplitude)
                prior = np.zeros_like(y)
                result = solve_pcc(prior, y, np.zeros_like(y), projector)
                rhs = matrix.T @ y.ravel()
                residual = rhs - (matrix.T @ matrix + np.eye(2)) @ result.ravel()
                ratio = np.linalg.norm(residual) / np.linalg.norm(rhs)
                self.assertGreater(ratio, 1e-6)
                self.assertLess(ratio, 1e-4)
                self.assertEqual(projector.forward_calls, 2)

    def test_iteration_cap(self):
        matrix = np.diag(np.geomspace(1., 100., 24))
        projector = MatrixProjector(matrix)
        y = np.ones((1, 24))
        solve_pcc(np.zeros_like(y), y, np.zeros_like(y), projector)
        self.assertEqual(projector.forward_calls, 11)


if __name__ == "__main__":
    unittest.main()
