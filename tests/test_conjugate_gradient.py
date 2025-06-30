import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geometry.entities import Mesh, Vertex
from modules.steppers.conjugate_gradient import ConjugateGradient


def quadratic_energy(v: Vertex) -> float:
    return 0.5 * np.dot(v.position, v.position)


def constant_energy() -> float:
    return 0.0


def test_line_search_success_and_history_update():
    mesh = Mesh()
    v = Vertex(0, np.array([1.0, 0.0, 0.0]))
    mesh.vertices[0] = v

    stepper = ConjugateGradient()

    grad = {0: v.position.copy()}

    success, step_size = stepper.step(mesh, grad, 1.0, lambda: quadratic_energy(v))

    assert success
    assert step_size > 1.0  # grew by gamma
    assert np.allclose(v.position, np.zeros(3))
    assert stepper.iter_count == 1
    assert np.allclose(stepper.prev_grad[0], grad[0])
    assert np.allclose(stepper.prev_dir[0], -grad[0] / np.linalg.norm(grad[0]))


def test_line_search_failure_preserves_history():
    mesh = Mesh()
    v = Vertex(0, np.array([1.0, 0.0, 0.0]))
    mesh.vertices[0] = v

    stepper = ConjugateGradient()

    grad = {0: v.position.copy()}
    success, step_size = stepper.step(mesh, grad, 1.0, lambda: quadratic_energy(v))
    assert success

    # Reset vertex position to attempt another step
    v.position[:] = [1.0, 0.0, 0.0]

    old_grad = stepper.prev_grad.copy()
    old_dir = stepper.prev_dir.copy()
    old_iter = stepper.iter_count

    grad2 = {0: v.position.copy()}
    success2, step_size2 = stepper.step(mesh, grad2, step_size, constant_energy)

    assert not success2
    assert step_size2 == step_size
    assert np.allclose(v.position, [1.0, 0.0, 0.0])
    assert stepper.prev_grad == old_grad
    assert stepper.prev_dir == old_dir
    assert stepper.iter_count == old_iter

