import numpy as np
from geometry.entities import Mesh, Vertex
from parameters.global_parameters import GlobalParameters
from runtime.energy_manager import EnergyModuleManager
from runtime.constraint_manager import ConstraintModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def create_mesh():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.zeros(3))
    mesh.global_parameters = GlobalParameters()
    mesh.energy_modules = []
    mesh.constraint_modules = []
    return mesh


def test_minimize_calls_callback():
    mesh = create_mesh()
    energy_manager = EnergyModuleManager(mesh.energy_modules)
    constraint_manager = ConstraintModuleManager(mesh.constraint_modules)
    stepper = GradientDescent()
    minimizer = Minimizer(mesh, mesh.global_parameters, stepper,
                          energy_manager, constraint_manager)

    called = []

    def cb(m):
        called.append(id(m))

    minimizer.minimize(n_steps=1, callback=cb)

    # Should be called at least once
    assert called, "callback was not invoked"
