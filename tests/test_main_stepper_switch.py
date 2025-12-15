import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from geometry.entities import Mesh, Vertex
from main import parse_instructions
from parameters.global_parameters import GlobalParameters
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.conjugate_gradient import ConjugateGradient
from runtime.steppers.gradient_descent import GradientDescent


def create_mesh():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.zeros(3))
    mesh.global_parameters = GlobalParameters()
    mesh.energy_modules = []
    mesh.constraint_modules = []
    return mesh


def test_stepper_switch_between_cg_and_gd(monkeypatch):
    mesh = create_mesh()
    energy_manager = EnergyModuleManager(mesh.energy_modules)
    constraint_manager = ConstraintModuleManager(mesh.constraint_modules)

    stepper = GradientDescent()
    minimizer = Minimizer(mesh, mesh.global_parameters, stepper,
                          energy_manager, constraint_manager)

    called = []

    def fake_minimize(self, n_steps=1):
        called.append(self.stepper.__class__.__name__)
        return {"mesh": self.mesh, "energy": 0.0}

    monkeypatch.setattr(Minimizer, "minimize", fake_minimize)

    instructions = parse_instructions(["cg", "g1", "gd", "g1"])

    for cmd in instructions:
        if cmd == "cg":
            stepper = ConjugateGradient()
            minimizer.stepper = stepper
        elif cmd == "gd":
            stepper = GradientDescent()
            minimizer.stepper = stepper
        elif cmd.startswith("g"):
            cmd = cmd.replace(" ", "")
            minimizer.minimize(n_steps=int(cmd[1:]))

    assert called == ["ConjugateGradient", "GradientDescent"]
