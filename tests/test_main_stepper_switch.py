import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from commands.context import CommandContext
from commands.minimization import SetStepperCommand
from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Mesh, Vertex
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


def test_stepper_switch_between_cg_and_gd():
    mesh = create_mesh()
    energy_manager = EnergyModuleManager(mesh.energy_modules)
    constraint_manager = ConstraintModuleManager(mesh.constraint_modules)

    stepper = GradientDescent()
    minimizer = Minimizer(
        mesh, mesh.global_parameters, stepper, energy_manager, constraint_manager
    )

    ctx = CommandContext(mesh, minimizer, stepper)

    # Initial state is GradientDescent
    assert isinstance(ctx.stepper, GradientDescent)
    assert isinstance(ctx.minimizer.stepper, GradientDescent)

    # Switch to CG
    cmd = SetStepperCommand("cg")
    cmd.execute(ctx, [])

    assert isinstance(ctx.stepper, ConjugateGradient)
    assert isinstance(ctx.minimizer.stepper, ConjugateGradient)

    # Switch back to GD
    cmd = SetStepperCommand("gd")
    cmd.execute(ctx, [])

    assert isinstance(ctx.stepper, GradientDescent)
    assert isinstance(ctx.minimizer.stepper, GradientDescent)
