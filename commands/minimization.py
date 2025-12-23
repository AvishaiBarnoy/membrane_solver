import logging

from commands.base import Command
from runtime.steppers.bfgs import BFGS
from runtime.steppers.conjugate_gradient import ConjugateGradient
from runtime.steppers.gradient_descent import GradientDescent
from runtime.topology import detect_vertex_edge_collisions

logger = logging.getLogger("membrane_solver")


class GoCommand(Command):
    """Run N minimization steps (e.g., g, g10)."""

    def execute(self, context, args):
        n_steps = 1
        # Handle 'g 10' or 'g10' (where args might be ['10'])
        if args and args[0].isdigit():
            n_steps = int(args[0])

        logger.debug(
            f"Minimizing for {n_steps} steps using {context.stepper.__class__.__name__}"
        )

        callback = None
        if getattr(context.minimizer, "live_vis", False):
            import matplotlib.pyplot as plt

            from visualization.plotting import plot_geometry

            plt.ion()

            def cb(mesh, i):
                plt.clf()
                ax = plt.axes(projection="3d")
                plot_geometry(mesh, ax=ax, show=False)
                plt.title(f"Step {i}")
                plt.draw()
                plt.pause(0.001)

            callback = cb

        result = context.minimizer.minimize(n_steps=n_steps, callback=callback)
        context.mesh = result["mesh"]

        logger.info(
            f"Minimization complete. Final energy: {result['energy'] if result else 'N/A'}"
        )

        collisions = detect_vertex_edge_collisions(context.mesh)
        if collisions:
            logger.warning(
                f"TOPOLOGY WARNING: {len(collisions)} vertex-edge collisions detected!"
            )


class SetStepperCommand(Command):
    """Switch between CG and GD steppers."""

    def __init__(self, stepper_type):
        self.stepper_type = stepper_type

    def execute(self, context, args):
        if self.stepper_type == "cg":
            logger.info("Switching to Conjugate Gradient stepper.")
            context.stepper = ConjugateGradient()
        elif self.stepper_type == "bfgs":
            logger.info("Switching to BFGS stepper.")
            context.stepper = BFGS()
        elif self.stepper_type == "gd":
            logger.info("Switching to Gradient Descent stepper.")
            context.stepper = GradientDescent()
        context.minimizer.stepper = context.stepper


class HessianCommand(Command):
    """Run a one-off Hessian (BFGS) step without switching the active stepper."""

    def execute(self, context, args):
        steps = 1
        if args and args[0].isdigit():
            steps = max(1, int(args[0]))

        stepper = BFGS()
        for _ in range(steps):
            energy, grad = context.minimizer.compute_energy_and_gradient()
            context.minimizer.project_constraints(grad)
            step_success, context.minimizer.step_size = stepper.step(
                context.mesh,
                grad,
                context.minimizer.step_size,
                context.minimizer.compute_energy,
                constraint_enforcer=context.minimizer._enforce_constraints
                if context.minimizer._has_enforceable_constraints
                else None,
            )
            if not step_success:
                break
        logger.info(
            "Hessian step complete (%d step%s).",
            steps,
            "" if steps == 1 else "s",
        )


class LiveVisCommand(Command):
    """Toggle live visualization."""

    def execute(self, context, args):
        if not hasattr(context.minimizer, "live_vis"):
            context.minimizer.live_vis = False
        context.minimizer.live_vis = not context.minimizer.live_vis
        logger.info(
            f"Live visualization {'enabled' if context.minimizer.live_vis else 'disabled'}"
        )
