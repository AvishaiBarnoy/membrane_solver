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
            from visualization.plotting import update_live_vis

            state = getattr(context.minimizer, "live_vis_state", None)

            def cb(mesh, i):
                nonlocal state
                state = update_live_vis(mesh, state=state, title=f"Step {i}")
                context.minimizer.live_vis_state = state

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
        for i in range(steps):
            energy, grad = context.minimizer.compute_energy_and_gradient()
            context.minimizer.project_constraints(grad)
            if not getattr(context.minimizer, "quiet", False):
                total_area = sum(
                    facet.compute_area(context.mesh)
                    for facet in context.mesh.facets.values()
                )
                print(
                    f"Hess {i + 1:4d}: Area = {total_area:.5f}, "
                    f"Energy = {energy:.5f}, "
                    f"Step Size  = {context.minimizer.step_size:.2e}"
                )
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
            if getattr(context.minimizer, "live_vis", False):
                from visualization.plotting import update_live_vis

                state = getattr(context.minimizer, "live_vis_state", None)
                state = update_live_vis(context.mesh, state=state, title="Hessian step")
                context.minimizer.live_vis_state = state
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
        if context.minimizer.live_vis:
            context.minimizer.live_vis_state = None
        else:
            state = getattr(context.minimizer, "live_vis_state", None)
            if state and "fig" in state:
                import matplotlib.pyplot as plt

                plt.close(state["fig"])
            context.minimizer.live_vis_state = None
        logger.info(
            f"Live visualization {'enabled' if context.minimizer.live_vis else 'disabled'}"
        )
