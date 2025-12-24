import logging

from commands.base import Command
from runtime.equiangulation import equiangulate_mesh
from runtime.refinement import refine_polygonal_facets, refine_triangle_mesh
from runtime.vertex_average import vertex_average

logger = logging.getLogger("membrane_solver")


class RefineCommand(Command):
    def execute(self, context, args):
        count = 1
        if args and args[0].isdigit():
            count = int(args[0])

        for i in range(count):
            logger.info("Refining mesh... (%d/%d)", i + 1, count)
            context.mesh = refine_polygonal_facets(context.mesh)
            context.mesh = refine_triangle_mesh(context.mesh)
            context.minimizer.mesh = context.mesh
            context.minimizer.enforce_constraints_after_mesh_ops(context.mesh)
            if getattr(context.minimizer, "live_vis", False):
                from visualization.plotting import update_live_vis

                state = getattr(context.minimizer, "live_vis_state", None)
                state = update_live_vis(
                    context.mesh, state=state, title=f"Refine {i + 1}/{count}"
                )
                context.minimizer.live_vis_state = state
        logger.info("Mesh refinement complete after %d pass(es).", count)


class VertexAverageCommand(Command):
    def execute(self, context, args):
        n_passes = 1
        if args and args[0].isdigit():
            n_passes = int(args[0])

        for _ in range(n_passes):
            vertex_average(context.mesh)
        logger.info("Vertex averaging done.")
        context.minimizer.enforce_constraints_after_mesh_ops(context.mesh)
        if getattr(context.minimizer, "live_vis", False):
            from visualization.plotting import update_live_vis

            state = getattr(context.minimizer, "live_vis_state", None)
            state = update_live_vis(context.mesh, state=state, title="Vertex average")
            context.minimizer.live_vis_state = state


class EquiangulateCommand(Command):
    def execute(self, context, args):
        logger.info("Starting equiangulation...")
        context.mesh = equiangulate_mesh(context.mesh)
        context.minimizer.mesh = context.mesh
        context.minimizer.enforce_constraints_after_mesh_ops(context.mesh)
        logger.info("Equiangulation complete.")
        if getattr(context.minimizer, "live_vis", False):
            from visualization.plotting import update_live_vis

            state = getattr(context.minimizer, "live_vis_state", None)
            state = update_live_vis(context.mesh, state=state, title="Equiangulate")
            context.minimizer.live_vis_state = state
