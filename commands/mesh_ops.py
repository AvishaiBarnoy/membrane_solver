import logging

import numpy as np

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


class PerturbCommand(Command):
    """Add small random noise to vertex positions."""

    def execute(self, context, args):
        scale = 0.01
        if args:
            try:
                scale = float(args[0])
            except ValueError:
                pass

        logger.info(f"Perturbing vertex positions (scale={scale})...")
        for v in context.mesh.vertices.values():
            if not v.fixed:
                v.position += scale * np.random.normal(size=3)
        context.mesh.increment_version()
        context.mesh.build_position_cache()


class SnapshotCommand(Command):
    """
    Snapshot geometric properties to current values.
    Syntax: snapshot [edges|facets|all] [where key=value]
    Example: snapshot edges where preset=paper
    """

    def execute(self, context, args):
        if not args:
            print("Usage: snapshot [edges|facets|all] [where key=value]")
            return

        target_type = args[0].lower()
        filter_key = None
        filter_val = None

        if "where" in args:
            idx = args.index("where")
            if len(args) > idx + 1:
                kv = args[idx + 1]
                if "=" in kv:
                    filter_key, filter_val = kv.split("=", 1)
                else:
                    # Handle 'where preset paper'
                    filter_key = kv
                    if len(args) > idx + 2:
                        filter_val = args[idx + 2]

        mesh = context.mesh
        edges_to_fix = []
        facets_to_fix = []

        if target_type in ["edges", "edge", "all"]:
            edges_to_fix = list(mesh.edges.values())
        if target_type in ["facets", "facet", "faces", "face", "all"]:
            facets_to_fix = list(mesh.facets.values())

        def matches(obj):
            if not filter_key:
                return True
            opts = getattr(obj, "options", {}) or {}
            val = opts.get(filter_key)
            return str(val) == str(filter_val)

        logger.info(f"Fixing {target_type} properties...")

        fixed_e = 0
        fixed_f = 0

        for edge in edges_to_fix:
            if matches(edge):
                if not edge.options:
                    edge.options = {}
                L = edge.compute_length(mesh)
                edge.options["target_length"] = L
                # Auto-enable energy penalty
                if "energy" not in edge.options:
                    edge.options["energy"] = []
                if "edge_length_penalty" not in edge.options["energy"]:
                    if isinstance(edge.options["energy"], list):
                        edge.options["energy"].append("edge_length_penalty")
                    else:
                        edge.options["energy"] = [
                            edge.options["energy"],
                            "edge_length_penalty",
                        ]
                fixed_e += 1

        for facet in facets_to_fix:
            if matches(facet):
                if not facet.options:
                    facet.options = {}
                A = facet.compute_area(mesh)
                facet.options["target_area"] = A
                # Auto-enable constraint
                if "constraints" not in facet.options:
                    facet.options["constraints"] = []
                if "fix_facet_area" not in facet.options["constraints"]:
                    if isinstance(facet.options["constraints"], list):
                        facet.options["constraints"].append("fix_facet_area")
                    else:
                        facet.options["constraints"] = [
                            facet.options["constraints"],
                            "fix_facet_area",
                        ]

                # Ensure the module is registered globally on the mesh
                if "fix_facet_area" not in mesh.constraint_modules:
                    mesh.constraint_modules.append("fix_facet_area")

                fixed_f += 1

        logger.info(f"Fixed {fixed_e} edges and {fixed_f} facets.")

        # Tell the minimizer to refresh its module lists if it exists
        if context.minimizer:
            context.minimizer.refresh_modules()

        context.mesh.increment_version()
