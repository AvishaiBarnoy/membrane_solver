import logging

from commands.base import Command
from geometry.geom_io import save_geometry
from visualization.plotting import plot_geometry

logger = logging.getLogger("membrane_solver")


class SaveCommand(Command):
    def execute(self, context, args):
        filename = args[0] if args else "interactive.temp"
        save_geometry(context.mesh, filename)
        logger.info(f"Saved geometry to {filename}")


class VisualizeCommand(Command):
    def execute(self, context, args):
        plot_geometry(context.mesh, show_indices=False)


class PropertiesCommand(Command):
    def execute(self, context, args):
        mesh = context.mesh
        total_area = mesh.compute_total_surface_area()
        total_volume = mesh.compute_total_volume()

        print("=== Physical Properties ===")
        print(f"Vertices: {len(mesh.vertices)}")
        print(f"Edges   : {len(mesh.edges)}")
        print(f"Facets  : {len(mesh.facets)}")
        print(f"Bodies  : {len(mesh.bodies)}")
        print()
        print(f"Total surface area: {total_area:.6f}")
        print(f"Total volume      : {total_volume:.6f}")
        total_rg = mesh.compute_surface_radius_of_gyration()
        print(f"Surface radius of gyration: {total_rg:.6f}")

        if mesh.bodies:
            print()
            print("Per‑body properties:")
            for body_idx, body in mesh.bodies.items():
                body_vol = body.compute_volume(mesh)
                body_area = body.compute_surface_area(mesh)
                body_rg = mesh.compute_surface_radius_of_gyration(body.facet_indices)
                print(
                    f"  Body {body_idx}: volume = {body_vol:.6f}, "
                    f"surface area = {body_area:.6f}, "
                    f"surface Rg = {body_rg:.6f}"
                )
