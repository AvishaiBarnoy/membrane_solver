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
        minimizer = getattr(context, "minimizer", None)
        color_by = getattr(minimizer, "vis_color_by", None) if minimizer else None
        show_tilt_arrows = (
            getattr(minimizer, "vis_show_tilt_arrows", False) if minimizer else False
        )
        if args:
            tokens = [str(tok).strip().lower() for tok in args if str(tok).strip()]
            if any(tok in {"tilt", "t", "mag", "abs"} for tok in tokens):
                color_by = "tilt_mag"
            elif any(tok in {"div", "divt"} for tok in tokens):
                color_by = "tilt_div"
            elif any(tok in {"plain", "none", "off"} for tok in tokens):
                color_by = None

            if any(tok in {"noarrows", "noarrow"} for tok in tokens):
                show_tilt_arrows = False
            elif any(tok in {"arrows", "arrow", "quiver"} for tok in tokens):
                show_tilt_arrows = True

            supported = {
                "tilt",
                "t",
                "mag",
                "abs",
                "div",
                "divt",
                "plain",
                "none",
                "off",
                "arrows",
                "arrow",
                "quiver",
                "noarrows",
                "noarrow",
            }
            unknown = [tok for tok in tokens if tok not in supported]
            if unknown:
                print("Usage: s [tilt|div|plain] [arrows|noarrows]")
                return

            if minimizer is not None:
                setattr(minimizer, "vis_color_by", color_by)
                setattr(minimizer, "vis_show_tilt_arrows", show_tilt_arrows)
        else:
            if color_by is None:
                try:
                    import numpy as np

                    tilts = context.mesh.tilts_view()
                    if np.any(np.linalg.norm(tilts, axis=1) > 0):
                        color_by = "tilt_mag"
                        if minimizer is not None:
                            setattr(minimizer, "vis_color_by", color_by)
                except Exception:
                    pass

        # Interactive visualize ("s") is primarily used for inspecting geometry
        # during minimization, so draw edges by default and keep facets opaque.
        plot_geometry(
            context.mesh,
            show_indices=False,
            draw_edges=True,
            transparent=False,
            color_by=color_by,
            show_tilt_arrows=show_tilt_arrows,
        )


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
                target_volume = body.target_volume
                if target_volume is None:
                    target_volume = body.options.get("target_volume")
                print(
                    f"  Body {body_idx}: volume = {body_vol:.6f}, "
                    f"surface area = {body_area:.6f}, "
                    f"surface Rg = {body_rg:.6f}, "
                    f"target volume = {target_volume}"
                )
