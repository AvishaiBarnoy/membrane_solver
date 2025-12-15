import numpy as np

from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from parameters.global_parameters import GlobalParameters


def create_quad():

    mesh = Mesh()

    # A unit square in the XY plane
    v0 = Vertex(0, np.array([0, 0, 0]))
    v1 = Vertex(1, np.array([1 , 0, 0]))
    v2 = Vertex(2, np.array([1 , 1, 0]))
    v3 = Vertex(3, np.array([0, 1, 0]))
    vertices = [v0, v1, v2, v3]

    e0 = Edge(1, v0.index, v1.index)
    e1 = Edge(2, v1.index, v2.index)
    e2 = Edge(3, v2.index, v3.index)
    e3 = Edge(4, v3.index, v0.index)
    edges = [e0, e1, e2, e3]

    facet = Facet(0, [e0.index, e1.index, e2.index, e3.index])
    facets = [facet]

    body = Body(0, [facet.index], options={"target_volume": 0})
    bodies = [body]

    mesh = Mesh()
    for v in vertices:
        mesh.vertices[v.index] = v
    for e in edges:
        mesh.edges[e.index] = e
    for f in facets:
        mesh.facets[f.index] = f
    for b in bodies:
        mesh.bodies[b.index] = b

    return mesh

def test_global_parameters_loading():
    # Mock input data
    data_params = {
            "surface_tension": 10.0,
            "volume_stiffness": 500.0,
            "custom_param": 42.0
    }

    # Parse geometry
    mesh = create_quad()
    mesh.global_parameters = GlobalParameters()
    mesh.global_parameters.update(data_params)

    # Check that global_parameters is an instance of GlobalParameters
    assert isinstance(mesh.global_parameters, GlobalParameters), "global_parameters should be an instance of GlobalParameters"

    # Check that predefined parameters are updated correctly
    assert mesh.global_parameters.get("surface_tension") == 10.0, "surface_tension should be updated to 10.0"
    assert mesh.global_parameters.get("volume_stiffness") == 500.0, "volume_stiffness should be updated to 500.0"

    # Check that user-defined parameters are added
    assert mesh.global_parameters.get("custom_param") == 42.0, "custom_param should be added with value 42.0"

    # Check that default parameters are preserved
    assert mesh.global_parameters.get("step_size") == 1e-3, "step_size should retain its default value"

def test_global_parameters_defaults():
    # Initialize GlobalParameters without any input
    gp = GlobalParameters()

    # Check default values
    assert gp.get("surface_tension") == 1.0, "Default surface_tension should be 1.0"
    assert gp.get("volume_stiffness") == 1000.0, "Default volume_stiffness should be 1000.0"
    assert gp.get("step_size") == 1e-3, "Default step_size should be 1e-3"

def test_global_parameters_update():
    # Initialize GlobalParameters with some defaults
    gp = GlobalParameters()

    # Update with new values
    gp.update({
        "surface_tension": 15.0,
        "new_param": 99.0
    })

    # Check updated values
    assert gp.get("surface_tension") == 15.0, "surface_tension should be updated to 15.0"
    assert gp.get("new_param") == 99.0, "new_param should be added with value 99.0"

    # Check that other defaults are preserved
    assert gp.get("volume_stiffness") == 1000.0, "volume_stiffness should retain its default value"
