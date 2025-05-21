# modules/constraints/volume.py

    for body in mesh.bodies.values():
        V_target = body.target_volume
        V_actual = body.compute_volume(mesh)
        deltaV = V_actual - V_target

        if abs(deltaV) < tol:
            continue

        grad = body.compute_volume_gradient(mesh)
        norm2 = sum(np.dot(g, g) for g in grad.values()) + 1e-12
        λ = deltaV / norm2

        for vidx, v in mesh.vertices.items():
            if vidx in grad and not v.fixed:
                v.position -= λ * grad[vidx]

