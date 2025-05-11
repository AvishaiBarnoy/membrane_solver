# modules/constraints/fixed_volume.py

# TODO: this should be body specific!!!
def FixedVolume(mesh, gp, resolver):
    # TODO: what is gp?
    # TODO: what is resolver?
    l = gp.volume_multiplier
    E = l*(mesh.compute_total_volume() - gp.target_volume)
    grad = {… gradient of volume w.r.t. vertex positions …}
    return E, grad

