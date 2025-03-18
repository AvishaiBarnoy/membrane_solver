# volume.py
# Here goes energy terms relevant for volume of defined bodies


# Pseudo code for volume difference calculation
#   energy should be spread over all relevant facets, scaling by the surface
#   area of the facets.
# if target_volume:
#   if current_vol != target_volume:
#       total_energy += k_vol * (current_volume - target_volume)
