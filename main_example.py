# main_example.py

from modules.minimizer import Minimizer
from modules.steppers.gradient_descent import GradientDescent
from modules.steppers.conjugate_gradient import ConjugateGradient
from geometry.geom_io import load_data, parse_geometry, save_geometry

import sys

# Build mesh & global_params from sample_geometry.json…
SAMPLE_FILE = "./meshes/sample_geometry.json"
data = load_data(SAMPLE_FILE)
mesh = parse_geometry(data)
gp = mesh.global_parameters

# Pick a stepper based on user input or config:
if gp.get("algorithm", "cg"):
    stepper = ConjugateGradient()
else:
    stepper = GradientDescent()

print(stepper)

engine = Minimizer(mesh, gp, stepper=stepper, step_size=gp.step_size)
print(engine)
sys.exit()
engine.minimize()


# If you want to switch mid‐run:
engine.stepper = ConjugateGradient()
engine.step_size = 1e-2
engine.minimize(max_iter=200)


