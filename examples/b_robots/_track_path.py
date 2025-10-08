import matplotlib.pyplot as plt
import mujoco
import numpy as np

from ariel.simulation.environments import OlympicArena as World

# --- Load your MuJoCo model ---
# Base world
world = World()

# Spawn a test object to validate the environment
xml = r"""
<mujoco>
<worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
</worldbody>
</mujoco>
"""
test_object = mujoco.MjSpec.from_string(xml)
world.spawn(test_object, correct_spawn_for_collisions=True)

# Compile the model and create data
model = world.spec.compile()
data = mujoco.MjData(model)

# --- Camera setup: top-down view ---
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=10000)
# context = mujoco.MjrContext(model)

# Position the camera above the environment
cam.lookat[:] = [0.0, 0.0, 0.0]  # look at origin
cam.distance = 3.0  # distance from target
cam.elevation = -90.0  # looking straight down
cam.azimuth = 0.0

# --- Simulation loop ---
trajectory = []
steps = 2000
body_name = "robot1-world"  # change to your robot part (e.g., "ee", "base")

body_id = model.body(name=body_name)

for _ in range(steps):
    mujoco.mj_step(model, data)
    trajectory.append(data.xpos[body_id][:2].copy())

trajectory = np.array(trajectory)

# --- Render one top-down RGB frame ---
width, height = 480, 480
viewport = mujoco.MjrRect(0, 0, width, height)

# Update scene with camera
mujoco.mjv_updateScene(
    model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
)

# Create RGB buffer
rgb = np.zeros((height, width, 3), dtype=np.uint8)
depth = np.zeros((height, width), dtype=np.float32)

# mujoco.mjr_render(viewport, scene, context)
# mujoco.mjr_readPixels(rgb, depth, viewport, context)

# --- Overlay trajectory on top-down image ---
# Normalize trajectory into image coordinates (manual scaling)
x = trajectory[:, 0]
y = trajectory[:, 1]
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

# Map to pixel space
px = ((x - x_min) / (x_max - x_min + 1e-8) * (width - 1)).astype(int)
py = ((y - y_min) / (y_max - y_min + 1e-8) * (height - 1)).astype(int)
py = height - py  # flip y for image coordinates

plt.imshow(rgb)
plt.plot(px, py, color="red", linewidth=2)
plt.title("Top-down environment + trajectory")
plt.axis("off")
plt.show()
