import pydrake.all
from ipywidgets import FloatSlider, ToggleButton
from pydrake.all import (AddMultibodyPlantSceneGraph, DiagramBuilder,
                         LinearQuadraticRegulator, Parser, Saturation,
                         SceneGraph, Simulator, WrapToSystem,plot_system_graphviz)

from pydrake.common import FindResourceOrThrow
from pydrake.common.containers import namedview
from pydrake.examples.acrobot import AcrobotGeometry, AcrobotPlant
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.jupyter_widgets import WidgetSystem
import matplotlib.pyplot as plt
builder = DiagramBuilder()
plant, _ = AddMultibodyPlantSceneGraph(builder, 0.0)
Parser(plant).AddModelFromFile(
    FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf"))
plant.Finalize()
diagram = builder.Build()
simulator = Simulator(diagram)

print("test")
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess

proc_planar, zmq_url_planar, web_url_planar = start_zmq_server_as_subprocess()

proc, zmq_url, web_url = start_zmq_server_as_subprocess()



builder = DiagramBuilder()
acrobot = builder.AddSystem(AcrobotPlant())

# Setup visualization
scene_graph = builder.AddSystem(SceneGraph())

AcrobotGeometry.AddToBuilder(builder, acrobot.get_output_port(0), scene_graph)
visualizer = pydrake.systems.meshcat_visualizer.ConnectMeshcatVisualizer(
    builder, 
    scene_graph=scene_graph, 
    zmq_url=zmq_url_planar)

visualizer.vis.delete()
visualizer.set_planar_viewpoint(xmin=-4, xmax=4, ymin=-4, ymax=4)

# Setup slider input
slider = FloatSlider(value=0.0, min=-5., max=5., step=0.1, description='u', continuous_update=True)
torque_system = builder.AddSystem(WidgetSystem([slider]))
builder.Connect(torque_system.get_output_port(0), acrobot.get_input_port(0))

diagram = builder.Build()

plt.figure(figsize=(500, 100))

plot_system_graphviz(diagram)
plt.show()

# Set up a simulator to run this diagram
simulator = Simulator(diagram)
context = simulator.get_mutable_context()

stop_button = ToggleButton(value=False, description='Stop Simulation')

# Set the initial conditions
context.SetContinuousState([1., 0, 0, 0]) # theta1, theta2, theta1dot, theta2dot
context.SetTime(0.0)

simulator.set_target_realtime_rate(1.0)

while not stop_button.value:
    simulator.AdvanceTo(simulator.get_context().get_time() + 1.0)
stop_button.value = False
