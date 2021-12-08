import numpy as np

from pydrake.multibody.parsing import (Parser, ProcessModelDirectives,
                                       LoadModelDirectives)
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder

from manipulation.scenarios import AddRgbdSensors
from manipulation.utils import FindResource, AddPackagePaths


def MustardExampleSystem():
    builder = DiagramBuilder()

    # Create the physics engine + scene graph.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    AddPackagePaths(parser)
    ProcessModelDirectives(
        LoadModelDirectives(FindResource("models/mustard_w_cameras.yaml")),
        plant, parser)

    plant.Finalize()

    # Add a visualizer just to help us see the object.
    use_meshcat = False
    if use_meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph))
        builder.Connect(scene_graph.get_query_output_port(),
                        meshcat.get_geometry_query_input_port())

    AddRgbdSensors(builder, plant, scene_graph)

    diagram = builder.Build()
    diagram.set_name("depth_camera_demo_system")
    return diagram
