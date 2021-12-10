import os
import time

import numpy as np
from IPython.display import display, HTML, clear_output
from functools import partial
import matplotlib.pyplot as plt
from pydrake.all import ( 
    AddMultibodyPlantSceneGraph, 
    DiagramBuilder, RigidTransform, RotationMatrix, Box,    
    CoulombFriction, FindResourceOrThrow, FixedOffsetFrame, 
    Parser, PlanarJoint,  PointCloud,
    JointIndex, Simulator, ProcessModelDirectives, LoadModelDirectives,
    ConnectPlanarSceneGraphVisualizer,
    DrakeVisualizer, AddContactMaterial,
    DepthRenderCamera, RenderCameraCore, RgbdSensor, CameraInfo, ClippingRange, DepthRange,
    RandomGenerator, UniformlyRandomRotationMatrix, RollPitchYaw,
    MakeRenderEngineVtk, RenderEngineVtkParams, UnitInertia,AddRigidHydroelasticProperties,AddSoftHydroelasticProperties,
    Sphere, Cylinder, Box, Capsule, Ellipsoid, SpatialInertia, Rgba,ProximityProperties, SpatialVelocity, MeshcatAnimation, MeshcatVisualizerCpp,
    MeshcatVisualizerParams, Meshcat
)
import pydrake

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess

proc_planar, zmq_url_planar, web_url_planar = start_zmq_server_as_subprocess()

proc, zmq_url, web_url = start_zmq_server_as_subprocess()

def clutter_gen():
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0)

    box = Box(10., 10., 10.)
    X_WBox = RigidTransform([0, 0, -5])
    mu = 1.0
    ground_props = ProximityProperties()
    
    AddContactMaterial(hydroelastic_modulus = 5.0e10,dissipation=.2,friction=CoulombFriction(.01,.01) ,
                       properties=ground_props)
    AddRigidHydroelasticProperties(ground_props)
    
    plant.RegisterCollisionGeometry(plant.world_body(), X_WBox, box, "ground", ground_props)
    plant.RegisterVisualGeometry(plant.world_body(), X_WBox, box, "ground", [.9, .9, .9, 1.0])
    planar_joint_frame = plant.AddFrame(FixedOffsetFrame("planar_joint_frame", plant.world_frame(), RigidTransform(RotationMatrix.MakeXRotation(np.pi/2))))
    #FindResourceOrThrow("drake/examples/manipulation_station/models/061_foam_brick.sdf")

    parser = Parser(plant)
#open text file in read mode
    #text_file = open("ball.urdf", "r")
    
    #read whole file to a string
    #sdf = text_file.read()
    #print(sdf)
    #close file
    #text_file.close()
    #sdf = FindResourceOrThrow("ball.urdf")

    #for i in range(1):
        #instance = parser.AddModelFromFile("ball.sdf", f"urdf")
        #plant.AddJoint(PlanarJoint(f"joint{i}", planar_joint_frame, plant.GetFrameByName("base_link", instance), damping=[0,0,0]))
    ball_props = ProximityProperties()
    
    AddContactMaterial(hydroelastic_modulus = 5.0e4,dissipation=0,friction=CoulombFriction(.01,.01) ,
                       properties=ball_props)
    AddSoftHydroelasticProperties(.03,ball_props)
    
    book = plant.AddModelInstance("book")
    book_body = plant.AddRigidBody("book_body", book, 
                                 SpatialInertia(
                                     mass = 0.2, 
                                     p_PScm_E=np.array([0., 0., 0.]),
                                     G_SP_E = UnitInertia(1.0, 1.0, 1.0)))
  
    shape = Sphere(.03)

    print(book)
    if plant.geometry_source_is_registered():
        plant.RegisterCollisionGeometry(book_body, RigidTransform(), shape, "book_body", ball_props)
        plant.RegisterVisualGeometry(book_body, RigidTransform(), shape, "book_body", [.9, .2, .2, 1.0])
        #plant.AddJoint(PlanarJoint(f"joint1", planar_joint_frame, plant.GetFrameByName("book_body", book), damping=[0,0,0]))


    plant.Finalize()
    plant.set_penetration_allowance(0.001)
    
    visualizer = pydrake.systems.meshcat_visualizer.ConnectMeshcatVisualizer(   
        builder, 
        scene_graph=scene_graph, 
        zmq_url=zmq_url_planar)

    visualizer.vis.delete()
    visualizer.set_planar_viewpoint(xmin=-.2, xmax=.2, ymin=-.1, ymax=.6)

    

    diagram = builder.Build()
    simulator = Simulator(diagram)
    plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())

    rs = np.random.RandomState()
    z = 10
    #for i in range(plant.num_joints()):
        #joint = plant.get_joint(JointIndex(i))
        #joint.set_pose(plant_context, [rs.uniform(0, 0), z], rs.uniform(-np.pi/2.0, np.pi/2.0))
        #z += 0.1

    plant.SetFreeBodyPose(plant_context, 
                        plant.GetBodyByName("book_body"),
                        RigidTransform([0, 0.0, 1])) # limit is between 0.5 and 0.55
     
    vel = SpatialVelocity([0,0,0],[.4,0,3])
    plant.SetFreeBodySpatialVelocity(plant.GetBodyByName("book_body"),vel,plant_context)
    simulator.set_target_realtime_rate(1.0)
    return visualizer, simulator
    
   

viz, sim = clutter_gen()

sim.AdvanceTo(3)

