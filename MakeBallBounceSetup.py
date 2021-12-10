from IPython.display import display, SVG
from debugpy import debug_this_thread
import numpy as np
import os
import pydot
import sys

from pydrake.all import (
    Adder, AddMultibodyPlantSceneGraph, Demultiplexer, DiagramBuilder, 
    InverseDynamicsController, MakeMultibodyStateToWsgStateSystem, 
    MeshcatVisualizerCpp, MultibodyPlant,Parser, PassThrough, 
    SchunkWsgPositionController, StateInterpolatorWithDiscreteDerivative,
    RigidTransform,RollPitchYaw,ConnectMeshcatVisualizer, Simulator,RevoluteJoint,
    AddContactMaterial,CoulombFriction,AddRigidHydroelasticProperties,ProximityProperties,AddSoftHydroelasticProperties,Sphere,
    SpatialInertia,UnitInertia
)
from manipulation.meshcat_cpp_utils import StartMeshcat
from manipulation.scenarios import AddIiwa, AddWsg, AddRgbdSensors
from manipulation.utils import FindResource


if  os.getenv("DISPLAY") is None:
    from pyvirtualdisplay import Display
    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()


def CreateControllerPlant(time_step = 0.002):
    plant = MultibodyPlant(time_step=time_step)
    iiwa = AddIiwa(plant)
    
    #Remove the WSG Manipulator
    #wsg = AddWsg(plant, iiwa)
    # Set default positions:
    q0 = [0, np.pi/3, 0, -np.pi/2, 0, -np.pi/3, 0]
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint,RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1
    parser = Parser(plant)
    parser.AddModelFromFile("models/paddle.sdf")
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7"), plant.GetFrameByName("base_link"), RigidTransform(RollPitchYaw(0, -np.pi/2, 0), [0, 0, 0.25]))
    plant.Finalize()

    return plant

def MakeManipulationStation(time_step=0.00):
    builder = DiagramBuilder()

    # Add (only) the iiwa, WSG, and cameras to the scene.
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=time_step)
    iiwa = AddIiwa(plant)
    
    #Remove the WSG Manipulator
    #wsg = AddWsg(plant, iiwa)
    # Set default positions:
    q0 = [0, np.pi/3, 0, -np.pi/2, 0, -np.pi/3, 0]
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint,RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1
    parser = Parser(plant)
    #parser.AddModelFromFile(
    #    FindResource("models/camera_box.sdf"), "camera0")

    parser.AddModelFromFile("models/floor.sdf")
    parser.AddModelFromFile("models/paddle.sdf")
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7"), plant.GetFrameByName("base_link"), RigidTransform(RollPitchYaw(0, -np.pi/2, 0), [0, 0, 0.25]))

    #Add Properties to ground + Paddle to make bouncy
    

   
    ins = scene_graph.model_inspector()


    floor_coll = plant.GetCollisionGeometriesForBody(plant.GetBodyByName("floor"))


    floor_props = ins.GetProximityProperties(floor_coll[0])
    floor_props.UpdateProperty("material","coulomb_friction",CoulombFriction(0.1,0.1))
    floor_props.UpdateProperty("material","hunt_crossley_dissipation",float(0.0))
    floor_props.UpdateProperty('material',"point_contact_stiffness",float(5e10))
    AddRigidHydroelasticProperties(floor_props)


    paddle_coll = plant.GetCollisionGeometriesForBody(plant.GetBodyByName("base_link"))

    paddle_props = ins.GetProximityProperties(paddle_coll[0])
    paddle_props.UpdateProperty("material","coulomb_friction",CoulombFriction(0.1,0.1))
    paddle_props.UpdateProperty("material","hunt_crossley_dissipation",float(0.0))
    paddle_props.UpdateProperty('material',"point_contact_stiffness",float(5e10))
    AddRigidHydroelasticProperties(paddle_props)

    ball_props = ProximityProperties()
        
    AddContactMaterial(hydroelastic_modulus = 5.0e4,dissipation=0,friction=CoulombFriction(.01,.01) ,
                        properties=ball_props)
    AddSoftHydroelasticProperties(.1,ball_props)


    ball = plant.AddModelInstance("ball")
    ball_body = plant.AddRigidBody("ball_body", ball, 
                                    SpatialInertia(
                                        mass = 0.01, 
                                        p_PScm_E=np.array([0., 0., 0.]),
                                        G_SP_E = UnitInertia(1.0, 1.0, 1.0)))

    shape = Sphere(.1)

    if plant.geometry_source_is_registered():
        plant.RegisterCollisionGeometry(ball_body, RigidTransform(), shape, "ball_body", ball_props)
        plant.RegisterVisualGeometry(ball_body, RigidTransform(), shape, "ball_body", [.9, .2, .2, 1.0])


    plant.Finalize()


    num_iiwa_positions = plant.num_positions(iiwa)

    # I need a PassThrough system so that I can export the input port.
    iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
    builder.ExportInput(iiwa_position.get_input_port(), "iiwa_position")
    builder.ExportOutput(iiwa_position.get_output_port(), "iiwa_position_command")
    
    # Export the iiwa "state" outputs.
    demux = builder.AddSystem(Demultiplexer(
        2 * num_iiwa_positions, num_iiwa_positions))
    builder.Connect(plant.get_state_output_port(iiwa), demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(0), "iiwa_position_measured")
    builder.ExportOutput(demux.get_output_port(1), "iiwa_velocity_estimated")
    builder.ExportOutput(plant.get_state_output_port(iiwa), "iiwa_state_estimated")

    # Make the plant for the iiwa controller to use.
    controller_plant = MultibodyPlant(time_step=0)
    AddIiwa(controller_plant)
    #AddWsg(controller_plant, controller_iiwa, welded=True)
    controller_plant.Finalize()

    # Add the iiwa controller
    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(
            controller_plant,
            kp=[1000]*num_iiwa_positions,
            ki=[4]*num_iiwa_positions,
            kd=[40]*num_iiwa_positions,
            has_reference_acceleration=True))
    iiwa_controller.set_name("iiwa_controller")
    builder.Connect(
        plant.get_state_output_port(iiwa), iiwa_controller.get_input_port_estimated_state())
    
    
    builder.ExportInput(iiwa_controller.get_input_port_desired_acceleration(),"desired_accel")

    # Add in the feed-forward torque
    adder = builder.AddSystem(Adder(2, num_iiwa_positions))
    builder.Connect(iiwa_controller.get_output_port_control(),
                    adder.get_input_port(0))
    # Use a PassThrough to make the port optional (it will provide zero values if not connected).
    torque_passthrough = builder.AddSystem(PassThrough([0]*num_iiwa_positions))
    builder.Connect(torque_passthrough.get_output_port(),
                    adder.get_input_port(1))
    builder.ExportInput(torque_passthrough.get_input_port(), 
                        "iiwa_feedforward_torque")
    builder.Connect(adder.get_output_port(),
                    plant.get_actuation_input_port(iiwa))

    # Add discrete derivative to command velocities.
    #desired_state_from_position = builder.AddSystem(
    #    StateInterpolatorWithDiscreteDerivative(
    #        num_iiwa_positions, 1e-3, suppress_initial_transient=True))
    ##desired_state_from_position.set_name("desired_state_from_position")
    #builder.Connect(desired_state_from_position.get_output_port(),      
    #               iiwa_controller.get_input_port_desired_state())
    #builder.Connect(iiwa_position.get_output_port(), 
    #               desired_state_from_position.get_input_port())

    builder.ExportInput(iiwa_controller.get_input_port_desired_state(),"Desired State")
    

    # Export commanded torques.
    #builder.ExportOutput(adder.get_output_port(), "iiwa_torque_commanded")
    #builder.ExportOutput(adder.get_output_port(), "iiwa_torque_measured")

    # Wsg controller.
    # wsg_controller = builder.AddSystem(SchunkWsgPositionController())
    # wsg_controller.set_name("wsg_controller")
    # builder.Connect(
    #     wsg_controller.get_generalized_force_output_port(),             
    #     plant.get_actuation_input_port(wsg))
    # builder.Connect(plant.get_state_output_port(wsg),
    #                 wsg_controller.get_state_input_port())
    # builder.ExportInput(wsg_controller.get_desired_position_input_port(), 
    #                     "wsg_position")
    # builder.ExportInput(wsg_controller.get_force_limit_input_port(),  
    #                     "wsg_force_limit")
    # wsg_mbp_state_to_wsg_state = builder.AddSystem(
    #     MakeMultibodyStateToWsgStateSystem())
    # builder.Connect(plant.get_state_output_port(wsg), 
    #                 wsg_mbp_state_to_wsg_state.get_input_port())
    # builder.ExportOutput(wsg_mbp_state_to_wsg_state.get_output_port(), 
    #                      "wsg_state_measured")
    # builder.ExportOutput(wsg_controller.get_grip_force_output_port(), 
    #                      "wsg_force_measured")

    # Cameras.
    #AddRgbdSensors(builder, plant, scene_graph)

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "geometry_query")
    builder.ExportOutput(plant.get_contact_results_output_port(), 
                         "contact_results")
    builder.ExportOutput(plant.get_state_output_port(), 
                         "plant_continuous_state")

    diagram = builder.Build()
    return diagram, plant
