# Start writing code here..
import numpy as np
from IPython.display import display, HTML
import matplotlib.pyplot as plt

from pydrake.all import (
    DiagramBuilder, ConnectPlanarSceneGraphVisualizer, Simulator, RigidTransform, LeafSystem, BasicVector,
    JacobianWrtVariable, RollPitchYaw, VectorLogSink, SpatialInertia, UnitInertia, Box, CoulombFriction, ContactModel,Sphere,
    WitnessFunctionDirection, UnrestrictedUpdateEvent, MathematicalProgram, le,SnoptSolver
)

from pydrake.examples.manipulation_station import ManipulationStation




class TorqueController(LeafSystem):
  """Wrapper System for Commanding Pure Torques to planar iiwa.
    @param plant MultibodyPlant of the simulated plant.
    @param ctrl_fun function object to implement torque control law.
    @param vx Velocity towards the linear direction. 
  """
  def __init__(self, plant, ctrl_fun, vx):
    LeafSystem.__init__(self)
    self._plant = plant 
    self._plant_context = plant.CreateDefaultContext() 
    self._iiwa = plant.GetModelInstanceByName("iiwa")
    self._G = plant.GetBodyByName("body").body_frame()
    self._W = plant.world_frame()
    self._ctrl_fun = ctrl_fun 
    self._vx = vx

    self.DeclareVectorInputPort("iiwa_position_measured", BasicVector(3))
    self.DeclareVectorInputPort("iiwa_velocity_measured", BasicVector(3))

    # If we want, we can add this in to do closed-loop force control on z.
    #self.DeclareVectorInputPort("iiwa_torque_external", BasicVector(3))

    self.DeclareVectorOutputPort("iiwa_position_command", BasicVector(3),
                                 self.CalcPositionOutput)
    self.DeclareVectorOutputPort("iiwa_torque_cmd", BasicVector(3),
                                 self.CalcTorqueOutput)
    # Compute foward kinematics so we can log the wsg position for grading. 
    self.DeclareVectorOutputPort("wsg_position", BasicVector(3),
                                 self.CalcWsgPositionOutput)
    
  def CalcPositionOutput(self, context, output):
    """Set q_d = q_now. This ensures the iiwa goes into pure torque mode in sim by setting the 
    position control torques in InverseDynamicsController to zero. 
    NOTE(terry-suh): Do not use this method on hardware or deploy this notebook on hardware. 
    We can only simulate pure torque control mode for iiwa on sim. 
    """
    q_now = self.get_input_port(0).Eval(context)
    output.SetFromVector(q_now)

  def CalcTorqueOutput(self, context, output):    
    # Hard-coded position and force profiles. Can be connected from Trajectory class. 
    if (context.get_time() < 2.0):
      px_des = 0.65
    else:
      px_des = 0.65 + self._vx * (context.get_time() - 2.0)

    fz_des = 10 

    # Read inputs 
    q_now = self.get_input_port(0).Eval(context)
    v_now = self.get_input_port(1).Eval(context)
    #tau_now = self.get_input_port(2).Eval(context) 

    self._plant.SetPositions(self._plant_context, self._iiwa, q_now)

    # 1. Convert joint space quantities to Cartesian quantities.
    X_now = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._G)

    rpy_now = RollPitchYaw(X_now.rotation()).vector()
    p_xyz_now = X_now.translation()

    J_G = self._plant.CalcJacobianSpatialVelocity(
        self._plant_context, JacobianWrtVariable.kQDot, 
        self._G, [0,0,0], self._W, self._W)
    
    # Only select relevant terms. We end up with J_G of shape (3,3). 
    # Rows correspond to (pitch, x, z).
    # Columns correspond to (q0, q1, q2). 
    J_G = J_G[(1,3,5),7:10]
    v_pxz_now = J_G.dot(v_now)

    p_pxz_now = np.array([rpy_now[1], p_xyz_now[0], p_xyz_now[2]])

    # 2. Apply ctrl_fun 
    F_pxz = self._ctrl_fun(p_pxz_now, v_pxz_now, px_des, fz_des)

    # 3. Convert back to joint coordinates
    tau_cmd = J_G.T.dot(F_pxz)
    output.SetFromVector(tau_cmd)

  def CalcWsgPositionOutput(self, context, output):
    """
    Compute Forward kinematics. Needed to log the position trajectory for grading.
    """
    q_now = self.get_input_port(0).Eval(context)
    self._plant.SetPositions(self._plant_context, self._iiwa, q_now)
    X_now = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._G)

    rpy_now = RollPitchYaw(X_now.rotation()).vector()
    p_xyz_now = X_now.translation()
    p_pxz_now = np.array([rpy_now[1], p_xyz_now[0], p_xyz_now[2]])

    output.SetFromVector(p_pxz_now)

def AddBook(plant):
  mu = 10.0
  book = plant.AddModelInstance("book")
  book_body = plant.AddRigidBody("book_body", book, 
                                 SpatialInertia(
                                     mass = 0.2, 
                                     p_PScm_E=np.array([0., 0., 0.]),
                                     G_SP_E = UnitInertia(1.0, 1.0, 1.0)))
  
  shape = Sphere(.03)
  print(book)
  if plant.geometry_source_is_registered():
    #plant.RegisterCollisionGeometry(book_body, RigidTransform(), shape, "book_body", CoulombFriction(mu, mu))
    plant.RegisterVisualGeometry(book_body, RigidTransform(), shape, "book_body", [.9, .2, .2, 1.0])

  return book 

def BuildAndSimulate(ctrl_fun, velocity, duration):
  builder = DiagramBuilder()

  # Add ManipulationStation 
  station = builder.AddSystem(ManipulationStation(time_step = 1e-3))
  station.SetupPlanarIiwaStation()
  book = AddBook(station.get_mutable_multibody_plant())
  #bounce  = builder.AddSystem(BouncingBall())
  
  station.Finalize()


  controller = builder.AddSystem(
      TorqueController(station.get_multibody_plant(), ctrl_fun, velocity))

  #pos_to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(book, input_multibody_state=False))

  logger = builder.AddSystem(VectorLogSink(3))

  builder.Connect(controller.get_output_port(0),
                  station.GetInputPort("iiwa_position"))
  builder.Connect(controller.get_output_port(1),
                  station.GetInputPort("iiwa_feedforward_torque"))
  builder.Connect(controller.get_output_port(2),
                  logger.get_input_port(0))
  
  builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                  controller.get_input_port(0))
  builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),
                  controller.get_input_port(1))

  #builder.Connect()


  #builder.Connect(book.Get)

  
  diagram = builder.Build()

  # Initialize default positions for plant. 
  plant = station.get_mutable_multibody_plant()
  simulator = Simulator(diagram)
  plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
  plant.SetFreeBodyPose(plant_context, 
                        plant.GetBodyByName("book_body"),
                        RigidTransform([0.8, 0.0, 0.5])) # limit is between 0.5 and 0.55
  plant.SetPositions(plant_context, 
                     plant.GetModelInstanceByName("iiwa"),
                     np.array([np.pi/2, 0, np.pi/3]))
  
  station_context = station.GetMyContextFromRoot(simulator.get_mutable_context())
  station.GetInputPort("wsg_position").FixValue(station_context, [0.02])
  vis = ConnectPlanarSceneGraphVisualizer(
        builder,
        station.get_scene_graph(),
        output_port=station.GetOutputPort("query_object"),
        xlim=[-0.5, 1.2],
        ylim=[-0.8, 2],
        show=False)
  vis.start_recording()

  simulator.AdvanceTo(duration)
  vis.stop_recording()
  
  ani = vis.get_recording_as_animation(repeat=False)
  display(HTML(ani.to_jshtml()))
    
  pose = plant.GetFreeBodyPose(plant_context,
                               plant.GetBodyByName("book_body"))
  
  # Return these so that we can check the pose of each object. 
  return logger.FindLog(simulator.get_context()), plant, plant_context

def compute_ctrl(p_pxz_now, v_pxz_now, x_des, f_des):
  """Compute control action given current position and velocities, as well as 
  desired x-direction position p_des(t) / desired z-direction force f_des. 
  You may set theta_des yourself, though we recommend regulating it to zero. 
  Input:
    - p_pxz_now: np.array (dim 3), position of the finger. [thetay, px, pz] 
    - v_pxz_now: np.array (dim 3), velocity of the finger. [wy, vx, vz] 
    - x_des: float, desired position of the finger along the x-direction. 
    - f_des: float, desired force on the book along the z-direction. 
  Output:
    - u    : np.array (dim 3), spatial torques to send to the manipulator. [tau_y, fx, fz] 
  """
  u=np.array([0,0,0])
  return u
velocity = -.124 # p_des = 0.65 + velocity * max\{time - 2.0, 0\}
duration = 6.2  # duration to simulate. We check the book pose at the end of duration. set to 5~10.
log, plant, plant_context = BuildAndSimulate(compute_ctrl, velocity, duration)