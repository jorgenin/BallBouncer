import numpy as np



from pydrake.all import (
    MathematicalProgram, le, Solve,eq,ge,sin,cos
)

#DYNAMICS FUNCTIONS


#returns the position and the velocity of teh ball at the bounce location.
def BounceManip2d(q_m,q_mdot,q_b,q_bdot,h,e,g):
    theta = q_m[2]
    normal =np.array([sin(theta),-cos(theta)])
    tangent = np.array([cos(theta),sin(theta)])
    Vel_ball = q_bdot + np.array([0,-h*g])
    
    Vel_relative = Vel_ball - np.array(q_mdot[0:2])
    Veln = -normal.dot(Vel_relative)*e
    Velt = tangent.dot(Vel_ball)

    Pos_Ball = q_b+ np.array([h*q_bdot[0],h*q_bdot[1]-1/2*g*h*h])

    return Pos_Ball,Veln*normal+Velt*tangent

#returns the velocity and the position of the ball at the bounce location
def BounceFloor2d(q_m,q_mdot,h,e,g):
    Vel_ball = q_mdot[3:5] + [0,-h*g]
    
    

    Pos_Ball = np.array(q_m[3:5]) + np.array([h*q_mdot[3],h*q_mdot[4]-1/2*g*h*h])

    return Pos_Ball, [Vel_ball[0],-Vel_ball[1]*e]
    


#OPTIMIZER

def Optimizer2d(N,m=1,r=.1,e=.8):

    g = 9.81 #m/s^2

    N = 10 #number of bounces

    #q_m => x_m, z_m, theta_m, x_b, z_b, 
    # q_mdot => x_mdot, z_mdot, theta_mdot, x_bdot,z_bdot]
    #This if for bounces on the floor
    #q_b => x_b z_b

    prog = MathematicalProgram()
    h = prog.NewContinuousVariables(2,N,'h')


    prog.AddBoundingBoxConstraint(0.001,10,h)

    q_m = prog.NewContinuousVariables(5,N,'q_m')

    q_mdot = prog.NewContinuousVariables(5,N,'q_mdot')

    q_b = prog.NewContinuousVariables(2,N,'q_b')
    q_bdot = prog.NewContinuousVariables(2,N,'q_bdot')

    x_pos =np.ones((10,1))*.7

    #This is the constraints for the Z
    prog.AddConstraint(ge(q_m[1,:],.3)).evaluator().set_description('Min Manip Z ')
    prog.AddConstraint(le(q_m[1,:],.6)).evaluator().set_description('Max Manip Z ')
    prog.AddConstraint(ge(q_m[0,:],.5)).evaluator().set_description('Min Manip X ')
    prog.AddConstraint(le(q_m[0,:],.9)).evaluator().set_description('Max Manip X ')

    #Here are constraints for the Rotation
    prog.AddConstraint(le(q_m[2,:],np.pi/4)).evaluator().set_description('Max Manip theta ')
    prog.AddConstraint(ge(q_m[2,:],-np.pi/4)).evaluator().set_description('Min Manip theta ')

    initialqb = [.5,0]
    initialq_bdot = np.array([.2,5])

    #initialen = initialq_bdot.dot(initialq_bdot)*m/2

    
    for i in range(N-1):
        NewPos, newVel = BounceManip2d(q_m[:,i],q_mdot[:,i],q_b[:,i],q_bdot[:,i],h[0,i],e,g)
        theta = q_m[2,i]
        Normal = np.array([sin(theta),-cos(theta)])
        tangent =  np.array([cos(theta),sin(theta)])
        
        #Add a contraint to make sure the ball bounces from the the hand to the floor
        prog.AddConstraint(eq(q_mdot[3:5,i], newVel)).evaluator().set_description(' Manip Vel ' + str(i))
        prog.AddConstraint(eq(q_m[3:5,i], NewPos)).evaluator().set_description(' Manip Pos '+ str(i) )
        normdist = np.array((q_m[3:5,i]-q_m[0:2,i])).dot(Normal)
        
        prog.AddConstraint(normdist == r*r).evaluator().set_description('Manip Bounce '+ str(i) )
        

        tangentdist = np.array((q_m[3:5,i]-q_m[0:2,i])).dot(tangent)
        
        prog.AddConstraint(tangentdist >= -.05).evaluator().set_description('Min Manip Distance '+ str(i))
        prog.AddConstraint(tangentdist <= .05).evaluator().set_description('Max Manip Distance ' + str(i) )

        prog.AddLinearConstraint(q_mdot[0,i] >= -3)
        prog.AddLinearConstraint(q_mdot[1,i] <= 0)
        prog.AddLinearConstraint(q_mdot[1,i] >= -2)
        prog.AddLinearConstraint(q_mdot[0,i] <= 2)
        Vel_ball = q_bdot[1,i] - h[0,i]*g


        #The ball needs to be still traveling upwards when hit
        prog.AddConstraint(Vel_ball >= 0.01).evaluator().set_description('Pos Vel '+ str(i))


        ball_floorPos, ball_floorVel = BounceFloor2d(q_m[:,i],q_mdot[:,i],h[1,i],e,g)
        #Add a contraint to make the ball bounce on to the manipulator
        prog.AddConstraint(eq(q_b[:,i+1]- ball_floorPos ,0)).evaluator().set_description('Floor Vel ' + str(i))
        prog.AddConstraint(eq(q_bdot[:,i+1] , ball_floorVel)).evaluator().set_description('FloorPos ' + str(i) )
        prog.AddConstraint(q_b[1,i+1] == r).evaluator().set_description('Floor Bounce ' + str(i))

        prog.AddCost(100*(q_b[0,i+1]-x_pos[i])**2)
        #prog.AddCost(.1*q_bdot[0,i+1]**2)

        #energy= q_mdot[3:5,i].dot(q_mdot[3:5,i])*m/2 + q_m[4,i]*m*g
        #timestepChange = q_m[0:2,i]-q_m[0:2,i+1]
        #distancesqrd = timestepChange.dot(timestepChange)
        #value = 10*distancesqrd

        #prog.AddCost(10*(energy-initialen)**2)
        prog.AddCost(10*q_mdot[0:2,i].dot(q_mdot[0:2,i]))
        #cost to how far you are from the previous point
        #prog.AddCost(value)
        #cost to how much how far you are from the ball
        #prog.AddCost(10*q_mdot[0:2,i].dot(q_mdot[3:5,i]))

        prog.AddCost(30*q_bdot[0,i+1]**2)
        prog.AddCost(100*(q_m[0,i]-q_b[0,i+1])**2)
        #prog.AddCost(-100*(h[0,i]+h[1,i]))

    
    



    prog.AddConstraint(eq(q_b[:,0],initialqb))
    prog.AddConstraint(eq(q_bdot[:,0],initialq_bdot))



    #Inital Guess
    for i in range(N-1):
        q_m_guess = [0,0,0,0,0]
        q_mdot_guess =[0,0,0,0,0]
        h_guess=.1
        NewPos, NewVel=BounceManip2d(q_m_guess,q_mdot_guess,initialqb,initialq_bdot,h_guess,e,g)
        q_m_guess = [NewPos[0],NewPos[1],0,NewPos[0], NewPos[1]]
        q_mdot_guess = [0,0,0,NewVel[0],NewVel[1]]

        prog.SetInitialGuess(q_m[:,i],q_m_guess)
        prog.SetInitialGuess(q_mdot[:,i],q_mdot_guess)

        prog.SetInitialGuess(h[0,i],h_guess)
        prog.SetInitialGuess(h[1,i],h_guess)

        

        ball_floorPos, ball_floorVel = BounceFloor2d(q_m_guess,q_mdot_guess,h_guess,e,g)
    
        prog.SetInitialGuess(q_b[:,i+1],ball_floorPos)
        prog.SetInitialGuess(q_bdot[:,i+1],ball_floorVel)


    result = Solve(prog)



    if not result.is_success():
        infesible = result.GetInfeasibleConstraints(prog)
        print("Infeasible Constraints:")
        for i in range(len(infesible)):
            print(infesible[i])

    h_res = result.GetSolution(h)
    q_m_res = result.GetSolution(q_m)
    q_mdot_res = result.GetSolution(q_mdot)
    q_b_res = result.GetSolution(q_b)
    q_bdot_res = result.GetSolution(q_bdot)

    return h_res,q_m_res,q_mdot_res,q_b_res,q_bdot_res

