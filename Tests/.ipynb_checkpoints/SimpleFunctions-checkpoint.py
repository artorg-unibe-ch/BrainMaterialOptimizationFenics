import matplotlib.pyplot as plt           # Allows to do plots
import fenics as fe                       # Allors to use the FEniCS functions
import os                                 # Allows to use path
import pandas as pd                       # Allows to use data in tables
from IPython.display import display, clear_output
from scipy.interpolate import interp1d
from scipy.optimize import minimize, fmin
import numpy as np
import time



# ----------------------------------------------------------------------------
# Mesh Generation
# ----------------------------------------------------------------------------        

def MeshDefinition(Dimensions, NumberElements, Type='Lagrange', PolynomDegree=1):

    # Mesh
    Mesh = fe.BoxMesh(fe.Point(-Dimensions[0]/2, -Dimensions[1]/2, -Dimensions[2]/2), fe.Point(Dimensions[0]/2, Dimensions[1]/2, Dimensions[2]/2), NumberElements, NumberElements, NumberElements)

    # Functions spaces
    V_ele = fe.VectorElement(Type, Mesh.ufl_cell(), PolynomDegree)
    V = fe.VectorFunctionSpace(Mesh, Type, PolynomDegree)

    # Finite element functions
    du = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    u = fe.Function(V)
    
    return [Mesh, V, u, du, v]



# ----------------------------------------------------------------------------
# Kinematics Variables Computation
# ---------------------------------------------------------------------------- 

def Kinematics(u):
    # Kinematics
    d = u.geometric_dimension()
    I = fe.Identity(d)                    # Identity tensor
    F = I + fe.grad(u)                    # Deformation gradient
    F = fe.variable(F)                    # To differentiate Psi(F)
    J = fe.det(F)                         # Jacobian of F
    C = F.T*F                             # Right Cauchy-Green deformation tensor
    Ic = fe.tr(C)                         # Trace of C
    
    return [F, J, C, Ic]



# ----------------------------------------------------------------------------
# Load Case Definition
# ----------------------------------------------------------------------------

def LoadCaseDefinition(LoadCase, FinalRelativeStretch, RelativeStepSize, Dimensions, BCsType=False):
    
    Normal = fe.Constant((0, 0, 1))                                                                    # Normal to moving side
    
    if LoadCase == 'Compression':
        InitialState = 1
        
        if BCsType == 'Ideal':
            u_0 = fe.Constant((1E-3))                                                                  # Little displacement to avoid NaN values (Ogden)
            u_1 = fe.Expression(('(s-1)*h'), degree=1, s = InitialState, h = Dimensions[2] )           # Displacement imposed
        else:
            u_0 = fe.Constant((0, 0, 1E-3))                                                            # Little displacement to avoid NaN values (Ogden)
            u_1 = fe.Expression(('0', '0', '(s-1)*h'), degree=1, s = InitialState, h = Dimensions[2] ) # Displacement imposed
            
        Dir = fe.Constant((0,0,1))                                                                     # Deformation direction
        NumberSteps = FinalRelativeStretch / RelativeStepSize                                          # Number of steps
        DeltaStretch = -RelativeStepSize
                
    elif LoadCase == 'Tension':
        InitialState = 1
        
        if BCsType == 'Ideal':
            u_0 = fe.Constant((-1E-3))                                                                 # Little displacement to avoid NaN values (Ogden)
            u_1 = fe.Expression(('(s-1)*h'), degree=1, s = InitialState, h = Dimensions[2] )           # Displacement imposed
        else:
            u_0 = fe.Constant((0, 0, -1E-3))                                                           # Little displacement to avoid NaN values (Ogden)
            u_1 = fe.Expression(('0', '0', '(s-1)*h'), degree=1, s = InitialState, h = Dimensions[2] ) # Displacement imposed
            
        Dir = fe.Constant((0,0,1))                                                                     # Deformation direction
        NumberSteps = FinalRelativeStretch / RelativeStepSize                                          # Number of steps
        DeltaStretch = RelativeStepSize
        
        
    elif LoadCase == 'SimpleShear':
        InitialState = 0
        
        if BCsType == 'Ideal':
            u_0 = fe.Constant((-1E-3, 0, 0))                                                           # Little displacement to avoid NaN values
            u_1 = fe.Expression(('s*h', '0', '0'), degree=1, s = InitialState, h = Dimensions[2] )     # Displacement imposed
        else:
            u_0 = fe.Constant((-1E-3, 0, 0))                                                           # Little displacement to avoid NaN values
            u_1 = fe.Expression(('s*h', '0', '0'), degree=1, s = InitialState, h = Dimensions[2] )     # Displacement imposed
            
        Dir = fe.Constant((1,0,0))                                                                     # Deformation direction
        NumberSteps = FinalRelativeStretch / RelativeStepSize                                          # Number of steps
        DeltaStretch = RelativeStepSize
        
        
    else :
        print('Incorrect load case name')
        print('Load cases available are:')
        print('Compression')
        print('Tension')
        print('SimpleShear')
        
    return [u_0, u_1, InitialState, Dir, Normal, NumberSteps, DeltaStretch]



# ----------------------------------------------------------------------------
# Subdomains definition and Boundary conditions application
# ----------------------------------------------------------------------------

def BCsDefinition(Dimensions, Mesh, V, u_0, u_1, LoadCase, BCsType=False):

    # Define geometric spaces
    class LowerSide(fe.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and fe.near(x[2], -Dimensions[2]/2, tol)

    class UpperSide(fe.SubDomain):
        def inside(self, x, on_boundary):
            tol = 1E-14
            return on_boundary and fe.near(x[2], Dimensions[2]/2, tol)

    # Define integration over subdpmains
    Domains_Facets = fe.MeshFunction('size_t', Mesh, Mesh.geometric_dimension()-1)
    ds = fe.Measure('ds', domain=Mesh, subdomain_data=Domains_Facets)

    # Mark all domain facets with 0
    Domains_Facets.set_all(0)

    # Mark bottom facets with 1
    bottom = LowerSide()
    bottom.mark(Domains_Facets, 1)

    # Mark upper facets with 2
    upper = UpperSide()
    upper.mark(Domains_Facets, 2)

    # Apply boundary conditions
    if BCsType == 'Ideal':
        
        if LoadCase == 'SimpleShear':
            bcl = fe.DirichletBC(V, u_0, Domains_Facets, 1)
            bcu = fe.DirichletBC(V, u_1, Domains_Facets, 2)

        else:
            bcl = fe.DirichletBC(V.sub(2), u_0, Domains_Facets, 1)
            bcu = fe.DirichletBC(V.sub(2), u_1, Domains_Facets, 2)
    
    else:
        
        bcl = fe.DirichletBC(V, u_0, Domains_Facets, 1)
        bcu = fe.DirichletBC(V, u_1, Domains_Facets, 2)

    # Set of boundary conditions
    BoundaryConditions = [bcl, bcu]
    
    return [BoundaryConditions, ds]



# ----------------------------------------------------------------------------
# Constitutive Models Definition
# ----------------------------------------------------------------------------   

def CompressibleNeoHookean(Mu, Lambda, Ic, J):
        
    Psi = (Mu/2)*(Ic - 3) - Mu*fe.ln(J) + (Lambda/2)*(fe.ln(J))**2
    
    return Psi
    
def CompressibleOgden(Mu, Alpha, D, C, Ic, J):
    
    # Invariant of Right Cauchy-Green deformation tensor
    def I1(C):
        return fe.tr(C)

    def I2(C):
        c1 = C[0,0]*C[1,1] + C[0,0]*C[2,2] + C[1,1]*C[2,2]
        c2 = C[0,1]*C[0,1] + C[0,2]*C[0,2] + C[1,2]*C[1,2]
        return c1 - c2

    def I3(C):
        return fe.det(C)

    # Define function necessary for eigenvalues computation
    def v_inv(C):
        return (I1(C)/3.)**2 - I2(C)/3.

    def s_inv(C):
        return (I1(C)/3.)**3 - I1(C)*I2(C)/6. + I3(C)/2.

    def phi_inv(C):
        arg = s_inv(C)/v_inv(C)*fe.sqrt(1./v_inv(C))
        # numerical issues if arg~0
        # https://fenicsproject.org/qa/12299
        # /nan-values-when-computing-arccos-1-0-bug/
        arg_cond = fe.conditional( fe.ge(arg, 1-fe.DOLFIN_EPS),
        1-fe.DOLFIN_EPS,fe.conditional( fe.le(arg, -1+fe.DOLFIN_EPS),
        -1+fe.DOLFIN_EPS, arg ))
        return fe.acos(arg_cond)/3.

    # Eigenvalues of the strech tensor C
    lambda_1 = Ic/3. + 2*fe.sqrt(v_inv(C))*fe.cos(phi_inv(C))
    lambda_2 = Ic/3. - 2*fe.sqrt(v_inv(C))*fe.cos(fe.pi/3. + phi_inv(C))
    lambda_3 = Ic/3. - 2*fe.sqrt(v_inv(C))*fe.cos(fe.pi/3. - phi_inv(C))

    # Constitutive model
    Psi = 2 * Mu * (J**(-1/3)*lambda_1**(Alpha/2.) +  J**(-1/3)*lambda_2**(Alpha/2.) +  J**(-1/3)*lambda_3**(Alpha/2.) - 3) / Alpha**2 + 1/D * (J-1)**2
        
    return Psi
    
    
    
# ----------------------------------------------------------------------------
# Displacement Field Estimation
# ----------------------------------------------------------------------------   
    
def Estimate(Ic, J, u, v, du, BoundaryConditions, InitialState, u_1):
    
    # Use Neo-Hookean constitutive model
    Nu_H   = 0.49                         # (-)
    Mu_NH  = 1.15                         # (kPa)
    Lambda = 2*Mu_NH*Nu_H/(1-2*Nu_H)      # (kPa)
    Psi    = CompressibleNeoHookean(Mu_NH, Lambda, Ic, J)
    Pi = Psi * fe.dx
    
    # First directional derivative of the potential energy
    Fpi = fe.derivative(Pi,u,v)

    # Jacobian of Fpi
    Jac = fe.derivative(Fpi,u,du)
    
    # Define option for the compiler (optional)
    ffc_options = {"optimize": True, \
                   "eliminate_zeros": True, \
                   "precompute_basis_const": True, \
                   "precompute_ip_const": True }
    
    # Define the problem
    Problem = fe.NonlinearVariationalProblem(Fpi, u, BoundaryConditions, Jac, form_compiler_parameters=ffc_options)

    # Define the solver
    Solver = fe.NonlinearVariationalSolver(Problem)
    
    # Set solver parameters (optional)
    Prm = Solver.parameters
    Prm['nonlinear_solver'] = 'newton'
    Prm['newton_solver']['linear_solver'] = 'cg'             # Conjugate gradient
    Prm['newton_solver']['preconditioner'] = 'icc'           # Incomplete Choleski   
    Prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
    
    # Set initial displacement
    u_1.s = InitialState
    
    # Compute solution and save displacement
    Solver.solve()
    
    return u

    
    
# ----------------------------------------------------------------------------
# Problem Solving
# ----------------------------------------------------------------------------  
    
def SolveProblem(LoadCase, ConstitutiveModel, BCsType, FinalRelativeStretch, RelativeStepSize, Dimensions, NumberElements, Mesh, V, u, du, v, Ic, J, F, Psi, Plot = False, Paraview = False):
    
    if LoadCase == 'Compression':
                
        # Load case
        [u_0, u_1, InitialState, Direction, Normal, NumberSteps, DeltaStretch] = LoadCaseDefinition(LoadCase, FinalRelativeStretch, RelativeStepSize, Dimensions, BCsType)
        
        
    elif LoadCase == 'Tension':
                
        # Load case
        [u_0, u_1, InitialState, Direction, Normal, NumberSteps, DeltaStretch] = LoadCaseDefinition(LoadCase, FinalRelativeStretch, RelativeStepSize, Dimensions, BCsType)

        
    elif LoadCase == 'SimpleShear':
                
        # Load case
        [u_0, u_1, InitialState, Direction, Normal, NumberSteps, DeltaStretch] = LoadCaseDefinition(LoadCase, FinalRelativeStretch*2, RelativeStepSize*2, Dimensions, BCsType)

    # Boundary conditions
    [BoundaryConditions, ds] = BCsDefinition(Dimensions, Mesh, V, u_0, u_1, LoadCase, BCsType)
    
    # Estimation of the displacement field using Neo-Hookean model (necessary for Ogden)
    u = Estimate(Ic, J, u, v, du, BoundaryConditions, InitialState, u_1)
    
    # Reformulate the problem with the correct constitutive model
    Pi = Psi * fe.dx

    # First directional derivative of the potential energy
    Fpi = fe.derivative(Pi,u,v)

    # Jacobian of Fpi
    Jac = fe.derivative(Fpi,u,du)
    
    # Define option for the compiler (optional)
    ffc_options = {"optimize": True, \
                   "eliminate_zeros": True, \
                   "precompute_basis_const": True, \
                   "precompute_ip_const": True }

    # Define the problem
    Problem = fe.NonlinearVariationalProblem(Fpi, u, BoundaryConditions, Jac, form_compiler_parameters=ffc_options)

    # Define the solver
    Solver = fe.NonlinearVariationalSolver(Problem)

    # Set solver parameters (optional)
    Prm = Solver.parameters
    Prm['nonlinear_solver'] = 'newton'
    Prm['newton_solver']['linear_solver'] = 'cg'             # Conjugate gradient
    Prm['newton_solver']['preconditioner'] = 'icc'           # Incomplete Choleski
    Prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
    
    # Data frame to store values
    cols = ['Stretches','P']
    df = pd.DataFrame(columns=cols, index=range(int(NumberSteps)+1), dtype='float64')
    
    if Paraview == True:
        # Results File
        Output_Path = os.path.join('OptimizationResults', BCsType, ConstitutiveModel)
        ResultsFile = xdmffile = fe.XDMFFile(os.path.join(Output_Path, str(NumberElements) + 'Elements_' + LoadCase + '.xdmf'))
        ResultsFile.parameters["flush_output"] = True
        ResultsFile.parameters["functions_share_mesh"] = True
    
    if Plot == True:
        plt.rc('figure', figsize=[12,7])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    
    # Set the stretch state to initial state
    StretchState = InitialState
    
    # Loop to solve for each step
    for Step in range(int(NumberSteps+1)):

        # Update current state
        u_1.s = StretchState

        # Compute solution and save displacement
        Solver.solve()

        # First Piola Kirchoff (nominal) stress
        P = fe.diff(Psi, F)

        # Nominal stress vectors normal to upper surface
        p = fe.dot(P,Normal)

        # Reaction force on the upper surface
        f = fe.assemble(fe.inner(p,Direction)*ds(2))

        # Mean nominal stress on the upper surface
        Pm = f/fe.assemble(1*ds(2))

        # Save values to table
        df.loc[Step].Stretches = StretchState
        df.loc[Step].P = Pm

        # Plot
        if Plot == True:
            ax.cla()
            ax.plot(df.Stretches, df.P,  color = 'r', linestyle = '--', label = 'P', marker = 'o', markersize = 8, fillstyle='none')
            ax.set_xlabel('Stretch ratio (-)')
            ax.set_ylabel('Stresses (kPa)')
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.02))
            ax.legend(loc='upper left', frameon=True, framealpha=1)
            display(fig)
            clear_output(wait=True)
            
        if Paraview == True:
            # Project the displacement onto the vector function space
            u_project = fe.project(u, V, solver_type='cg')
            u_project.rename('Displacement (mm)', '')
            ResultsFile.write(u_project,Step)
            
            # Compute nominal stress vector
            p_project = fe.project(p, V)
            p_project.rename("Nominal stress vector (kPa)","")
            ResultsFile.write(p_project,Step)



        # Update the stretch state
        StretchState += DeltaStretch

    return df



# ----------------------------------------------------------------------------
# Interpolation Definition
# ----------------------------------------------------------------------------  

def Interpolation(LoadCase, DataFrame, FinalRelativeStretch, RelativeStepSize, Plot = False):
    
    # Experimental Data
    FolderPath = os.path.join('/home/msimon/Desktop/FEniCS/ExperimentalData/')
    FilePath = os.path.join(FolderPath, 'CR_' + LoadCase + '_ExpDat.csv')
    ExpData = pd.read_csv(FilePath, sep=';', header=None, decimal=',')

    # Interpolation
    InterpExpData = interp1d(ExpData[0], ExpData[1], kind='linear', fill_value='extrapolate')
    InterpSimPred = interp1d(DataFrame.Stretches, DataFrame.P, kind='linear', fill_value='extrapolate')

    NumberPoints=int(FinalRelativeStretch / RelativeStepSize +1)
    XInterp = np.linspace(DataFrame.iloc[0][0],DataFrame.iloc[-1][0],NumberPoints)

    # Control Plot
    if Plot == True :        
        plt.rc('figure', figsize=[12,7])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(XInterp, InterpExpData(XInterp),  color = 'g', linestyle = '--', label = 'Interpolated data', marker = 'o', markersize = 15, fillstyle='none')
        ax.plot(ExpData[0], ExpData[1],  color = 'b', linestyle = '--', label = 'Original Data', marker = 'o', markersize = 8, fillstyle='none')
        ax.plot(XInterp, InterpSimPred(XInterp),  color = 'k', linestyle = '--', label = 'Interpolated data', marker = 'o', markersize = 15, fillstyle='none')
        ax.plot(DataFrame.Stretches, DataFrame.P,  color = 'r', linestyle = '--', label = 'Simulation Prediction', marker = 'o', markersize = 8, fillstyle='none')
        ax.set_xlabel('Stretch ratio (-)')
        ax.set_ylabel('Stresses (kPa)')
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.02))
        ax.legend(loc='upper left', frameon=True, framealpha=1)
        plt.title(LoadCase)

    return [XInterp, InterpExpData, InterpSimPred]



# ----------------------------------------------------------------------------
# Cost Function Definition
# ----------------------------------------------------------------------------  

def CostFunction(Parameters, ConstitutiveModel, BCsType, NumberElements, LoadCases, RelativeWeights, FinalRelativeStretch, RelativeStepSize, Mesh, V, u, du, v, F, J, C, Ic, Dimensions = [5,5,5], Plot = False):

    # Mesh
#    [Mesh, V, u, du, v] = MeshDefinition(Dimensions, NumberElements)
#    [F, J, C, Ic] = Kinematics(u)
    
    Nu = Parameters[0]
    Mu = Parameters[1]

    if len(Parameters) == 3:
        Alpha = Parameters[2]
        D     = 3*(1-2*Nu)/(Mu*(1+Nu))     # (1/kPa)
        
    elif len(Parameters) == 2:
        Lambda = 2*Mu*Nu/(1-2*Nu)          # (kPa)
        
    Output_Path = os.path.join('OptimizationResults', BCsType, ConstitutiveModel)
    FileName = open(Output_Path + str(NumberElements) + 'Elements.txt', 'a+')
    
    if len(Parameters) == 3:
        FileName.write('%.3f %.3f %.3f' % (Nu, Mu, Alpha))
    elif len(Parameters) == 2:
        FileName.write('%.3f %.3f %.3f' % (Nu, Mu, Lambda))
        
    FileName.close()
        
    if 'Compression' in LoadCases:
        
        LoadCase = 'Compression'
        
        # Mesh
        [Mesh, V, u, du, v] = MeshDefinition(Dimensions, NumberElements)
        [F, J, C, Ic] = Kinematics(u)

        if len(Parameters) == 3:
            Psi = CompressibleOgden(Mu, Alpha, D, C, Ic, J)

        elif len(Parameters) == 2:
            Psi = CompressibleNeoHookean(Mu, Lambda, Ic, J)
        
        # Solve
        DataFrame = SolveProblem(LoadCase, ConstitutiveModel, BCsType, FinalRelativeStretch, RelativeStepSize, Dimensions, NumberElements, Mesh, V, u, du, v, Ic, J, F, Psi)

        # Interpolation
        [XInterp, InterpExpData, InterpSimPred] = Interpolation(LoadCase, DataFrame, FinalRelativeStretch, RelativeStepSize, Plot)
            
        # Compute partial compression cost
        CompressionDelta2 = []
        for X in XInterp:
            CompressionDelta2.append(((InterpExpData(X)-InterpSimPred(X))/ InterpExpData(XInterp[-1]))**2)
            
    if 'Tension' in LoadCases:
        
        LoadCase = 'Tension'
        
        # Mesh
        [Mesh, V, u, du, v] = MeshDefinition(Dimensions, NumberElements)
        [F, J, C, Ic] = Kinematics(u)

        if len(Parameters) == 3:
            Psi = CompressibleOgden(Mu, Alpha, D, C, Ic, J)

        elif len(Parameters) == 2:
            Psi = CompressibleNeoHookean(Mu, Lambda, Ic, J)
                
        # Solve
        DataFrame = SolveProblem(LoadCase, ConstitutiveModel, BCsType, FinalRelativeStretch, RelativeStepSize, Dimensions, NumberElements, Mesh, V, u, du, v, Ic, J, F, Psi)

        # Interpolation
        [XInterp, InterpExpData, InterpSimPred] = Interpolation(LoadCase, DataFrame, FinalRelativeStretch, RelativeStepSize, Plot)

        # Compute partial tension cost
        TensionDelta2 = []
        for X in XInterp:
            TensionDelta2.append(((InterpExpData(X)-InterpSimPred(X))/ InterpExpData(XInterp[-1]))**2)
            
            
    if 'SimpleShear' in LoadCases:
        
        LoadCase = 'SimpleShear'
                
        # Mesh
        [Mesh, V, u, du, v] = MeshDefinition(Dimensions, NumberElements)
        [F, J, C, Ic] = Kinematics(u)
        
        if len(Parameters) == 3:
            Psi = CompressibleOgden(Mu, Alpha, D, C, Ic, J)

        elif len(Parameters) == 2:
            Psi = CompressibleNeoHookean(Mu, Lambda, Ic, J)

        # Solve
        DataFrame = SolveProblem(LoadCase, ConstitutiveModel, BCsType, FinalRelativeStretch, RelativeStepSize, Dimensions, NumberElements, Mesh, V, u, du, v, Ic, J, F, Psi)

        # Interpolation
        [XInterp, InterpExpData, InterpSimPred] = Interpolation(LoadCase, DataFrame, FinalRelativeStretch, RelativeStepSize, Plot)

        # Compute partial simple shear cost
        SimpleShearDelta2 = []
        for X in XInterp:
            SimpleShearDelta2.append(((InterpExpData(X)-InterpSimPred(X))/ InterpExpData(XInterp[-1]))**2)
            
            
    # Compute total cost
    if 'Compression' in LoadCases and 'Tension' in LoadCases and 'SimpleShear' in LoadCases:
        TotalCost = np.sum(CompressionDelta2)  * RelativeWeights[0] + np.sum(TensionDelta2) * RelativeWeights[1] + np.sum(SimpleShearDelta2) * RelativeWeights[2]
        TotalCost = TotalCost / np.sum(RelativeWeights)
    elif 'Compression' in LoadCases and 'Tension' in LoadCases and 'SimpleShear' not in LoadCases:
        TotalCost = np.sum(CompressionDelta2)  * RelativeWeights[0] + np.sum(TensionDelta2) * RelativeWeights[1]
        TotalCost = TotalCost / np.sum(RelativeWeights)
    elif 'Compression' in LoadCases and 'Tension' not in LoadCases and 'SimpleShear' in LoadCases:
        TotalCost = np.sum(CompressionDelta2)  * RelativeWeights[0] + np.sum(SimpleShearDelta2) * RelativeWeights[1]
        TotalCost = TotalCost / np.sum(RelativeWeights)
    elif 'Compression' not in LoadCases and 'Tension' in LoadCases and 'SimpleShear' in LoadCases:
        TotalCost = np.sum(TensionDelta2)  * RelativeWeights[0] + np.sum(SimpleShearDelta2) * RelativeWeights[1]
        TotalCost = TotalCost / np.sum(RelativeWeights)
    elif 'Compression' in LoadCases and 'Tension' not in LoadCases and 'SimpleShear' not in LoadCases:
        TotalCost = np.sum(CompressionDelta2)  * RelativeWeights[0]
        TotalCost = TotalCost / np.sum(RelativeWeights)
    elif 'Compression' not in LoadCases and 'Tension' in LoadCases and 'SimpleShear' not in LoadCases:
        TotalCost = np.sum(TensionDelta2)  * RelativeWeights[0]
        TotalCost = TotalCost / np.sum(RelativeWeights)
    elif 'Compression' not in LoadCases and 'Tension' not in LoadCases and 'SimpleShear' in LoadCases:
        TotalCost = np.sum(SimpleShearDelta2) * RelativeWeights[0]
        TotalCost = TotalCost / np.sum(RelativeWeights)
            
    print('Cost:', np.sum(TotalCost))
    
    FileName = open(os.path.join(Output_Path, str(NumberElements) + 'Elements.txt'), 'a+')
    FileName.write(' %.3f\n' % (TotalCost))    
    FileName.close()    

    return np.sum(TotalCost)



# ----------------------------------------------------------------------------
# Optimization Function
# ----------------------------------------------------------------------------  

def ParametersOptimization(ConstitutiveModel, NumberElements, BCsType, LoadCases, RelativeWeights, FinalRelativeStretch, RelativeStepSize, Dimensions=[5,5,5], Plot = False):
    
    # Initialize time tracking
    start = time.time()
    
    # Folder for the results
    Output_Path = os.path.join('OptimizationResults', BCsType, ConstitutiveModel)
    os.makedirs(Output_Path, exist_ok=True)
    
    FileName = open(os.path.join(Output_Path, str(NumberElements) + 'Elements.txt'), 'a+')
    
    if ConstitutiveModel == 'Ogden':
        # Ogden initial guessed parameters
        Nu    = 0.49                          # (-)
        Mu    = 0.66                          # (kPa)
        Alpha = -24.3                         # (-)

        InitialGuess = np.array([Nu, Mu, Alpha])
        Bounds = [(0.4, 0.495), (1E-3, 2), (-100, -1E-3)]
        
        FileName.write('Nu Mu Alpha TotalCost\n')
    
    elif ConstitutiveModel == 'Neo-Hookean':
        # Neo-Hookean initial guessed parameters
        Nu    = 0.49                          # (-)
        Mu    = 1.15                          # (kPa)
        
        InitialGuess = np.array([Nu, Mu])
        Bounds = [(0.4, 0.495), (0.5, 2)]
        
        FileName.write('Nu Mu Lambda TotalCost\n')

    FileName.close()    
    
    ResultsOptimization = minimize(CostFunction, InitialGuess, args = (ConstitutiveModel, BCsType, NumberElements, LoadCases, RelativeWeights, FinalRelativeStretch, RelativeStepSize, Dimensions), method = 'L-BFGS-B', bounds = Bounds)
    
    if ConstitutiveModel == 'Ogden':
        [Nu, Mu, Alpha] = ResultsOptimization.x
        print('Final Nu = %1.3f' % (Nu))
        print('Final Mu = %1.3f' % (Mu))
        print('Final Alpha = %1.3f' % (Alpha))
        
    elif ConstitutiveModel == 'Neo-Hookean':
        [Nu, Mu] = ResultsOptimization.x
        print('Final Nu = %1.3f' % (Nu))
        print('Final Mu = %1.3f' % (Mu))
    
    TimeName = open(os.path.join(Output_Path, 'OptimizationTime.txt'), 'a+')
    TimeName.write('%1.3f\n' % (time.time() - start))
    TimeName.close()
        
    if Plot == True :
        
        df = pd.read_csv(os.path.join(Output_Path, str(NumberElements) + 'Elements.txt'), sep=' ', decimal='.')
            
        plt.rc('figure', figsize=[36,7])
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        ax.cla()
        ax.plot(df.Mu,  color = 'r', linestyle = '--', marker = 'o', markersize = 8, fillstyle='none')
        ax.set_xlabel('Iteration number(-)')
        ax.set_ylabel('Mu (kPa)')
        
        ax = fig.add_subplot(1, 3, 2)
        ax.cla()
        ax.plot(df.Alpha,  color = 'b', linestyle = '--', marker = 'o', markersize = 8, fillstyle='none')
        ax.set_xlabel('Iteration number(-)')
        ax.set_ylabel('Alpha (-)')
        
        ax = fig.add_subplot(1, 3, 3)
        ax.cla()
        ax.plot(df.TotalCost,  color = 'g', linestyle = '--', marker = 'o', markersize = 8, fillstyle='none')
        ax.set_xlabel('Iteration number(-)')
        ax.set_ylabel('Cost (-)')
        
    return ResultsOptimization