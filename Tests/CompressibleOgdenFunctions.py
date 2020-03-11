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
# Functions definition
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
        
    
    
def LoadCase(LoadCase, FinalRelativeStretch, RelativeStepSize, Dimensions):
    
    Normal = fe.Constant((0, 0, 1))                                                 # Normal to moving side
    
    if LoadCase == 'Compression':
        InitialState = 1
        u_0 = fe.Constant((0, 0, 1E-3))                                             # Little displacement to avoid NaN values
        u_1 = fe.Expression(('0', '0', '(s-1)*h'), degree=1, s = InitialState, h = Dimensions[2] )        # Displacement imposed
        Dir = fe.Constant((0,0,1))                                                  # Deformation direction
        NumberSteps = FinalRelativeStretch / RelativeStepSize                       # Number of steps
        DeltaStretch = -RelativeStepSize
                
    elif LoadCase == 'Tension':
        InitialState = 1
        u_0 = fe.Constant((-1E-3, 0, 0))                                            # Little displacement to avoid NaN values
        u_1 = fe.Expression(('0', '0', '(s-1)*h'), degree=1, s = InitialState, h = Dimensions[2] )        # Displacement imposed
        Dir = fe.Constant((0,0,1))                                                  # Deformation direction
        NumberSteps = FinalRelativeStretch / RelativeStepSize                       # Number of steps
        DeltaStretch = RelativeStepSize
        
        
    elif LoadCase == 'Simple Shear':
        InitialState = 0
        u_0 = fe.Constant((-1E-3, 0, 0))                                            # Little displacement to avoid NaN values
        u_1 = fe.Expression(('s*h', '0', '0'), degree=1, s = InitialState, h = Dimensions[2] )        # Displacement imposed
        Dir = fe.Constant((1,0,0))                                                  # Deformation direction
        NumberSteps = FinalRelativeStretch / RelativeStepSize                       # Number of steps
        DeltaStretch = RelativeStepSize
        
        
    else :
        print('Incorrect load case name')
        print('Load cases available are:')
        print('Compression')
        print('Tension')
        print('Simple Shear')
        
    return [u_0, u_1, InitialState, Dir, Normal, NumberSteps, DeltaStretch]



def SolveProblem(Psi, Nu, F, Ic, J, Mesh, V, u, v, du, u_0, u_1, Dimensions, InitialState, Normal, Dir, NumberSteps, DeltaStretch, Plot = False):
    
    # ----------------------------------------------------------------------------
    # Subdomains definition and Boundary conditions application
    # ----------------------------------------------------------------------------

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
    bcl = fe.DirichletBC(V, u_0, Domains_Facets, 1)
    bcu = fe.DirichletBC(V, u_1, Domains_Facets, 2)

    # Set of boundary conditions
    bcs = [bcl, bcu]
    
    # ----------------------------------------------------------------------------
    # Estimation of the results (only necessary for Ogden)
    # ----------------------------------------------------------------------------
    
    # Change constitutive model for Neo-Hookean
    Mu_NH  = 1.15                         # (kPa)
    Lambda = 2*Mu_NH*Nu/(1-2*Nu)          # (kPa)
    Psi_NH = CompressibleNeoHookean(Mu_NH, Lambda, Ic, J)
    Pi = Psi_NH * fe.dx
    
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
    Problem = fe.NonlinearVariationalProblem(Fpi, u, bcs, Jac, form_compiler_parameters=ffc_options)

    # Define the solver
    Solver = fe.NonlinearVariationalSolver(Problem)
    
    # Update boundary condition for non zero displacement
    u_1.s = InitialState

    # Compute solution and save displacement
    Solver.solve()
    
    # Reformulate the problem with the correct constitutive model
    Pi = Psi * fe.dx

    # First directional derivative of the potential energy
    Fpi = fe.derivative(Pi,u,v)

    # Jacobian of Fpi
    Jac = fe.derivative(Fpi,u,du)

    # Define the problem
    Problem = fe.NonlinearVariationalProblem(Fpi, u, bcs, Jac, form_compiler_parameters=ffc_options)

    # Define the solver
    Solver = fe.NonlinearVariationalSolver(Problem)

    # Set solver parameters (optional)
    Prm = Solver.parameters
    Prm['nonlinear_solver'] = 'newton'
    Prm['newton_solver']['linear_solver'] = 'cg'             # Conjugate gradient
    Prm['newton_solver']['preconditioner'] = 'icc'           # Incomplete Choleski    
    
    # Data frame to store values
    cols = ['Stretches','P']
    df = pd.DataFrame(columns=cols, index=range(int(NumberSteps)+1), dtype='float64')
    
    if Plot == True:
        plt.rc('figure', figsize=[12,7])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    
    # Set the stretch state to initial state
    StretchState = InitialState
    
    # ----------------------------------------------------------------------------
    # Problem Solving
    # ----------------------------------------------------------------------------

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
        f = fe.assemble(fe.inner(p,Dir)*ds(2))

        # Mean nominal stress on the upper surface
        Pm = f/fe.assemble(1*ds(2))

        # Project the displacement onto the vector function space
        u_project = fe.project(u, V, solver_type='cg')
        u_project.rename('displacement (mm)', '')
#         results.write(u_project,Step)

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

        # Update the stretch state
        StretchState += DeltaStretch

    return df

def Mesh2Invariants(Dimensions, NumberElements=1, Type='Lagrange', PolynomDegree=1):

    # Mesh
    Mesh = fe.BoxMesh(fe.Point(-Dimensions[0]/2, -Dimensions[1]/2, -Dimensions[2]/2), fe.Point(Dimensions[0]/2, Dimensions[1]/2, Dimensions[2]/2), NumberElements, NumberElements, NumberElements)

    # Functions spaces
    V_ele = fe.VectorElement(Type, Mesh.ufl_cell(), PolynomDegree)
    V = fe.VectorFunctionSpace(Mesh, Type, PolynomDegree)

    # Finite element functions
    du = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    u = fe.Function(V)
    
    # Kinematics
    d = u.geometric_dimension()
    I = fe.Identity(d)                    # Identity tensor
    F = I + fe.grad(u)                    # Deformation gradient
    F = fe.variable(F)                    # To differentiate Psi(F)
    J = fe.det(F)                         # Jacobian of F
    C = F.T*F                             # Right Cauchy-Green deformation tensor
    Ic = fe.tr(C)                         # Trace of C
    
    return [Mesh, V, u, du, v, F, J, C, Ic]


def OgdenCostFunction(Parameters, Nu, LoadCases, RelativeWeights, NumberElements, FinalRelativeStretch, RelativeStepSize, Dimensions, Plot = False):
    
    # Mesh
    [Mesh, V, u, du, v, F, J, C, Ic] = Mesh2Invariants(Dimensions, NumberElements)
    
    Mu = Parameters[0]

    if len(Parameters) == 2:
        Alpha = Parameters[1]
        D     = 3*(1-2*Nu)/(Mu*(1+Nu))     # (1/kPa)

        Psi = CompressibleOgden(Mu, Alpha, D, C, Ic, J)
    elif len(Parameters) == 1:
        Lambda = 2*Mu*Nu/(1-2*Nu)          # (kPa)
        
        Psi = CompressibleNeoHookean(Mu, Lambda, Ic, J)
        
    if 'Compression' in LoadCases:
        # Load case
        [u_0, u_1, InitialState, Dir, Normal, NumberSteps, DeltaStretch] = LoadCase('Compression', FinalRelativeStretch, RelativeStepSize, Dimensions)        
        
        # Solve
        df = SolveProblem(Psi, Nu, F, Ic, J, Mesh, V, u, v, du, u_0, u_1, Dimensions, InitialState, Normal, Dir, NumberSteps, DeltaStretch)

        # Interpolation
        FolderPath = os.path.join('/home/msimon/Desktop/SHARED/ScriptsAndData/ExperimentalData/')
        FilePath = os.path.join(FolderPath, 'CR_Compression_ExpDat.csv')
        ExpData = pd.read_csv(FilePath, sep=';', header=None, decimal=',')

        InterpExpData = interp1d(ExpData[0], ExpData[1], kind='linear', fill_value='extrapolate')
        InterpSimPred = interp1d(df.Stretches, df.P, kind='linear', fill_value='extrapolate')

        NumberPoints=int(FinalRelativeStretch / RelativeStepSize +1)
        XInterp = np.linspace(df.iloc[0][0],df.iloc[-1][0],NumberPoints)

        # Control Plot
        if Plot == True :        
            plt.rc('figure', figsize=[12,7])
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.cla()
            ax.plot(ExpData[0], ExpData[1],  color = 'b', linestyle = '--', label = 'Original Data', marker = 'o', markersize = 8, fillstyle='none')
            ax.plot(XInterp, InterpExpData(XInterp),  color = 'g', linestyle = '--', label = 'Interpolated data', marker = 'o', markersize = 15, fillstyle='none')
            ax.plot(df.Stretches, df.P,  color = 'r', linestyle = '--', label = 'Simulation Prediction', marker = 'o', markersize = 8, fillstyle='none')
            ax.plot(XInterp, InterpSimPred(XInterp),  color = 'k', linestyle = '--', label = 'Interpolated data', marker = 'o', markersize = 15, fillstyle='none')
            ax.set_xlabel('Stretch ratio (-)')
            ax.set_ylabel('Stresses (kPa)')
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.02))
            ax.legend(loc='upper left', frameon=True, framealpha=1)
            plt.title('Compression')
            
        CompressionDelta2 = []
        for X in XInterp:
            CompressionDelta2.append(((InterpExpData(X)-InterpSimPred(X))/ InterpExpData(XInterp[-1]))**2)
            
    if 'Tension' in LoadCases:
        # Load case
        [u_0, u_1, InitialState, Dir, Normal, NumberSteps, DeltaStretch] = LoadCase('Tension', FinalRelativeStretch, RelativeStepSize, Dimensions)

        # Solve
        df = SolveProblem(Psi, Nu, F, Ic, J, Mesh, V, u, v, du, u_0, u_1, Dimensions, InitialState, Normal, Dir, NumberSteps, DeltaStretch)

        # Interpolation
        FolderPath = os.path.join('/home/msimon/Desktop/SHARED/ScriptsAndData/ExperimentalData/')
        FilePath = os.path.join(FolderPath, 'CR_Tension_ExpDat.csv')
        ExpData = pd.read_csv(FilePath, sep=';', header=None, decimal=',')

        InterpExpData = interp1d(ExpData[0], ExpData[1], kind='linear', fill_value='extrapolate')
        InterpSimPred = interp1d(df.Stretches, df.P, kind='linear', fill_value='extrapolate')

        NumberPoints=int(FinalRelativeStretch / RelativeStepSize +1)
        XInterp = np.linspace(df.iloc[0][0],df.iloc[-1][0],NumberPoints)

        # Control Plot
        if Plot == True :        
            plt.rc('figure', figsize=[12,7])
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.cla()
            ax.plot(ExpData[0], ExpData[1],  color = 'b', linestyle = '--', label = 'Original Data', marker = 'o', markersize = 8, fillstyle='none')
            ax.plot(XInterp, InterpExpData(XInterp),  color = 'g', linestyle = '--', label = 'Interpolated data', marker = 'o', markersize = 15, fillstyle='none')
            ax.plot(df.Stretches, df.P,  color = 'r', linestyle = '--', label = 'Simulation Prediction', marker = 'o', markersize = 8, fillstyle='none')
            ax.plot(XInterp, InterpSimPred(XInterp),  color = 'k', linestyle = '--', label = 'Interpolated data', marker = 'o', markersize = 15, fillstyle='none')
            ax.set_xlabel('Stretch ratio (-)')
            ax.set_ylabel('Stresses (kPa)')
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.02))
            ax.legend(loc='upper left', frameon=True, framealpha=1)
            plt.title('Tension')

        TensionDelta2 = []
        for X in XInterp:
            TensionDelta2.append(((InterpExpData(X)-InterpSimPred(X))/ InterpExpData(XInterp[-1]))**2)
            
            
    if 'Simple Shear' in LoadCases:
        # Load case
        [u_0, u_1, InitialState, Dir, Normal, NumberSteps, DeltaStretch] = LoadCase('Simple Shear', FinalRelativeStretch*2, RelativeStepSize*2, Dimensions)

        # Solve
        df = SolveProblem(Psi, Nu, F, Ic, J, Mesh, V, u, v, du, u_0, u_1, Dimensions, InitialState, Normal, Dir, NumberSteps, DeltaStretch)

        # Interpolation
        FolderPath = os.path.join('/home/msimon/Desktop/SHARED/ScriptsAndData/ExperimentalData/')
        FilePath = os.path.join(FolderPath, 'CR_SimpleShear_ExpDat.csv')
        ExpData = pd.read_csv(FilePath, sep=';', header=None, decimal=',')

        InterpExpData = interp1d(ExpData[0], ExpData[1], kind='linear', fill_value='extrapolate')
        InterpSimPred = interp1d(df.Stretches, df.P, kind='linear', fill_value='extrapolate')

        NumberPoints=int(FinalRelativeStretch / RelativeStepSize +1)
        XInterp = np.linspace(df.iloc[0][0],df.iloc[-1][0],NumberPoints)

        # Control Plot
        if Plot == True :        
            plt.rc('figure', figsize=[12,7])
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.cla()
            ax.plot(ExpData[0], ExpData[1],  color = 'b', linestyle = '--', label = 'Original Data', marker = 'o', markersize = 8, fillstyle='none')
            ax.plot(XInterp, InterpExpData(XInterp),  color = 'g', linestyle = '--', label = 'Interpolated data', marker = 'o', markersize = 15, fillstyle='none')
            ax.plot(df.Stretches, df.P,  color = 'r', linestyle = '--', label = 'Simulation Prediction', marker = 'o', markersize = 8, fillstyle='none')
            ax.plot(XInterp, InterpSimPred(XInterp),  color = 'k', linestyle = '--', label = 'Interpolated data', marker = 'o', markersize = 15, fillstyle='none')
            ax.set_xlabel('Stretch ratio (-)')
            ax.set_ylabel('Stresses (kPa)')
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.02))
            ax.legend(loc='upper left', frameon=True, framealpha=1)
            plt.title('Simple Shear')

        SimpleShearDelta2 = []
        for X in XInterp:
            SimpleShearDelta2.append(((InterpExpData(X)-InterpSimPred(X))/ InterpExpData(XInterp[-1]))**2)
            
            
        # Compute total cost
    if 'Compression' in LoadCases and 'Tension' in LoadCases and 'Simple Shear' in LoadCases:
        TotalCost = np.sum(CompressionDelta2)  * RelativeWeights[0] + np.sum(TensionDelta2) * RelativeWeights[1] + np.sum(SimpleShearDelta2) * RelativeWeights[2]
        TotalCost = TotalCost / np.sum(RelativeWeights)
    elif 'Compression' in LoadCases and 'Tension' in LoadCases and 'Simple Shear' not in LoadCases:
        TotalCost = np.sum(CompressionDelta2)  * RelativeWeights[0] + np.sum(TensionDelta2) * RelativeWeights[1]
        TotalCost = TotalCost / np.sum(RelativeWeights)
    elif 'Compression' in LoadCases and 'Tension' not in LoadCases and 'Simple Shear' in LoadCases:
        TotalCost = np.sum(CompressionDelta2)  * RelativeWeights[0] + np.sum(SimpleShearDelta2) * RelativeWeights[1]
        TotalCost = TotalCost / np.sum(RelativeWeights)
    elif 'Compression' not in LoadCases and 'Tension' in LoadCases and 'Simple Shear' in LoadCases:
        TotalCost = np.sum(TensionDelta2)  * RelativeWeights[0] + np.sum(SimpleShearDelta2) * RelativeWeights[1]
        TotalCost = TotalCost / np.sum(RelativeWeights)
    elif 'Compression' in LoadCases and 'Tension' not in LoadCases and 'Simple Shear' not in LoadCases:
        TotalCost = np.sum(CompressionDelta2)  * RelativeWeights[0]
        TotalCost = TotalCost / np.sum(RelativeWeights)
    elif 'Compression' not in LoadCases and 'Tension' in LoadCases and 'Simple Shear' not in LoadCases:
        TotalCost = np.sum(TensionDelta2)  * RelativeWeights[0]
        TotalCost = TotalCost / np.sum(RelativeWeights)
    elif 'Compression' not in LoadCases and 'Tension' not in LoadCases and 'Simple Shear' in LoadCases:
        TotalCost = np.sum(SimpleShearDelta2) * RelativeWeights[0]
        TotalCost = TotalCost / np.sum(RelativeWeights)
            
    print('Cost:', np.sum(TotalCost))
    
    FileName = open(str(NumberElements) + 'ElementsNHOptimizationResults.txt', 'a+')
#    FileName.write('%.3f %.3f %.3f\n' % (Mu, Alpha, TotalCost))
    FileName.write('%.3f %.3f %.3f\n' % (Mu, Lambda, TotalCost))
    FileName.close()    

    return np.sum(TotalCost)



def OgdenOptimization(Nu, LoadCases, RelativeWeights, NumberElements, FinalRelativeStretch, RelativeStepSize, Dimensions, Plot = False):
    
    # Initialize time tracking
    start = time.time()
    
    FileName = open(str(NumberElements) + 'ElementsNHOptimizationResults.txt', 'a+')
#    FileName.write('Mu Alpha TotalCost\n')
    FileName.write('Mu Lambda TotalCost\n')
    FileName.close() 
    
    Mu    = 0.66                          # (kPa)
    Alpha = -24.3                         # (-)
    D     = 3*(1-2*Nu)/(Mu*(1+Nu))        # (1/kPa)

    InitialGuess = np.array([Mu, Alpha])
    
    # Neo-Hookean (test)
    Mu    = 1.15                          # (kPa)
    InitialGuess = np.array([Mu])

    
    ResOpt = minimize(OgdenCostFunction, InitialGuess, args = (Nu, LoadCases, RelativeWeights, NumberElements, FinalRelativeStretch, RelativeStepSize, Dimensions), method = 'Nelder-Mead', options={'xatol':1E-1})
    
    [Mu, Alpha] = ResOpt.x
    print('Final Mu = ', Mu)
    print('Final Alpha =', Alpha)
    
    TimeName = open('OptimizationTime.txt', 'a+')
    TimeName.write('%1.3f\n' % (time.time() - start))
    TimeName.close()
        
    if Plot == True :
        
        df = pd.read_csv(str(NumberElements) + 'ElementsNHOptimizationResults.txt', sep=' ', decimal='.')
            
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
        
    return ResOpt
    