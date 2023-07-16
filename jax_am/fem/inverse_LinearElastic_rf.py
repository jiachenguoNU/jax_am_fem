import numpy as onp
import jax
import jax.numpy as np
import meshio
import time
import os
import glob
import scipy.optimize as opt
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from core import FEM
from solver_rforce import solver, ad_wrapper
from utils import modify_vtu_file, save_sol
from generate_mesh import Mesh, box_mesh

onp.random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# TODO
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class LinearElasticity(FEM):
    def custom_init(self, name):
        self.name = name
    
    def get_tensor_map(self): #stress-strain relationship
        def stress(u_grad, theta):
            E = theta[0]
            nu = theta[1]
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress

    # def get_body_map(self): #what's this one?
    #     return lambda x: x



    def set_params(self, params): #this parameter is only the parameter to be optimized

        theta0 = params.reshape((1, -1))[0]
        theta0 = theta0[None, None, :] #dim [num_cell, num_gauss, num_para_1cell]
        theta  = np.repeat(theta0, self.num_cells, axis = 0)
        theta_gauss = np.repeat(theta, self.num_quads, axis = 1) #dim: (num_cell, num_quads)
        
        self.internal_vars['laplace']= [theta_gauss]
    
    def compute_rforce(self, sol, param): #partial derivative 2nd term how to get
        # compute reaction force

        def location_fns(point):
            return np.isclose(point[0], 1.0, atol=1e-5) #find mesh points that x-coord is close to surf_coord
        node_inds = onp.argwhere(jax.vmap(location_fns)(self.mesh.points)).reshape(-1)

        theta0 = param.reshape((1, -1))[0]
        theta0 = theta0[None, None, :] #dim [num_cell, num_gauss, num_para_1cell]
        theta  = np.repeat(theta0, self.num_cells, axis = 0)
        theta_gauss = np.repeat(theta, self.num_quads, axis = 1) #dim: (num_cell, num_quads)

        internal_vars = {"laplace" : [theta_gauss]}
        nodal_force = self.compute_residual_vars(sol, **internal_vars) + self.neumann #it's actually related 2 varibales; but the main func only is defined on 1 variable
        neumann_force = nodal_force[node_inds, :].reshape(-1, self.vec)
        reaction_force = np.sum(neumann_force)
        return reaction_force
    

#ground truth generation: using a finite element forward simulation as the DIC result
ele_type = 'HEX8'
data_dir = os.path.join(os.path.dirname(__file__), 'data')
meshio_mesh = box_mesh(2, 2, 2, 1., 1., 1., data_dir, ele_type=ele_type)


mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1., atol=1e-5)

def zero_dirichlet_val(point):
    return 0.

def dirichlet_val(point):
    return 0.1

dirichlet_bc_info = [[left, left, left, right, right, right], 
                        [0, 1, 2, 0, 1, 2], 
                        [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, 
                        dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]]

problem_name = "DIC"                 
DIC = LinearElasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info, additional_info=(problem_name,))
DIC.set_params(onp.array([100., 0.3])) #set the material parameters for the analysis

DIC_sol = solver(DIC, linear=True, precond=True)
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(DIC, DIC_sol, vtk_path)

#############################################################################################################################
#inverse problem def starts
#############################################################################################################################
data_dir = os.path.join(os.path.dirname(__file__), 'data')
files = glob.glob(os.path.join(data_dir, f'vtk/inverse/*'))


problem_inv_name = "inverse"
problem_inv = LinearElasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info, additional_info=(problem_inv_name,))
#I didn't set any material parameters here because now it's a variable to be optimized
fwd_pred, rf_fun = ad_wrapper(problem_inv, linear=True)
files = glob.glob(os.path.join(data_dir, f'vtk/{problem_inv_name}/*'))
for f in files:
    os.remove(f)


def J_fn(dofs, DIC_solution):
    """J(u, p) #l2 loss of u
    """
    sol = dofs.reshape((problem_inv.num_total_nodes, problem_inv.vec))
    pred_vals = sol
    assert pred_vals.shape == DIC_solution.shape
    l2_loss = np.sum((pred_vals - DIC_solution)**2) 
    # l2_loss = problem_fwd.compute_L2(true_sol - sol)
    # reg = 1e-5*problem_fwd.compute_L2(params)
    # print(f"{bcolors.HEADER}Predicted force L2 integral = {problem_fwd.compute_L2(params)}{bcolors.ENDC}")
    return l2_loss

def J_fn_rf(dofs, rf, DIC_solution):
    """J(u, p) #l2 loss of u
    """
    lam = 0.1  #penalty number for the reaction force term
    #kinematic field
    sol = dofs.reshape((problem_inv.num_total_nodes, problem_inv.vec))
    pred_vals = sol
    assert pred_vals.shape == DIC_solution.shape
 
    rf_loss = lam * (rf - 11.298076923076922)**2
    # l2_loss = problem_fwd.compute_L2(true_sol - sol)
    # reg = 1e-5*problem_fwd.compute_L2(params)
    # print(f"{bcolors.HEADER}Predicted force L2 integral = {problem_fwd.compute_L2(params)}{bcolors.ENDC}")
    return rf_loss

def J_total(params, DIC_solution):
    """J(u(p), p)
                    
    """ 
    #kinematic field    
    sol = fwd_pred(params) #u(theta) fwd_pred: primal is the u(theta); jacobian is the first term in dJ/d theta; calls the problem set parameters
    dofs = sol.reshape(-1)
    rf = rf_fun(params)
    #reaction force
    #rf = problem_inv.compute_rforce(sol, 1.0) #感觉这里是有问题的，因为这里problem_inv并没有更新

    obj_val = J_fn(dofs, DIC_solution) + J_fn_rf(dofs, rf, DIC_solution)
    return obj_val

outputs = []
def output_sol(params, obj_val):
    print(f"\nOutput solution - need to solve the forward problem again...")
    sol = fwd_pred(params)
    vtu_path = os.path.join(data_dir, f"vtk/{problem_inv_name}/sol_{output_sol.counter:03d}.vtu")
    save_sol(problem_inv, sol, vtu_path)

    print(f"loss = {obj_val}")
    print(f"current parameters = {params}")
    outputs.append([obj_val])
    output_sol.counter += 1

output_sol.counter = 0


#no need to change this part
#define the initial parameters for optimization
params_ini = onp.array([100., 0.3])

def objective_wrapper(x):
    obj_val, dJ = jax.value_and_grad(J_total, (0,))(x, DIC_sol)
    objective_wrapper.dJ = dJ
    output_sol(x, obj_val)
    print(f"{bcolors.HEADER}obj_val = {obj_val}{bcolors.ENDC}")
    return obj_val

def derivative_wrapper(x):
    grads = objective_wrapper.dJ
    print(f"grads.shape = {grads}")
    # 'L-BFGS-B' requires the following conversion, otherwise we get an error message saying
    # -- input not fortran contiguous -- expected elsize=8 but got 4
    return onp.array(grads, order='F', dtype=onp.float64)

bounds = [(50, 350),(0.2, 0.4)]
options = {'maxiter': 20, 'disp': True, 'ftol': 1e-5}  # CG or L-BFGS-B or Newton-CG or SLSQP
res = opt.minimize(fun=objective_wrapper,
                    x0=params_ini,
                    method='SLSQP',
                    jac=derivative_wrapper,
                    bounds=bounds,
                    callback=None,
                    options=options)



