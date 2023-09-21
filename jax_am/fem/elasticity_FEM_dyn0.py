import jax
import jax.numpy as np
import os
import matplotlib.pyplot as plt
import numpy as onp

from core_dyn import FEM
from solver_fem_dyn import solver, ad_wrapper
from utils import modify_vtu_file, save_sol1
from generate_mesh import Mesh, box_mesh
from jax.config import config
from read_mesh import read_mesh_ABQ_3D
config.update("jax_enable_x64", True)


gpu_idx = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)


class LinearElasticity(FEM):
    
    def get_tensor_map(self): #stress-strain relationship
        def stress(u_grad):
            E = 30
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress

    
    def gauss2nodal(self, values):
        ave_values = np.average(values, axis = 1).reshape(-1, 1) #(num_cells, 1)
        ave_values = np.repeat(ave_values, self.num_nodes, axis = 1) #(num_cells, num_nodes_per_ele)
        ave_nodal_values = []
        for i in range(self.num_total_nodes):
            idx = np.array(i, dtype= onp.int64)
            indices_mask = np.isin(self.cells, idx)
            ave_nodal = np.average(ave_values[indices_mask])
            ave_nodal_values.append(ave_nodal)
        ave_nodal_values = np.array(ave_nodal_values)
        return ave_nodal_values
    
    def update_acc_vel_til(self): #how to enforce the essential b.c. here?
        self.sol_til   = self.sol + self.dt * self.sol_v + (self.dt ** 2) * 0.5 * (1 - 2 * self.beta) * self.sol_a 
        self.sol_v_til = self.sol_v + (1 - self.gamma) * self.dt * self.sol_a 

    def update_acc_vel(self): #accounting for the modified essential b.c.
        self.sol_a = 1./ (self.beta * self.dt ** 2) * (self.sol - self.sol_til)
        self.sol_v = self.sol_v_til + self.gamma * self.dt * self.sol_a
        #print(self.sol_v)
    
    def init(self):
       
        neumann = self.compute_Neumann_integral_vars(**self.internal_vars) #(num_total_nodes, vec)
        f = self.body_force.reshape(-1,1) + neumann.reshape(-1,1) #self.mass_BCOO
        self.sol_a, _ = jax.scipy.sparse.linalg.bicgstab(self.mass_BCOO, f, x0=None, tol=1e-10, atol=1e-10, maxiter=1000000000)
        self.sol_a = self.sol_a.reshape(self.num_total_nodes, self.vec)
        # modify the essential boundary nodes 
        for i in range(len(self.node_inds_list)):
            self.sol_a = self.sol_a.at[self.node_inds_list[i], self.vec_inds_list[i]].set(0.)
        #print(self.sol_a)

# data_dir = os.path.join(os.path.dirname(__file__), 'data')
# inp_dir = os.path.join(os.path.dirname(__file__), 'inp')
# input_file_name = 'dyn_body_force.inp'
# etype = 'C3D8'

# XY_host, Elem_nodes_host, _, _ = read_mesh_ABQ_3D(inp_dir, input_file_name, etype)
# mesh = Mesh(XY_host, Elem_nodes_host)

ele_type = 'HEX8'
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 1., 1., 1.
meshio_mesh = box_mesh(Nx = 20, Ny = 20, Nz = 20, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])



def top(point):
    return np.isclose(point[2], Lz, atol=1e-5)

def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)


def dirichlet_val_bottom(point):
    return 0.

def get_dirichlet_top(disp):
    def val_fn(point):
        return disp
    return val_fn


def body_force(point):
    val = np.array([0.0, 0.0, 1.0])
    return val





max_step = 100

# location_fns = [bottom, top]
# value_fns = [dirichlet_val_bottom, get_dirichlet_top(0.)]
# vecs = [2, 2]

#acc boundary condition
location_fns = [bottom]
value_fns = [dirichlet_val_bottom]
vecs = [2]

dirichlet_bc_info = [location_fns, vecs, value_fns]

problem = LinearElasticity(mesh, vec=3, dim=3, dirichlet_bc_info = dirichlet_bc_info, source_info = body_force)


#set up Newmark parametes
problem.density = 1.
problem.gamma = 0.7 #>=0.5 
problem.beta = 0.6 * problem.gamma #>=0.5 * gamma

#initialize disp, vel and acc
problem.sol   = np.zeros((problem.num_total_nodes, 3)) #initial B.C.
problem.sol_v = np.zeros((problem.num_total_nodes, 3))  #initial B.C.
problem.sol_a = np.zeros((problem.num_total_nodes, 3)) 
problem.dt = 0.01
time_total = 1
max_same_inc_step = 4
same_inc_step = 0
disp0 = 0.
time0 = 0.
#initialization the 0 step
problem.init()
for step in range(max_step):
    time = time0 + problem.dt
    if time <=  time_total:
        print(f"\n Current time to be solved = {time} \n")
        print(f"Current time inc = {problem.dt} \n")
        problem.update_acc_vel_til()
        problem.sol, flag = solver(problem, use_petsc = False)
        if flag:
            same_inc_step = same_inc_step + 1
            problem.update_acc_vel()           
            vtk_path = os.path.join(data_dir, f'vtk/FEM_dyn_U{step:03d}.vtu')
            save_sol1(problem, problem.sol, vtk_path)
            vtk_path = os.path.join(data_dir, f'vtk/FEM_dyn_A{step:03d}.vtu')
            save_sol1(problem, problem.sol_a, vtk_path)
            vtk_path = os.path.join(data_dir, f'vtk/FEM_dyn_V{step:03d}.vtu')
            save_sol1(problem, problem.sol_v, vtk_path)     
            time0 = time
        else:            
            print('half time step size')
            problem.dt = problem.dt / 2.
            same_inc_step = 0
        
    else:
        if  time0 < time_total:
            time = time_total
            print(f"\n last step \n")
            print(f"Current time inc = {problem.dt} \n")
            problem.update_acc_vel_til()  
            problem.sol, flag = solver(problem, use_petsc = False)
            if flag:
                same_inc_step = same_inc_step + 1
                problem.update_acc_vel() 
                vtk_path = os.path.join(data_dir, f'vtk/FEM_dyn_U{step:03d}.vtu')
                save_sol1(problem, problem.sol, vtk_path)
                vtk_path = os.path.join(data_dir, f'vtk/FEM_dyn_A{step:03d}.vtu')
                save_sol1(problem, problem.sol_a, vtk_path)
                vtk_path = os.path.join(data_dir, f'vtk/FEM_dyn_V{step:03d}.vtu')
                save_sol1(problem, problem.sol_v, vtk_path)
                time0 = time
            else:            
                print('half time step size')
                problem.dt = problem.dt / 2.
                same_inc_step = 0
            
        break