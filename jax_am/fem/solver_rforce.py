import jax
import jax.numpy as np
import numpy as onp
from jax.experimental.sparse import BCOO
import scipy
import time
from functools import partial

import petsc4py
# petsc4py.init()
from petsc4py import PETSc


################################################################################
# PETSc linear solver or JAX linear solver

def petsc_solve(A, b, ksp_type, pc_type):
    rhs = PETSc.Vec().createSeq(len(b))
    rhs.setValues(range(len(b)), onp.array(b))
    ksp = PETSc.KSP().create() 
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.setType(ksp_type)
    ksp.pc.setType(pc_type)
    print (f'PETSc - Solving with ksp_type = {ksp.getType()}, pc = {ksp.pc.getType()}') 
    x = PETSc.Vec().createSeq(len(b))
    ksp.solve(rhs, x) 
    return x.getArray()


def jax_solve(problem, A_fn, b, x0, precond):
    pc = get_jacobi_precond(jacobi_preconditioner(problem)) if precond else None
    #Bi-Conjugate Gradient Stable iteration to solve Ax = b; b is the RHS external force
    #here A_fn(x) is a function that computes the residual; namely RHS of the linear equation
    x, info = jax.scipy.sparse.linalg.bicgstab(A_fn, b, x0=x0, M=pc, tol=1e-10, atol=1e-10, maxiter=10000)
    RHS = A_fn(x)
    print(f"JAX scipy linear solve res = {np.linalg.norm(A_fn(x) - b)}")
    return x


################################################################################
# "row elimination" solver

def apply_bc_vec(res_vec, dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    res = res_vec.reshape(sol.shape)
    for i in range(len(problem.node_inds_list)):
        res = (res.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set
               (sol[problem.node_inds_list[i], problem.vec_inds_list[i]], unique_indices=True))
        res = res.at[problem.node_inds_list[i], problem.vec_inds_list[i]].add(-problem.vals_list[i]) #what does this line mean?
    return res.reshape(-1)


def apply_bc(res_fn, problem):
    def A_fn(dofs):
        """Apply Dirichlet boundary conditions
        """
        res_vec = res_fn(dofs)
        return apply_bc_vec(res_vec, dofs, problem)
    return A_fn


def row_elimination(res_fn, problem): #returns the residual vector
    def fn_dofs_row(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec)) #reshape assembled uiI into this [num_all_nodes, vec]
        res = res_fn(dofs).reshape(sol.shape)
        for i in range(len(problem.node_inds_list)):
            res = (res.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set
                   (sol[problem.node_inds_list[i], problem.vec_inds_list[i]], unique_indices=True)) #update b.c. in the assembled uiI?
        return res.reshape(-1)
    return fn_dofs_row #returns a function


def assign_bc(dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    for i in range(len(problem.node_inds_list)):
        sol = sol.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set(problem.vals_list[i])
    return sol.reshape(-1)

 
def assign_ones_bc(dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    for i in range(len(problem.node_inds_list)):
        sol = sol.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set(1.)
    return sol.reshape(-1)


def copy_bc(dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    new_sol = np.zeros_like(sol)
    for i in range(len(problem.node_inds_list)):
        new_sol = (new_sol.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set(sol[problem.node_inds_list[i], 
            problem.vec_inds_list[i]]))
    return new_sol.reshape(-1)


def get_flatten_fn(fn_sol, problem):
    def fn_dofs(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        val_sol = fn_sol(sol)
        return val_sol.reshape(-1)
    return fn_dofs


def get_A_fn_linear_fn(dofs, fn):
    """Not quite used.
    """
    def A_fn_linear_fn(inc):
        primals, tangents = jax.jvp(fn, (dofs,), (inc,))
        return tangents
    return A_fn_linear_fn


def get_A_fn_linear_fn_JFNK(dofs, fn):
    """Jacobian-free Newton–Krylov (JFNK) method. 
    Not quite used since we have auto diff to compute exact JVP.
    Knoll, Dana A., and David E. Keyes. 
    "Jacobian-free Newton–Krylov methods: a survey of approaches and applications." 
    Journal of Computational Physics 193.2 (2004): 357-397.
    """
    def A_fn_linear_fn(inc):
        EPS = 1e-3
        return (fn(dofs + EPS*inc) - fn(dofs))/EPS
    return A_fn_linear_fn


def operator_to_matrix(operator_fn, problem):
    """Only used for when debugging.
    Can be used to print the matrix, check the conditional number, etc.
    """
    J = jax.jacfwd(operator_fn)(np.zeros(problem.num_total_nodes*problem.vec))
    return J


def jacobi_preconditioner(problem):
    print(f"Compute and use jacobi preconditioner")
    jacobi = np.array(problem.A_sp_scipy.diagonal())
    jacobi = assign_ones_bc(jacobi.reshape(-1), problem) 
    return jacobi


def get_jacobi_precond(jacobi):
    def jacobi_precond(x):
        return x * (1./jacobi)
    return jacobi_precond


def test_jacobi_precond(problem, jacobi, A_fn):
    num_total_dofs = problem.num_total_nodes*problem.vec
    for ind in range(500):
        test_vec = np.zeros(num_total_dofs)
        test_vec = test_vec.at[ind].set(1.)
        print(f"{A_fn(test_vec)[ind]}, {jacobi[ind]}, ratio = {A_fn(test_vec)[ind]/jacobi[ind]}")

    print(f"test jacobi preconditioner")
    print(f"np.min(jacobi) = {np.min(jacobi)}, np.max(jacobi) = {np.max(jacobi)}")
    print(f"finish jacobi preconditioner")
 

def linear_guess_solve(problem, A_fn, precond, use_petsc):
    print(f"Linear guess solve...")
    # b = np.zeros((problem.num_total_nodes, problem.vec))
    b = problem.body_force + problem.neumann
    b = assign_bc(b, problem)
    if use_petsc:
        dofs = petsc_solve(A_fn, b, 'bcgsl', 'ilu')
    else:
        dofs = jax_solve(problem, A_fn, b, b, precond)
    return dofs


def linear_incremental_solver(problem, res_vec, A_fn, dofs, precond, use_petsc):
    """Lift solver
    """
    print(f"Solving linear system with lift solver...")
    b = -res_vec

    if use_petsc:
        inc = petsc_solve(A_fn, b, 'bcgsl', 'ilu')
    else:
        x0_1 = assign_bc(np.zeros_like(b), problem) 
        x0_2 = copy_bc(dofs, problem)
        x0 = x0_1 - x0_2
        inc = jax_solve(problem, A_fn, b, x0, precond) #what's jax_solve: jax linear solver

    dofs = dofs + inc #update of the disp
    return dofs


def get_A_fn(problem, use_petsc): #stiffness matrix
    print(f"Creating sparse matrix with scipy...")
    #csr_array((data, (row_ind, col_ind)), [shape=(M, N)] where data, row_ind and col_ind satisfy the relationship a[row_ind[k], col_ind[k]] = data[k]#
    A_sp_scipy = scipy.sparse.csr_array((problem.V, (problem.I, problem.J)), shape=(problem.num_total_dofs, problem.num_total_dofs)) #from core line 642
    print(f"Creating sparse matrix from scipy using JAX BCOO...")
    A_sp = BCOO.from_scipy_sparse(A_sp_scipy).sort_indices()
    print(f"self.A_sp.data.shape = {A_sp.data.shape}")
    print(f"Global sparse matrix takes about {A_sp.data.shape[0]*8*3/2**30} G memory to store.")
    problem.A_sp_scipy = A_sp_scipy

    def compute_linearized_residual(dofs): #i guess dofs is the assembled solution vector uiI
        return A_sp @ dofs #i guess this is the residual force

    if use_petsc:
        A = PETSc.Mat().createAIJ(size=A_sp_scipy.shape, csr=(A_sp_scipy.indptr, A_sp_scipy.indices, A_sp_scipy.data))
        for i in range(len(problem.node_inds_list)):
            row_inds = onp.array(problem.node_inds_list[i]*problem.vec + problem.vec_inds_list[i], dtype=onp.int32)
            A.zeroRows(row_inds)
    else:
        A = row_elimination(compute_linearized_residual, problem) #A here is a function of row_elimination
        #what's the size of A?
    return A


def solver_row_elimination(problem, linear, precond, initial_guess, use_petsc):
    """Imposing Dirichlet B.C. with "row elimination" method.
    """
    print(f"Calling the row elimination solver for imposing Dirichlet B.C.")
    print("Start timing")
    start = time.time()
    sol_shape = (problem.num_total_nodes, problem.vec)
    dofs = np.zeros(sol_shape).reshape(-1) #vector that has the solution

    def newton_update_helper(dofs):
        res_vec = problem.newton_update(dofs.reshape(sol_shape)).reshape(-1) #residual vector?->yes: newton_update from core [number_all_nodes * vec]
        res_vec = apply_bc_vec(res_vec, dofs, problem)
        A_fn = get_A_fn(problem, use_petsc) 
        return res_vec, A_fn

    # TODO: detect np.nan and assert
    if linear:
        dofs = assign_bc(dofs, problem)  #U vector
        res_vec, A_fn = newton_update_helper(dofs) #residual vector, Tangent stiffess?
        dofs = linear_incremental_solver(problem, res_vec, A_fn, dofs, precond, use_petsc)
    else:
        if initial_guess is None:
            res_vec, A_fn = newton_update_helper(dofs)
            # TODO: If dofs not satisfying B.C., nan occurs. Why?
            dofs = linear_guess_solve(problem, A_fn, precond, use_petsc)
        else:
            dofs = initial_guess.reshape(-1)

        res_vec, A_fn = newton_update_helper(dofs)
        res_val = np.linalg.norm(res_vec)
        print(f"Before, res l_2 = {res_val}") 
        tol = 1e-6
        while res_val > tol: #newton's iteration
            dofs = linear_incremental_solver(problem, res_vec, A_fn, dofs, precond, use_petsc)
            res_vec, A_fn = newton_update_helper(dofs)
            # test_jacobi_precond(problem, jacobi_preconditioner(problem, dofs), A_fn)
            res_val = np.linalg.norm(res_vec)
            print(f"res l_2 = {res_val}") 
            
    sol = dofs.reshape(sol_shape)
    end = time.time()
    solve_time = end - start
    print(f"Solve took {solve_time} [s]")
    print(f"max of sol = {np.max(sol)}")
    print(f"min of sol = {np.min(sol)}")

    return sol


################################################################################
# Lagrangian multiplier solver

def aug_dof_w_zero_bc(problem, dofs):
    aug_size = 0
    for i in range(len(problem.node_inds_list)):
        aug_size += len(problem.node_inds_list[i])
    for i in range(len(problem.p_node_inds_list_A)):
        aug_size += len(problem.p_node_inds_list_A[i])
    return np.hstack((dofs, np.zeros(aug_size)))


def aug_dof_w_bc(problem, dofs, p_num_eps):
    aug_d = np.array([])
    for i in range(len(problem.node_inds_list)):
        aug_d = np.hstack((aug_d, p_num_eps*problem.vals_list[i]))
    for i in range(len(problem.p_node_inds_list_A)):
        aug_d = np.hstack((aug_d, np.zeros(len(problem.p_node_inds_list_A[i]))))
    return np.hstack((dofs, aug_d))


def linear_guess_solve_lm(problem, A_aug, p_num_eps, use_petsc):
    b = (problem.body_force + problem.neumann).reshape(-1)
    b_aug = aug_dof_w_bc(problem, b, p_num_eps)
    if use_petsc:
        dofs_aug = petsc_solve(A_aug, b_aug, 'minres', 'none')
    else:
        x0 = np.zeros((problem.num_total_nodes, problem.vec))
        x0 = assign_bc(x0, problem)
        x0 = aug_dof_w_zero_bc(problem, x0)
        dofs_aug = jax_solve(problem, A_aug, b_aug, x0, None)
    return dofs_aug


def linear_incremental_solver_lm(problem, A_aug, res_vec_aug, dofs_aug, p_num_eps, use_petsc):
    b_aug = -res_vec_aug
    if use_petsc:
        inc_aug = petsc_solve(A_aug, b_aug, 'minres', 'none')
    else:
        inc_aug = jax_solve(problem, A_aug, b_aug, None, None)
    dofs_aug = dofs_aug + inc_aug
    return dofs_aug


def compute_residual_lm(problem, res_vec, dofs_aug, p_num_eps):
    d_splits = np.cumsum(np.array([len(x) for x in problem.node_inds_list])).tolist()
    p_splits = np.cumsum(np.array([len(x) for x in problem.p_node_inds_list_A])).tolist()

    d_lmbda_len = d_splits[-1] if len(d_splits) > 0 else 0
    p_lmbda_len = p_splits[-1] if len(p_splits) > 0 else 0

    def get_Lagrangian():
        def split_lamda(lmbda):
            d_lmbda = lmbda[:d_lmbda_len]
            p_lmbda = lmbda[d_lmbda_len:]
            d_lmbda_split = np.split(d_lmbda, d_splits)
            p_lmbda_split = np.split(p_lmbda, p_splits)
            return d_lmbda_split, p_lmbda_split

        # @jax.jit
        def Lagrangian_fn(dofs_aug):
            dofs, lmbda = dofs_aug[:problem.num_total_dofs], dofs_aug[problem.num_total_dofs:]
            sol = dofs.reshape((problem.num_total_nodes, problem.vec))
            d_lmbda_split, p_lmbda_split = split_lamda(lmbda)
            lag = 0.
            for i in range(len(problem.node_inds_list)):
                lag += np.sum(d_lmbda_split[i] * (sol[problem.node_inds_list[i], problem.vec_inds_list[i]] - problem.vals_list[i]))

            for i in range(len(problem.p_node_inds_list_A)):
                lag += np.sum(p_lmbda_split[i] * (sol[problem.p_node_inds_list_A[i], problem.p_vec_inds_list[i]] - 
                                                    sol[problem.p_node_inds_list_B[i], problem.p_vec_inds_list[i]]))
            return p_num_eps*lag

        return Lagrangian_fn

    Lagrangian_fn = get_Lagrangian()
    A_fn = jax.grad(Lagrangian_fn) #tangent stiffness matrix
    res_vec_1 = A_fn(dofs_aug)
    res_vec_2 = aug_dof_w_zero_bc(problem, res_vec)
    res_vec_aug = res_vec_1 + res_vec_2

    return res_vec_aug


def get_A_fn_and_res_aug(problem, dofs_aug, res_vec, p_num_eps, use_petsc):
    def symmetry(I, J, V):
        I_sym = onp.hstack((I, J))
        J_sym = onp.hstack((J, I))
        V_sym = onp.hstack((V, V))
        return I_sym, J_sym, V_sym

    I_d = onp.array([])
    J_d = onp.array([])
    V_d = onp.array([])
    group_index = problem.num_total_dofs
    for i in range(len(problem.node_inds_list)):
        group_size = len(problem.node_inds_list[i])
        I_d = onp.hstack((I_d, problem.vec*problem.node_inds_list[i] + problem.vec_inds_list[i]))
        J_d = onp.hstack((J_d, group_index + onp.arange(group_size)))
        V_d = onp.hstack((V_d, p_num_eps*onp.ones(group_size)))
        group_index += group_size
    I_d_sym, J_d_sym, V_d_sym = symmetry(I_d, J_d, V_d)

    I_p = onp.array([])
    J_p = onp.array([])
    V_p = onp.array([])
    for i in range(len(problem.p_node_inds_list_A)):
        group_size = len(problem.p_node_inds_list_A[i])
        I_p = onp.hstack((I_p, problem.vec*problem.p_node_inds_list_A[i] + problem.p_vec_inds_list[i]))
        J_p = onp.hstack((J_p, group_index + onp.arange(group_size)))
        V_p = onp.hstack((V_p, p_num_eps*onp.ones(group_size)))
        I_p = onp.hstack((I_p, problem.vec*problem.p_node_inds_list_B[i] + problem.p_vec_inds_list[i]))
        J_p = onp.hstack((J_p, group_index + onp.arange(group_size)))
        V_p = onp.hstack((V_p, -p_num_eps*onp.ones(group_size)))
        group_index += group_size
    I_p_sym, J_p_sym, V_p_sym = symmetry(I_p, J_p, V_p)

    I = onp.hstack((problem.I, I_d_sym, I_p_sym))
    J = onp.hstack((problem.J, J_d_sym, J_p_sym))
    V = onp.hstack((problem.V, V_d_sym, V_p_sym))

    print(f"Aug - Creating sparse matrix with scipy...")
    A_sp_scipy_aug = scipy.sparse.csc_array((V, (I, J)), shape=(group_index, group_index))
    print(f"Aug - Creating sparse matrix from scipy using JAX BCOO...")
    A_sp_aug = BCOO.from_scipy_sparse(A_sp_scipy_aug).sort_indices()
    print(f"Aug - self.A_sp.data.shape = {A_sp_aug.data.shape}")
    print(f"Aug - Global sparse matrix takes about {A_sp_aug.data.shape[0]*8*3/2**30} G memory to store.")
    problem.A_sp_scipy_aug = A_sp_scipy_aug

    def compute_linearized_residual(dofs_aug):
        return A_sp_aug @ dofs_aug

    if use_petsc:
        A_aug = PETSc.Mat().createAIJ(size=A_sp_scipy_aug.shape, csr=(A_sp_scipy_aug.indptr, A_sp_scipy_aug.indices, A_sp_scipy_aug.data))
    else:
        A_aug = compute_linearized_residual

    res_vec_aug = compute_residual_lm(problem, res_vec, dofs_aug, p_num_eps)

    return A_aug, res_vec_aug


def solver_lagrange_multiplier(problem, linear, use_petsc=True):
    """Imposing Dirichlet B.C. and periodic B.C. with lagrangian multiplier method.

    The global matrix is of the form 
    [A   B 
     B^T 0]
    JAX built solver gmres and bicgstab sometimes fail to solve such a system.
    PESTc solver minres seems to work. 
    TODO: explore which solver in PESTc is the best, and which preconditioner should be used.

    Reference:
    https://ethz.ch/content/dam/ethz/special-interest/baug/ibk/structural-mechanics-dam/education/femI/Presentation.pdf
    """
    print(f"Calling the lagrange multiplier solver for imposing Dirichlet B.C. and periodic B.C.")
    print("Start timing")
    start = time.time()
    sol_shape = (problem.num_total_nodes, problem.vec)
    dofs = np.zeros(sol_shape).reshape(-1)

    # Ad-hoc parameter to get a better conditioned global matrix.
    if hasattr(problem, 'p_num_eps'):
        p_num_eps = problem.p_num_eps
    else:
        p_num_eps = 1.
    print(f"Setting p_num_eps = {p_num_eps}. If periodic B.C. fails to be applied, consider modifying this parameter.")

    def newton_update_helper(dofs_aug):
        res_vec = problem.newton_update(dofs_aug[:problem.num_total_dofs].reshape(sol_shape)).reshape(-1)
        A_aug, res_vec_aug = get_A_fn_and_res_aug(problem, dofs_aug, res_vec, p_num_eps, use_petsc)
        return res_vec_aug, A_aug

    if linear:
        # If we know the problem is linear, this way of solving seems faster.
        dofs = assign_bc(dofs, problem)
        dofs_aug = aug_dof_w_zero_bc(problem, dofs)
        res_vec_aug, A_aug = newton_update_helper(dofs_aug)
        dofs_aug = linear_incremental_solver_lm(problem, A_aug, res_vec_aug, dofs_aug, p_num_eps, use_petsc)
    else:
        dofs_aug = aug_dof_w_zero_bc(problem, dofs)
        res_vec_aug, A_aug = newton_update_helper(dofs_aug)
        dofs_aug = linear_guess_solve_lm(problem, A_aug, p_num_eps, use_petsc)

        res_vec_aug, A_aug = newton_update_helper(dofs_aug)
        res_val = np.linalg.norm(res_vec_aug)
        print(f"Before, res l_2 = {res_val}") 
        tol = 1e-6
        while res_val > tol:
            dofs_aug = linear_incremental_solver_lm(problem, A_aug, res_vec_aug, dofs_aug, p_num_eps, use_petsc)
            res_vec_aug, A_aug = newton_update_helper(dofs_aug)
            res_val = np.linalg.norm(res_vec_aug)
            print(f"res l_2 dofs_aug = {res_val}") 

    sol = dofs_aug[:problem.num_total_dofs].reshape(sol_shape)
    end = time.time()
    solve_time = end - start
    print(f"Solve took {solve_time} [s]")
    print(f"max of sol = {np.max(sol)}")
    print(f"min of sol = {np.min(sol)}")

    return sol


################################################################################
# General

def solver(problem, linear=False, precond=True, initial_guess=None, use_petsc=False):
    """periodic B.C. is a special form of adding a linear constraint. 
    Lagrange multiplier seems to be convenient to impose this constraint.
    """
    # TODO: print platform jax.lib.xla_bridge.get_backend().platform
    # and suggest PETSc or jax solver
    if problem.periodic_bc_info is None:
        return solver_row_elimination(problem, linear, precond, initial_guess, use_petsc)
    else:
        return solver_lagrange_multiplier(problem, linear, use_petsc)


################################################################################
# Implicit differentiation with the adjoint method

def implicit_vjp(problem, sol, params, v):
    def constraint_fn(dofs, params): #重点在于理解这个函数;这里其实是固定解u，而给param一个变化
        """c(u, p) constraint discretized PD residual force?
        """
        problem.set_params(params)
        res_fn = problem.compute_residual #this is a function that returns a value; the input argument is sol
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs) #residual force for this case here

    def constraint_fn_sol_to_sol(sol, params):
        return constraint_fn(sol.reshape(-1), params).reshape(sol.shape)

    def get_vjp_contraint_fn_dofs(dofs):
        # a function that computes LHS of the adjoint equation
        def adjoint_linear_fn(adjoint):
            primals, f_vjp = jax.vjp(A_fn, dofs) #A_fn is Stiffness matrix, as defined later on
            val, = f_vjp(adjoint) #\lambda.T * \partial C \partial U
            return val #\partial J \partial U
        return adjoint_linear_fn #return a linear system of equations that computes \partial J \partial U

    def get_partial_params_c_fn(sol):
        """c(u=u, p) #return ????????????????????????????????
        """
        def partial_params_c_fn(params):
            return constraint_fn_sol_to_sol(sol, params) #reshaped residual force 这个是怎么算出来的 \partial\C partial \theta
        return partial_params_c_fn

    def get_vjp_contraint_fn_params(params, sol):
        """v*(partial dc/dp) v = \lambda^T here; this is the first term in dJ/dtheta
        """
        partial_c_fn = get_partial_params_c_fn(sol)
        def vjp_linear_fn(v):
            primals, f_vjp = jax.vjp(partial_c_fn, params) #f_vjp = \partial C   \partial theta
            val, = f_vjp(v)
            return val
        return vjp_linear_fn #returns a function THAT COMPUTES THE FIRST TERM IN dJ dtheta

    problem.set_params(params)
    problem.newton_update(sol)
    A_fn = get_A_fn(problem, use_petsc=False) #what's this one? ->stiffness matrix FUN A(dof) = res
    adjoint_linear_fn = get_vjp_contraint_fn_dofs(sol.reshape(-1)) #adjoint * \partial res \partial U
    adjoint = jax_solve(problem, adjoint_linear_fn, v.reshape(-1), None, True) #I guess v is dJ/du   solving this eqn: adjoint_linear_fn *x = v
    vjp_linear_fn = get_vjp_contraint_fn_params(params, sol) #returns a function that take the cotangent vector as the input
    vjp_result = vjp_linear_fn(adjoint.reshape(sol.shape)) #first term in the dJd \theta
    vjp_result = jax.tree_map(lambda x: -x, vjp_result) #what does this line mean? I guess first term in dJ/dtheta should be a vector an example on this>

    return vjp_result #first term in the dJd \theta


def ad_wrapper(problem, linear=False, use_petsc=False): #needs to call this problem for the forward prediction
    @jax.custom_vjp
    def fwd_pred(params):
        problem.set_params(params)
        sol = solver(problem, linear=linear, use_petsc=use_petsc)
        return sol 
 
    def f_fwd(params):
        sol = fwd_pred(params)
        return sol, (params, sol)

    def f_bwd(res, v):
        print("\nRunning backward...")
        params, sol = res 
        vjp_result = implicit_vjp(problem, sol, params, v)
        return (vjp_result,)

    fwd_pred.defvjp(f_fwd, f_bwd)

    def rf_fun(params):
        sol = fwd_pred(params)
        rf = problem.compute_rforce(sol, params)
        return rf
    return fwd_pred, rf_fun 


