import numpy as np
import time
from stlpy.solvers.base import STLSolver
from stlpy.STL import LinearPredicate, NonlinearPredicate
from libs.bicycle_model import LinearAffineSystem

import gurobipy as gp
from gurobipy import GRB


class PCEMILPSolver(STLSolver):
    
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.LinearSystem`,
    solve the optimization problem

    .. math::

        \min & -\\rho^{\\varphi}(y_0,y_1,\dots,y_T) + \sum_{t=0}^T (x_t^TQx_t + u_t^TRu_t)

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = Al x_t + Bl u_t + El

        & hat{x}_{t+1} = Ap hat{x}_t + Bp v_t + Ep, \\ v_t \text{fixed}

        & \\rho^{\\varphi}(x_0,x_1,\dots,x_T, hat{z}_0, hat{z}_1, \dots, \hat{z}_T) \geq 0

    with Gurobi using mixed-integer convex programming. 

    .. note::

        This file is modified based upon "stlpy/solvers/gurobi/gurobi_micp.py"

    :param spec:            An :class:`.STLFormula` describing the specification.
    :param agents:          A dictionary containing :class:`.BicycleModel` describing agent models.
    :param T:               A positive integer :math:`T` describing control horizon.
    :param M:               (optional) A large positive scalar used to rewrite ``min`` and ``max`` as
                            mixed-integer constraints. Default is ``1000``.
    :param presolve:        (optional) A boolean indicating whether to use Gurobi's
                            presolve routines. Default is ``True``.
    :param verbose:         (optional) A boolean indicating whether to print detailed
                            solver info. Default is ``True``.
    """

    def __init__(self, spec, agents, T, M=1000, presolve=True, verbose=True):
        assert M > 0, "M should be a (large) positive scalar"
        
        self.agents = agents
        ego = agents["ego"]
        sys = LinearAffineSystem(ego.Al, ego.Bl, ego.Cl, ego.Dl, ego.El)
        super().__init__(spec, sys, ego.states[0], T, verbose)

        self.M = float(M)
        self.presolve = presolve

        # Set up the optimization problem
        self.model = gp.Model("PCE_STL_MICP")

        # Set some model parameters
        if not self.presolve:
            self.model.setParam('Presolve', 0)
        if not self.verbose:
            self.model.setParam('OutputFlag', 0)

        if self.verbose:
            print("---------------------------------------------------------")
            print("Setting up optimization variables (only done for once)...")
            st = time.time()  # for computing setup time

        # Create optimization variables
        self.x = self.model.addMVar((self.sys.n, self.T), lb=-float('inf'), name='x')
        self.u = self.model.addMVar((self.sys.m, self.T), lb=-float('inf'), name='u')
        self.rho = self.model.addMVar(1, name="rho", lb=0.0)  # lb sets minimum robustness

        if self.verbose:
            print(f"Optimization variables ready after {time.time() - st} seconds.")
            print("---------------------------------------------------------")
            print("Setting up STL constraints (only done for once)... ")
            print("This process may take up to 1-2 minutes, be patient...")
            st = time.time()

        # Set up STL constraints
        self.AddSTLConstraints()
        
        if self.verbose:
            print(f"STL constraints ready after {time.time() - st} seconds.")
            print("---------------------------------------------------------")
        
        if self.verbose:
            print(f"Initial setup complete. Ready for solving...")


    def AddControlBounds(self, u_min, u_max):
        for t in range(self.T):
            self.model.addConstr(u_min <= self.u[:, t])
            self.model.addConstr(self.u[:, t] <= u_max)

    def AddStateBounds(self, x_min, x_max):
        for t in range(self.T):
            self.model.addConstr(x_min <= self.x[:, t])
            self.model.addConstr(self.x[:, t] <= x_max)

    def AddQuadraticCost(self, t_curr):

        if t_curr < self.T:
            for t in range(t_curr, self.T):
                self.cost += self.u[:, t] @ self.agents['ego'].R @ self.u[:, t]
        else:
            self.cost += self.u[:, t_curr] @ self.agents['ego'].R @ self.u[:, t_curr]

        print(type(self.cost))

    def AddRobustnessCost(self):
        self.cost -= (1 * self.rho)

    def AddRobustnessConstraint(self, rho_min=0.0):
        self.model.addConstr(self.rho >= rho_min)

    def Solve(self):
        # Set the cost function now, right before we solve.
        # This is needed since model.setObjective resets the cost.
        self.model.setObjective(self.cost, GRB.MINIMIZE)

        # Do the actual solving
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            if self.verbose:
                print("\nOptimal Solution Found!\n")
            x = self.x.X
            u = self.u.X
            rho = self.rho.X[0]

            # Report optimal cost and robustness
            if self.verbose:
                print("Solve time: ", self.model.Runtime)
                print("Optimal robustness: ", rho)
                print("")
        else:
            if self.verbose:
                print(f"\nOptimization failed with status {self.model.status}.\n")
            x = None
            u = None
            rho = -np.inf

        return (x, u, rho, self.model.Runtime)


    def AddDynamicsConstraints(self, t_curr):

        self.model.update()
        ExistingConstrs = self.model.getConstrs()

        # History
        for t in range(t_curr + 1):
            self.model.addConstr( self.x[:,t] == self.agents['ego'].states[:,t])

        # Dynamics
        for t in range(t_curr, self.T - 1):
            self.model.addConstr( self.x[:,t+1] == self.x[:,t] + self.sys.A @ self.x[:,t] + self.sys.B @ self.u[:,t] + self.sys.E )
        
        self.model.update()
        # Get all history and dynamic constraints
        self.DynamicsConstrs = self.model.getConstrs()[len(ExistingConstrs):]
    

    def RemoveDynamicsConstraints(self):
        # Remove all existing history and dynamic constraints
        self.model.remove(self.DynamicsConstrs)
        self.model.update()
        self.DynamicsConstrs = []

    def AddInputConstraint(self, t, u):
        self.model.addConstr( self.u[:, t] == u )
        self.model.update()


    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        # Recursively traverse the tree defined by the specification
        # to add binary variables and constraints that ensure that
        # rho is the robustness value

        self.model.update()
        ExistingConstrs = self.model.getConstrs()
        
        z_spec = self.model.addMVar(1, vtype=GRB.CONTINUOUS)
        self.AddSubformulaConstraints(self.spec, z_spec, 0)
        self.model.addConstr(z_spec == 1)
        self.model.update()
        self.STLConstrs = self.model.getConstrs()[len(ExistingConstrs):]
    

    def AddSubformulaConstraints(self, formula, z, t):
        """
        Given an STLFormula (formula) and a binary variable (z),
        add constraints to the optimization problem such that z
        takes value 1 only if the formula is satisfied (at time t).

        If the formula is a predicate, this constraint uses the "big-M"
        formulation

            A[x(t);u(t)] - b + (1-z)M >= 0,

        which enforces A[x;u] - b >= 0 if z=1, where (A,b) are the
        linear constraints associated with this predicate.

        If the formula is not a predicate, we recursively traverse the
        subformulas associated with this formula, adding new binary
        variables z_i for each subformula and constraining

            z <= z_i  for all i

        if the subformulas are combined with conjunction (i.e. all
        subformulas must hold), or otherwise constraining

            z <= sum(z_i)

        if the subformulas are combined with disjuction (at least one
        subformula must hold).
        """
        # We're at the bottom of the tree, so add the big-M constraints
        if isinstance(formula, LinearPredicate):
            # a.T*x_t + c.T*_hat{z}_t - b + (1-z)*M >= rho

            if formula.name == "ego":
                self.model.addConstr(formula.a.T[:, :self.sys.n] @ self.x[:, t] - formula.b + (1 - z) * self.M >= self.rho)    
            else:
                self.model.addConstr(formula.a.T[:, :self.sys.n] @ self.x[:, t] + formula.a.T[:, self.sys.n:] @ self.agents[formula.name].pce_coefs[:, :, t].reshape(-1, 1) - formula.b + (1 - z) * self.M >= self.rho)

            # Force z to be binary
            b = self.model.addMVar(1, vtype=GRB.BINARY)
            self.model.addConstr(z == b)

        elif isinstance(formula, NonlinearPredicate):
            raise TypeError("Mixed integer programming does not support nonlinear predicates")

        # We haven't reached the bottom of the tree, so keep adding
        # boolean constraints recursively
        else:
            if formula.combination_type == "and":
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.model.addMVar(1, vtype=GRB.CONTINUOUS)
                    t_sub = formula.timesteps[i]  # the timestep at which this formula
                    # should hold
                    self.AddSubformulaConstraints(subformula, z_sub, t + t_sub)
                    self.model.addConstr(z <= z_sub)

            else:  # combination_type == "or":
                z_subs = []
                for i, subformula in enumerate(formula.subformula_list):
                    z_sub = self.model.addMVar(1, vtype=GRB.CONTINUOUS)
                    z_subs.append(z_sub)
                    t_sub = formula.timesteps[i]
                    self.AddSubformulaConstraints(subformula, z_sub, t + t_sub)
                self.model.addConstr(z <= sum(z_subs))

