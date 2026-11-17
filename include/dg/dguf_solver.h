#pragma once

#include "dg_solver.h"

enum Problem
{
    L2 = 0,       // L2 projection of f (mass matrix only)
    Helmholtz = 1 // Helmholtz problem
};

/*
 * dG ultraweak (dGUF) solver: first-order formulation in (u, q), with fields u, q_1, ..., q_d per element.
 * For all test functions (r, s) (s vector-valued) and all elements K:
 *
 *   i kappa (u, r)_K - (q, grad r)_K + < q^ . n_K, r >_{dK} = (f, r)_K,
 *   i kappa (q, s)_K - (u, div s)_K  + < u^, s . n_K >_{dK}  = 0,
 *
 * with the numerical fluxes on an interface F (n_F = n_{K_+}, [v] = v_+ - v_-, {v} = (v_+ + v_-)/2)
 *
 *   u^ = {u} + 1/(2 tau) [q] . n_F,        q^ . n_F = {q} . n_F + tau/2 [u],
 *
 * and on a boundary face (n outward), "u-rows" / "q-rows" denoting the terms tested with r / s . n:
 *
 *   Dirichlet         : u-rows  q . n + tau u                     q-rows  0
 *   Robin, robin_stab : u-rows  (q . n + tau u) / (1 + tau)        q-rows  (q . n + tau u) / (1 + tau)
 *   Robin, no stab    : u-rows  u                                  q-rows  q . n
 *   Neumann           : u-rows  0                                  q-rows  q . n / tau + u
 *
 *   tau        : stabilization parameter of the numerical fluxes,
 *   robin_stab : Robin boundary flux with (true) or without (false) the tau-stabilization.
 */
class DGUFSolver : public DGSolver
{
public:
    DGUFSolver(const Mesh &input_mesh,
               size_t degree,
               double input_kappa,
               BCType input_bc_type,
               ScalarFunction input_f_rhs,
               ScalarFunction input_u_exact,
               VectorFunction input_grad_u_exact,
               double input_tau,
               bool input_robin_stab = true,
               Problem input_problem = Helmholtz);

private:
    double tau;
    bool robin_stab;
    Problem problem;

    void assemble_element(size_t e) override;          // mass (+ half stiffness)
    void assemble_interface(const Face &F) override;     // numerical fluxes
    void assemble_boundary_face(const Face &F) override; // boundary fluxes
    std::string default_error_file() const override;
};
