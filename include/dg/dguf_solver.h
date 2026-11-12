#pragma once

#include "dg_solver.h"

enum Problem
{
    L2 = 0,       // L2 projection of f (mass matrix only)
    Helmholtz = 1 // Helmholtz problem
};

/*
 * dG ultraweak (dGUF) solver: first-order (mixed) formulation in (u, q), two fields per element.
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
               RhsFunction input_f_rhs,
               RhsFunction input_u_exact,
               RhsFunction input_du_exact,
               double input_tau,
               bool input_robin_stab = true,
               Problem input_problem = Helmholtz);

private:
    double tau;
    bool robin_stab;
    Problem problem;

    void assemble_volume() override;     // mass (+ half stiffness)
    void assemble_interfaces() override; // numerical fluxes
    void assemble_boundaries() override; // boundary fluxes
    std::string default_error_file() const override;

    void assemble_mass(std::complex<double> alpha);
    void assemble_half_stiffness(double beta);
};
