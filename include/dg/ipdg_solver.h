#pragma once

#include "dg_solver.h"

/*
 * Symmetric interior penalty dG (SIPG) solver for the Helmholtz problem
 *
 *      -kappa^2 u - u'' = f   in (0, L),
 *
 * with either homogeneous Dirichlet (u = 0) or Robin (d_n u + i kappa u = 0) boundary conditions,
 * following the formulation of the IPDG article (V3, eq. (sip)): find u_h in V_h = P_p(T_h) s.t.
 *
 *   b_h(u_h, w_h) = (f, w_h)   for all w_h in V_h, with
 *
 *   b_h(v, w) = -kappa^2 (v, w) + (v', w')_{T_h} + gamma/h <[v], [w]>_{F_h}
 *               - <{v'} n_F, [w]>_{F_h} - <[v], {w'} n_F>_{F_h}.
 *
 * Interfaces are oriented by n_F = +1, so that [v] = v(x^-) - v(x^+) and {v'} = (v'(x^-) + v'(x^+)) / 2.
 * Dirichlet: on boundary faces, [v] = {v} = v|_F and n_F is the outward normal (-1 at x = 0, +1 at x = L),
 *            i.e. the boundary is treated as a face (penalty + consistency terms).
 * Robin:     the face sums only run over interfaces, and the boundary term
 *            + i kappa <v, w>_{dOmega} is added (coming from -<d_n u, w> = i kappa <u, w>).
 *
 *   gamma : penalty parameter, must be larger than a threshold gamma_0 depending on p.
 */
class IPDGSolver : public DGSolver
{
public:
    IPDGSolver(const Mesh &input_mesh,
               size_t degree,
               double input_kappa,
               BCType input_bc_type,
               RhsFunction input_f_rhs,
               RhsFunction input_u_exact,
               RhsFunction input_du_exact,
               double input_gamma);

private:
    double gamma;

    void assemble_volume() override;     // -kappa^2 mass + stiffness
    void assemble_interfaces() override; // penalty + consistency terms on interfaces
    void assemble_boundaries() override; // Dirichlet faces or Robin term
    std::string default_error_file() const override;
    std::string extra_error_header() const override;
    std::string extra_error_columns(const ErrorNorms &norms) const override;

    void add_face_terms(const std::vector<size_t> &dofs,
                        const std::vector<double> &J,
                        const std::vector<double> &D,
                        double n_F);
};
