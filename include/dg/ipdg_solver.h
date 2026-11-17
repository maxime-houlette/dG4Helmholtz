#pragma once

#include "dg_solver.h"

/*
 * Symmetric interior penalty dG (SIPG) solver for the Helmholtz problem
 *
 *      -kappa^2 u - Delta u = f   in Omega,
 *
 * with either homogeneous Dirichlet (u = 0) or Robin (d_n u + i kappa u = 0) boundary conditions,
 * following the formulation of the IPDG article (V3, eq. (sip)): find u_h in V_h = P_p(T_h) s.t.
 *
 *   b_h(u_h, w_h) = (f, w_h)   for all w_h in V_h, with
 *
 *   b_h(v, w) = -kappa^2 (v, w) + (grad_h v, grad_h w) + < gamma/h_F [v], [w] >_{F_h}
 *               - < {grad_h v} . n_F, [w] >_{F_h} - < [v], {grad_h w} . n_F >_{F_h}.
 *
 * Interfaces are oriented by n_F = n_{K_+}, [v] = v_+ - v_-, {v} = (v_+ + v_-) / 2.
 * Dirichlet: on boundary faces, [v] = {v} = v|_F and n_F is the outward normal,
 *            i.e. the boundary is treated as a face (penalty + consistency terms).
 * Robin:     the face sums only run over interfaces, and the boundary term
 *            + i kappa <v, w>_{dOmega} is added (coming from -<d_n u, w> = i kappa <u, w>).
 *
 *   gamma : penalty parameter, must be larger than a threshold gamma_0 depending on p;
 *           it is scaled by the local face size h_F (min of the diameters of the adjacent elements).
 */
class IPDGSolver : public DGSolver
{
public:
    IPDGSolver(const Mesh &input_mesh,
               size_t degree,
               double input_kappa,
               BCType input_bc_type,
               ScalarFunction input_f_rhs,
               ScalarFunction input_u_exact,
               VectorFunction input_grad_u_exact,
               double input_gamma);

private:
    double gamma;

    void assemble_element(size_t e) override;          // -kappa^2 mass + stiffness
    void assemble_interface(const Face &F) override;     // penalty + consistency terms
    void assemble_boundary_face(const Face &F) override; // Dirichlet face or Robin term
    std::string default_error_file() const override;
    std::string extra_error_header() const override;
    std::string extra_error_columns(const ErrorNorms &norms) const override;

    /* Adds w (gamma/h_F J[c] J[a] - D[c] J[a] - J[c] D[a]) to A_loc(a, c), where J[k] (resp. D[k]) is the
     * contribution of local dof k to [v] (resp. {grad v} . n_F) at a face quadrature point. */
    void add_face_terms(Eigen::MatrixXcd &A_loc, const Eigen::VectorXd &J, const Eigen::VectorXd &D,
                        double w, double h_F) const;
};
