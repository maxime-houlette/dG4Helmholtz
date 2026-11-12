#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include "ipdg_solver.h"
#include "mass.h"
#include "stiffness.h"
using namespace std;

IPDGSolver::IPDGSolver(const Mesh &input_mesh,
                       size_t degree,
                       double input_kappa,
                       BCType input_bc_type,
                       RhsFunction input_f_rhs,
                       RhsFunction input_u_exact,
                       RhsFunction input_du_exact,
                       double input_gamma)
    : DGSolver(input_mesh, degree, 1, input_kappa, input_bc_type, input_f_rhs, input_u_exact, input_du_exact),
      gamma(input_gamma)
{
    if (bc_type == Neumann)
        throw std::invalid_argument("IPDGSolver: Neumann boundary conditions are not implemented.");

    assemble();
}

std::string IPDGSolver::default_error_file() const
{
    return bc_type == Robin ? "error_ipdg_robin.csv" : "error_ipdg.csv";
}

/* ======================================================================= */
/*                                 VOLUME                                  */
/* ======================================================================= */

/* -kappa^2 (v, w)_K + (v', w')_K */
void IPDGSolver::assemble_volume()
{
    size_t n_loc = n_local_dofs();

    Eigen::MatrixXcd M_loc(n_loc, n_loc);
    Eigen::MatrixXcd S_loc(n_loc, n_loc);
    mass(ref_element, mesh.h, M_loc);
    stiffness(ref_element, mesh.h, S_loc);

    Eigen::MatrixXcd K_loc = -kappa * kappa * M_loc + S_loc;

    for (size_t e = 0; e < mesh.N; e++)
        for (size_t i = 0; i < n_loc; i++)
            for (size_t j = 0; j < n_loc; j++)
                tripletList.push_back(TripletComplex(local_to_global(e, U, i), local_to_global(e, U, j), K_loc(i, j)));
}

/* ======================================================================= */
/*                                  FACES                                  */
/* ======================================================================= */

/*
 * For a face F with normal n_F, `dofs` are the dofs touching F, with
 *   J[k] = contribution of dof k to [v] on F,     D[k] = contribution of dof k to {v'} on F.
 * Adds, for trial dof c and test dof a (A(a, c) = b_h(phi_c, phi_a)),
 *   gamma/h J[c] J[a] - n_F D[c] J[a] - n_F J[c] D[a].
 */
void IPDGSolver::add_face_terms(const vector<size_t> &dofs, const vector<double> &J, const vector<double> &D, double n_F)
{
    double pen = gamma / mesh.h;
    for (size_t a = 0; a < dofs.size(); a++)
        for (size_t c = 0; c < dofs.size(); c++)
        {
            double val = pen * J[c] * J[a] - n_F * D[c] * J[a] - n_F * J[c] * D[a];
            if (val != 0.)
                tripletList.push_back(TripletComplex(dofs[a], dofs[c], val));
        }
}

/* Interfaces x_{e+1}, between K_- = K_e (left) and K_+ = K_{e+1} (right), n_F = +1 */
void IPDGSolver::assemble_interfaces()
{
    size_t n_loc = n_local_dofs();
    double dxi_dx = 2. / mesh.h;

    vector<size_t> dofs(2 * n_loc);
    vector<double> J(2 * n_loc), D(2 * n_loc);

    for (size_t e = 0; e + 1 < mesh.N; e++)
    {
        for (size_t i = 0; i < n_loc; i++)
        {
            dofs[i] = local_to_global(e, U, i);
            J[i] = phi_R[i];
            D[i] = 0.5 * (dxi_dx * dphi_R[i]);

            dofs[n_loc + i] = local_to_global(e + 1, U, i);
            J[n_loc + i] = -phi_L[i];
            D[n_loc + i] = 0.5 * (dxi_dx * dphi_L[i]);
        }
        add_face_terms(dofs, J, D, 1.);
    }
}

/* ======================================================================= */
/*                               BOUNDARIES                                */
/* ======================================================================= */

void IPDGSolver::assemble_boundaries()
{
    size_t n_loc = n_local_dofs();
    double dxi_dx = 2. / mesh.h;

    if (bc_type == Robin)
    {
        /* + i kappa <v, w>_{dOmega}, at x = 0 (element 0, xi = -1) and x = L (element N-1, xi = +1) */
        complex<double> ik = I * kappa;
        for (size_t i = 0; i < n_loc; i++)
            for (size_t j = 0; j < n_loc; j++)
            {
                tripletList.push_back(TripletComplex(local_to_global(0, U, i), local_to_global(0, U, j), ik * phi_L[j] * phi_L[i]));
                tripletList.push_back(TripletComplex(local_to_global(mesh.N - 1, U, i), local_to_global(mesh.N - 1, U, j), ik * phi_R[j] * phi_R[i]));
            }
        return;
    }

    /* Dirichlet: boundary faces, [v] = {v} = v|_F, outward normal */
    vector<size_t> dofs(n_loc);
    vector<double> J(n_loc), D(n_loc);

    /* x = 0, n_F = -1 */
    for (size_t i = 0; i < n_loc; i++)
    {
        dofs[i] = local_to_global(0, U, i);
        J[i] = phi_L[i];
        D[i] = dxi_dx * dphi_L[i];
    }
    add_face_terms(dofs, J, D, -1.);

    /* x = L, n_F = +1 */
    for (size_t i = 0; i < n_loc; i++)
    {
        dofs[i] = local_to_global(mesh.N - 1, U, i);
        J[i] = phi_R[i];
        D[i] = dxi_dx * dphi_R[i];
    }
    add_face_terms(dofs, J, D, 1.);
}

/* ======================================================================= */
/*                                 ERRORS                                  */
/* ======================================================================= */

/*
 * Additional columns: gamma and the relative energy error |||u - u_h|||_dG / |||u|||_dG, with
 *   Dirichlet: |||v|||_dG^2 = kappa^2 ||v||^2 + ||v'||_{T_h}^2 + gamma/h ||[v]||_{F_h}^2  (jumps on all faces)
 *   Robin:     |||v|||_dG^2 = kappa^2 ||v||^2 + ||v'||_{T_h}^2 + gamma/h ||[v]||_{F_h^o}^2 + kappa ||v||_{dOmega}^2
 * (-1 if the exact derivative is not provided).
 */
std::string IPDGSolver::extra_error_header() const
{
    return "gamma,error_energy,";
}

std::string IPDGSolver::extra_error_columns(const ErrorNorms &norms) const
{
    double error_energy = -1.;
    if (du_exact)
    {
        /* interior jumps of u_h (the exact solution has no jumps) */
        double jumps_sq = 0.;
        for (size_t e = 0; e + 1 < mesh.N; e++)
            jumps_sq += std::norm(eval_u(e, 1.) - eval_u(e + 1, -1.));

        double err_bnd_sq = 0., sol_bnd_sq = 0.;
        if (bc_type == Dirichlet)
            /* boundary jumps = traces of u_h (u = 0 on the boundary) */
            jumps_sq += std::norm(eval_u(0, -1.)) + std::norm(eval_u(mesh.N - 1, 1.));
        else
        {
            /* kappa ||.||_{dOmega}^2 */
            complex<double> u0 = u_exact(0.), uL = u_exact(mesh.L);
            err_bnd_sq = kappa * (std::norm(eval_u(0, -1.) - u0) + std::norm(eval_u(mesh.N - 1, 1.) - uL));
            sol_bnd_sq = kappa * (std::norm(u0) + std::norm(uL));
        }

        double k2 = kappa * kappa;
        error_energy = std::sqrt((k2 * norms.err_L2_sq + norms.err_H1_sq + gamma / mesh.h * jumps_sq + err_bnd_sq) /
                                 (k2 * norms.sol_L2_sq + norms.sol_H1_sq + sol_bnd_sq));
    }

    std::ostringstream os;
    os << gamma << "," << std::scientific << error_energy << ",";
    return os.str();
}
