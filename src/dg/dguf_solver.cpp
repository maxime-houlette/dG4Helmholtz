#include <iostream>
#include <cmath>
#include "dguf_solver.h"
#include "mass.h"
#include "half_stiffness.h"
using namespace std;

DGUFSolver::DGUFSolver(const Mesh &input_mesh,
                       size_t degree,
                       double input_kappa,
                       BCType input_bc_type,
                       RhsFunction input_f_rhs,
                       RhsFunction input_u_exact,
                       RhsFunction input_du_exact,
                       double input_tau,
                       bool input_robin_stab,
                       Problem input_problem)
    : DGSolver(input_mesh, degree, 2, input_kappa, input_bc_type, input_f_rhs, input_u_exact, input_du_exact),
      tau(input_tau),
      robin_stab(input_robin_stab),
      problem(input_problem)
{
    assemble();
}

std::string DGUFSolver::default_error_file() const
{
    if (bc_type == Dirichlet)
        return "error_dirichlet.csv";
    if (bc_type == Neumann)
        return "error_neumann.csv";
    return "error_robin.csv";
}

/* ======================================================================= */
/*                                 VOLUME                                  */
/* ======================================================================= */

void DGUFSolver::assemble_volume()
{
    if (problem == L2)
    {
        assemble_mass(1.);
        return;
    }
    assemble_mass(I * kappa);
    assemble_half_stiffness(-1.);
}

void DGUFSolver::assemble_mass(complex<double> alpha)
{
    size_t n_loc = n_local_dofs();

    Eigen::MatrixXcd M_loc(n_loc, n_loc);
    mass(ref_element, mesh.h, M_loc);

    for (size_t e = 0; e < mesh.N; e++)
        for (size_t i = 0; i < n_loc; i++)
            for (size_t j = 0; j < n_loc; j++)
            {
                size_t I_u = local_to_global(e, U, i);
                size_t J_u = local_to_global(e, U, j);
                size_t I_q = local_to_global(e, Q, i);
                size_t J_q = local_to_global(e, Q, j);

                /* Mass u,u */
                tripletList.push_back(TripletComplex(I_u, J_u, alpha * M_loc(i, j)));

                /* Mass q,q */
                tripletList.push_back(TripletComplex(I_q, J_q, alpha * M_loc(i, j)));
            }
}

void DGUFSolver::assemble_half_stiffness(double beta)
{
    size_t n_loc = n_local_dofs();

    Eigen::MatrixXcd hS_loc(n_loc, n_loc);
    half_stiffness(ref_element, hS_loc);

    for (size_t e = 0; e < mesh.N; e++)
        for (size_t i = 0; i < n_loc; i++)
            for (size_t j = 0; j < n_loc; j++)
            {
                size_t I_u = local_to_global(e, U, i);
                size_t J_u = local_to_global(e, U, j);
                size_t I_q = local_to_global(e, Q, i);
                size_t J_q = local_to_global(e, Q, j);

                /* Half stiffness u,q */
                tripletList.push_back(TripletComplex(I_u, J_q, beta * hS_loc(j, i)));

                /* Half stiffness q,u */
                tripletList.push_back(TripletComplex(I_q, J_u, beta * hS_loc(j, i)));
            }
}

/* ======================================================================= */
/*                               INTERFACES                                */
/* ======================================================================= */

void DGUFSolver::assemble_interfaces()
{
    if (problem == L2)
        return;

    size_t n_loc = n_local_dofs();

    for (size_t e = 0; e + 1 < mesh.N; e++)
    {
        size_t e_L = e;
        size_t e_R = e + 1;
        for (size_t i = 0; i < n_loc; i++)
            for (size_t j = 0; j < n_loc; j++)
            {
                /* Local to global numbering for the left element */
                size_t I_u_L = local_to_global(e_L, U, i);
                size_t J_u_L = local_to_global(e_L, U, j);
                size_t I_q_L = local_to_global(e_L, Q, i);
                size_t J_q_L = local_to_global(e_L, Q, j);

                /* Local to global numbering for the right element */
                size_t I_u_R = local_to_global(e_R, U, i);
                size_t J_u_R = local_to_global(e_R, U, j);
                size_t I_q_R = local_to_global(e_R, Q, i);
                size_t J_q_R = local_to_global(e_R, Q, j);

                /* Q test function of the left element */
                tripletList.push_back(TripletComplex(I_q_L, J_u_L, 1. * 0.5 * phi_R[j] * phi_R[i]));
                tripletList.push_back(TripletComplex(I_q_L, J_u_R, 1. * 0.5 * phi_L[j] * phi_R[i]));
                tripletList.push_back(TripletComplex(I_q_L, J_q_L, 1. / (2. * tau) * phi_R[j] * phi_R[i]));
                tripletList.push_back(TripletComplex(I_q_L, J_q_R, -1. / (2. * tau) * phi_L[j] * phi_R[i]));

                /* Q test function of the right element */
                tripletList.push_back(TripletComplex(I_q_R, J_u_L, -1. * 0.5 * phi_R[j] * phi_L[i]));
                tripletList.push_back(TripletComplex(I_q_R, J_u_R, -1. * 0.5 * phi_L[j] * phi_L[i]));
                tripletList.push_back(TripletComplex(I_q_R, J_q_L, -1. / (2. * tau) * phi_R[j] * phi_L[i]));
                tripletList.push_back(TripletComplex(I_q_R, J_q_R, 1. / (2. * tau) * phi_L[j] * phi_L[i]));

                /* U test function of the left element */
                tripletList.push_back(TripletComplex(I_u_L, J_q_L, 1. * 0.5 * phi_R[j] * phi_R[i]));
                tripletList.push_back(TripletComplex(I_u_L, J_q_R, 1. * 0.5 * phi_L[j] * phi_R[i]));
                tripletList.push_back(TripletComplex(I_u_L, J_u_L, tau / 2. * phi_R[j] * phi_R[i]));
                tripletList.push_back(TripletComplex(I_u_L, J_u_R, -tau / 2. * phi_L[j] * phi_R[i]));

                /* U test function of the right element */
                tripletList.push_back(TripletComplex(I_u_R, J_q_L, -1. * 0.5 * phi_R[j] * phi_L[i]));
                tripletList.push_back(TripletComplex(I_u_R, J_q_R, -1. * 0.5 * phi_L[j] * phi_L[i]));
                tripletList.push_back(TripletComplex(I_u_R, J_u_L, -tau / 2. * phi_R[j] * phi_L[i]));
                tripletList.push_back(TripletComplex(I_u_R, J_u_R, tau / 2. * phi_L[j] * phi_L[i]));
            }
    }
}

/* ======================================================================= */
/*                               BOUNDARIES                                */
/* ======================================================================= */

void DGUFSolver::assemble_boundaries()
{
    if (problem == L2)
        return;

    size_t n_loc = n_local_dofs();

    for (size_t i = 0; i < n_loc; i++)
        for (size_t j = 0; j < n_loc; j++)
        {
            /* Local to global for the boundary x=0 */
            size_t I_u_0 = local_to_global(0, U, i);
            size_t J_u_0 = local_to_global(0, U, j);
            size_t I_q_0 = local_to_global(0, Q, i);
            size_t J_q_0 = local_to_global(0, Q, j);

            /* Local to global for the boundary x=L */
            size_t I_u_N = local_to_global(mesh.N - 1, U, i);
            size_t J_u_N = local_to_global(mesh.N - 1, U, j);
            size_t I_q_N = local_to_global(mesh.N - 1, Q, i);
            size_t J_q_N = local_to_global(mesh.N - 1, Q, j);

            if (bc_type == Dirichlet)
            {
                /* Boundary x=0 */
                tripletList.push_back(TripletComplex(I_u_0, J_q_0, -1. * phi_L[j] * phi_L[i]));
                tripletList.push_back(TripletComplex(I_u_0, J_u_0, tau * phi_L[j] * phi_L[i]));

                /* Boundary x=L */
                tripletList.push_back(TripletComplex(I_u_N, J_q_N, 1. * phi_R[j] * phi_R[i]));
                tripletList.push_back(TripletComplex(I_u_N, J_u_N, tau * phi_R[j] * phi_R[i]));
            }
            else if (bc_type == Robin && !robin_stab)
            {
                /* Boundary x=0 (n = -1) */
                tripletList.push_back(TripletComplex(I_q_0, J_q_0, phi_L[i] * phi_L[j]));
                tripletList.push_back(TripletComplex(I_u_0, J_u_0, phi_L[i] * phi_L[j]));

                /* Boundary x=L (n = 1) */
                tripletList.push_back(TripletComplex(I_q_N, J_q_N, phi_R[i] * phi_R[j]));
                tripletList.push_back(TripletComplex(I_u_N, J_u_N, phi_R[i] * phi_R[j]));
            }
            else if (bc_type == Robin && robin_stab)
            {
                /* Boundary x=0 (n = -1) */
                tripletList.push_back(TripletComplex(I_q_0, J_q_0, 1. / (1 + tau) * phi_L[i] * phi_L[j]));
                tripletList.push_back(TripletComplex(I_q_0, J_u_0, -tau / (tau + 1) * phi_L[i] * phi_L[j]));
                tripletList.push_back(TripletComplex(I_u_0, J_q_0, -1. / (1 + tau) * phi_L[i] * phi_L[j]));
                tripletList.push_back(TripletComplex(I_u_0, J_u_0, tau / (1 + tau) * phi_L[i] * phi_L[j]));

                /* Boundary x=L (n = 1) */
                tripletList.push_back(TripletComplex(I_q_N, J_q_N, 1. / (1 + tau) * phi_R[i] * phi_R[j]));
                tripletList.push_back(TripletComplex(I_q_N, J_u_N, tau / (1 + tau) * phi_R[i] * phi_R[j]));
                tripletList.push_back(TripletComplex(I_u_N, J_q_N, 1. / (1 + tau) * phi_R[i] * phi_R[j]));
                tripletList.push_back(TripletComplex(I_u_N, J_u_N, tau / (1 + tau) * phi_R[i] * phi_R[j]));
            }
            else if (bc_type == Neumann)
            {
                /* Boundary x=0 */
                tripletList.push_back(TripletComplex(I_q_0, J_q_0, 1. / tau * phi_L[j] * phi_L[i]));
                tripletList.push_back(TripletComplex(I_q_0, J_u_0, -phi_L[j] * phi_L[i]));

                /* Boundary x=L */
                tripletList.push_back(TripletComplex(I_q_N, J_q_N, 1. / tau * phi_R[j] * phi_R[i]));
                tripletList.push_back(TripletComplex(I_q_N, J_u_N, phi_R[j] * phi_R[i]));
            }
        }
}
