#include <cmath>
#include "dguf_solver.h"
#include "local_matrices.h"
using namespace std;

DGUFSolver::DGUFSolver(const Mesh &input_mesh,
                       size_t degree,
                       double input_kappa,
                       BCType input_bc_type,
                       ScalarFunction input_f_rhs,
                       ScalarFunction input_u_exact,
                       VectorFunction input_grad_u_exact,
                       double input_tau,
                       bool input_robin_stab,
                       Problem input_problem)
    : DGSolver(input_mesh, degree, 1 + input_mesh.dim, input_kappa, input_bc_type,
               input_f_rhs, input_u_exact, input_grad_u_exact),
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
/*                                 ELEMENT                                 */
/* ======================================================================= */

/* i kappa (u, r)_K + i kappa (q, s)_K - (q, grad r)_K - (u, div s)_K   (only the mass if problem == L2) */
void DGUFSolver::assemble_element(size_t e)
{
    size_t n = n_local_dofs();
    const Element &K = mesh.elements[e];

    /* local index of (field f, basis function i) */
    auto idx = [n](size_t f, size_t i)
    { return f * n + i; };

    std::vector<size_t> dofs(n_fields * n);
    for (size_t f = 0; f < n_fields; f++)
        for (size_t i = 0; i < n; i++)
            dofs[idx(f, i)] = local_to_global(e, f, i);

    Eigen::MatrixXcd A_loc = Eigen::MatrixXcd::Zero(n_fields * n, n_fields * n);

    /* Mass on every field */
    Eigen::MatrixXcd M;
    local_mass(*ref, K, M);
    complex<double> alpha = (problem == L2) ? complex<double>(1.) : I * kappa;
    for (size_t f = 0; f < n_fields; f++)
        A_loc.block(f * n, f * n, n, n) = alpha * M;

    if (problem == Helmholtz)
    {
        /* -(q_k, d_k r)_K on the u rows and -(u, d_k s_k)_K on the q_k rows, H_k(i,j) = (phi_i, d_k phi_j)_K */
        Eigen::MatrixXcd H;
        for (size_t k = 0; k < dim(); k++)
        {
            local_half_stiffness(*ref, K, k, H);
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < n; j++)
                {
                    A_loc(idx(U, i), idx(Q + k, j)) -= H(j, i);
                    A_loc(idx(Q + k, i), idx(U, j)) -= H(j, i);
                }
        }
    }
    scatter(dofs, A_loc);
}

/* ======================================================================= */
/*                                INTERFACE                                */
/* ======================================================================= */

void DGUFSolver::assemble_interface(const Face &F)
{
    if (problem == L2)
        return;

    size_t n = n_local_dofs();
    const double sigma[2] = {1., -1.}; // n_K = sigma_s n_F on side s (K_+, K_-)

    /* local index of (side s, field f, basis function i) */
    auto idx = [n, this](size_t s, size_t f, size_t i)
    { return (s * n_fields + f) * n + i; };

    std::vector<size_t> dofs(2 * n_fields * n);
    for (size_t s = 0; s < 2; s++)
        for (size_t f = 0; f < n_fields; f++)
            for (size_t i = 0; i < n; i++)
                dofs[idx(s, f, i)] = local_to_global(F.elem[s], f, i);

    Eigen::MatrixXcd A_loc = Eigen::MatrixXcd::Zero(2 * n_fields * n, 2 * n_fields * n);
    const Point &nF = F.normal;

    for (const FaceQuadPoint &qp : face_quadrature(F))
    {
        Eigen::VectorXd phi[2] = {basis_values(qp.xi[0]), basis_values(qp.xi[1])};

        for (size_t s = 0; s < 2; s++)     // test side
            for (size_t t = 0; t < 2; t++) // trial side
                for (size_t i = 0; i < n; i++)
                    for (size_t j = 0; j < n; j++)
                    {
                        double pp = qp.w * phi[s](i) * phi[t](j);
                        double ss = sigma[s], st = sigma[t];

                        /* u rows: sigma_s ( {q} . n_F + tau/2 [u] ) */
                        A_loc(idx(s, U, i), idx(t, U, j)) += ss * st * tau / 2. * pp;
                        for (size_t m = 0; m < dim(); m++)
                            A_loc(idx(s, U, i), idx(t, Q + m, j)) += ss * 0.5 * nF(m) * pp;

                        /* q_k rows: sigma_s n_{F,k} ( {u} + 1/(2 tau) [q] . n_F ) */
                        for (size_t k = 0; k < dim(); k++)
                        {
                            A_loc(idx(s, Q + k, i), idx(t, U, j)) += ss * nF(k) * 0.5 * pp;
                            for (size_t m = 0; m < dim(); m++)
                                A_loc(idx(s, Q + k, i), idx(t, Q + m, j)) += ss * st * nF(k) * nF(m) / (2. * tau) * pp;
                        }
                    }
    }
    scatter(dofs, A_loc);
}

/* ======================================================================= */
/*                                BOUNDARY                                 */
/* ======================================================================= */

void DGUFSolver::assemble_boundary_face(const Face &F)
{
    if (problem == L2)
        return;

    size_t n = n_local_dofs();
    size_t e = F.elem[0];

    auto idx = [n](size_t f, size_t i)
    { return f * n + i; };

    std::vector<size_t> dofs(n_fields * n);
    for (size_t f = 0; f < n_fields; f++)
        for (size_t i = 0; i < n; i++)
            dofs[idx(f, i)] = local_to_global(e, f, i);

    Eigen::MatrixXcd A_loc = Eigen::MatrixXcd::Zero(n_fields * n, n_fields * n);
    const Point &nv = F.normal; // outward normal

    for (const FaceQuadPoint &qp : face_quadrature(F))
    {
        Eigen::VectorXd phi = basis_values(qp.xi[0]);

        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
            {
                double pp = qp.w * phi(i) * phi(j);

                if (bc_type == Dirichlet)
                {
                    /* u rows: q . n + tau u */
                    for (size_t m = 0; m < dim(); m++)
                        A_loc(idx(U, i), idx(Q + m, j)) += nv(m) * pp;
                    A_loc(idx(U, i), idx(U, j)) += tau * pp;
                }
                else if (bc_type == Robin && !robin_stab)
                {
                    /* u rows: u ; q_k rows: (q . n) n_k */
                    A_loc(idx(U, i), idx(U, j)) += pp;
                    for (size_t k = 0; k < dim(); k++)
                        for (size_t m = 0; m < dim(); m++)
                            A_loc(idx(Q + k, i), idx(Q + m, j)) += nv(k) * nv(m) * pp;
                }
                else if (bc_type == Robin && robin_stab)
                {
                    /* u rows: (q . n + tau u) / (1 + tau) ; q_k rows: n_k (q . n + tau u) / (1 + tau) */
                    A_loc(idx(U, i), idx(U, j)) += tau / (1. + tau) * pp;
                    for (size_t m = 0; m < dim(); m++)
                        A_loc(idx(U, i), idx(Q + m, j)) += nv(m) / (1. + tau) * pp;
                    for (size_t k = 0; k < dim(); k++)
                    {
                        A_loc(idx(Q + k, i), idx(U, j)) += nv(k) * tau / (1. + tau) * pp;
                        for (size_t m = 0; m < dim(); m++)
                            A_loc(idx(Q + k, i), idx(Q + m, j)) += nv(k) * nv(m) / (1. + tau) * pp;
                    }
                }
                else if (bc_type == Neumann)
                {
                    /* q_k rows: n_k (q . n / tau + u) */
                    for (size_t k = 0; k < dim(); k++)
                    {
                        A_loc(idx(Q + k, i), idx(U, j)) += nv(k) * pp;
                        for (size_t m = 0; m < dim(); m++)
                            A_loc(idx(Q + k, i), idx(Q + m, j)) += nv(k) * nv(m) / tau * pp;
                    }
                }
            }
    }
    scatter(dofs, A_loc);
}
