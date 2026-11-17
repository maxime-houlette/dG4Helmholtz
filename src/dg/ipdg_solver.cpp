#include <sstream>
#include <stdexcept>
#include <cmath>
#include "ipdg_solver.h"
#include "local_matrices.h"
using namespace std;

IPDGSolver::IPDGSolver(const Mesh &input_mesh,
                       size_t degree,
                       double input_kappa,
                       BCType input_bc_type,
                       ScalarFunction input_f_rhs,
                       ScalarFunction input_u_exact,
                       VectorFunction input_grad_u_exact,
                       double input_gamma)
    : DGSolver(input_mesh, degree, 1, input_kappa, input_bc_type, input_f_rhs, input_u_exact, input_grad_u_exact),
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
/*                                 ELEMENT                                 */
/* ======================================================================= */

/* -kappa^2 (v, w)_K + (grad v, grad w)_K */
void IPDGSolver::assemble_element(size_t e)
{
    size_t n = n_local_dofs();
    const Element &K = mesh.elements[e];

    Eigen::MatrixXcd M, S;
    local_mass(*ref, K, M);
    local_stiffness(*ref, K, S);

    std::vector<size_t> dofs(n);
    for (size_t i = 0; i < n; i++)
        dofs[i] = local_to_global(e, U, i);

    scatter(dofs, -kappa * kappa * M + S);
}

/* ======================================================================= */
/*                                  FACES                                  */
/* ======================================================================= */

void IPDGSolver::add_face_terms(Eigen::MatrixXcd &A_loc, const Eigen::VectorXd &J, const Eigen::VectorXd &D,
                                double w, double h_F) const
{
    double pen = gamma / h_F;
    for (Eigen::Index a = 0; a < J.size(); a++)
        for (Eigen::Index c = 0; c < J.size(); c++)
            A_loc(a, c) += w * (pen * J(c) * J(a) - D(c) * J(a) - J(c) * D(a));
}

/* Local dofs: (side s, basis function i) -> s * n + i, with s = 0 for K_+ and s = 1 for K_- */
void IPDGSolver::assemble_interface(const Face &F)
{
    size_t n = n_local_dofs();
    const double sigma[2] = {1., -1.}; // [v] = v_+ - v_-

    std::vector<size_t> dofs(2 * n);
    for (size_t s = 0; s < 2; s++)
        for (size_t i = 0; i < n; i++)
            dofs[s * n + i] = local_to_global(F.elem[s], U, i);

    Eigen::MatrixXcd A_loc = Eigen::MatrixXcd::Zero(2 * n, 2 * n);
    Eigen::VectorXd J(2 * n), D(2 * n);

    for (const FaceQuadPoint &qp : face_quadrature(F))
    {
        for (size_t s = 0; s < 2; s++)
        {
            Eigen::VectorXd phi = basis_values(qp.xi[s]);
            Eigen::VectorXd dn_phi = basis_grads(F.elem[s], qp.xi[s]) * F.normal; // grad phi . n_F
            for (size_t i = 0; i < n; i++)
            {
                J(s * n + i) = sigma[s] * phi(i);
                D(s * n + i) = 0.5 * dn_phi(i);
            }
        }
        add_face_terms(A_loc, J, D, qp.w, F.h);
    }
    scatter(dofs, A_loc);
}

void IPDGSolver::assemble_boundary_face(const Face &F)
{
    size_t n = n_local_dofs();
    size_t e = F.elem[0];

    std::vector<size_t> dofs(n);
    for (size_t i = 0; i < n; i++)
        dofs[i] = local_to_global(e, U, i);

    Eigen::MatrixXcd A_loc = Eigen::MatrixXcd::Zero(n, n);

    for (const FaceQuadPoint &qp : face_quadrature(F))
    {
        Eigen::VectorXd phi = basis_values(qp.xi[0]);

        if (bc_type == Robin)
            /* + i kappa <v, w>_F */
            A_loc += (I * kappa * qp.w) * (phi * phi.transpose()).cast<complex<double>>();
        else
        {
            /* Dirichlet: [v] = {v} = v|_F, n_F outward */
            Eigen::VectorXd dn_phi = basis_grads(e, qp.xi[0]) * F.normal;
            add_face_terms(A_loc, phi, dn_phi, qp.w, F.h);
        }
    }
    scatter(dofs, A_loc);
}

/* ======================================================================= */
/*                                 ERRORS                                  */
/* ======================================================================= */

/*
 * Additional columns: gamma and the relative energy error |||u - u_h|||_dG / |||u|||_dG, with
 *   Dirichlet: |||v|||_dG^2 = kappa^2 ||v||^2 + ||grad_h v||^2 + sum_{F in F_h}   gamma/h_F ||[v]||_F^2
 *   Robin:     |||v|||_dG^2 = kappa^2 ||v||^2 + ||grad_h v||^2 + sum_{F in F_h^o} gamma/h_F ||[v]||_F^2 + kappa ||v||_{dOmega}^2
 * (-1 if the exact gradient is not provided). The exact solution has no jumps and vanishes on a Dirichlet boundary.
 */
std::string IPDGSolver::extra_error_header() const
{
    return "gamma,error_energy,";
}

std::string IPDGSolver::extra_error_columns(const ErrorNorms &norms) const
{
    double error_energy = -1.;
    if (grad_u_exact)
    {
        double jumps_sq = 0.;                    // sum_F gamma/h_F ||[u - u_h]||_F^2
        double err_bnd_sq = 0., sol_bnd_sq = 0.; // kappa ||.||_{dOmega}^2 (Robin)

        for (const Face &F : mesh.faces)
            for (const FaceQuadPoint &qp : face_quadrature(F))
            {
                complex<double> u_plus = eval_u(F.elem[0], qp.xi[0]);
                if (!F.boundary)
                    jumps_sq += gamma / F.h * qp.w * std::norm(u_plus - eval_u(F.elem[1], qp.xi[1]));
                else if (bc_type == Dirichlet)
                    jumps_sq += gamma / F.h * qp.w * std::norm(u_plus);
                else
                {
                    complex<double> u_ex = u_exact(qp.x);
                    err_bnd_sq += kappa * qp.w * std::norm(u_plus - u_ex);
                    sol_bnd_sq += kappa * qp.w * std::norm(u_ex);
                }
            }

        double k2 = kappa * kappa;
        error_energy = std::sqrt((k2 * norms.err_L2_sq + norms.err_H1_sq + jumps_sq + err_bnd_sq) /
                                 (k2 * norms.sol_L2_sq + norms.sol_H1_sq + sol_bnd_sq));
    }

    std::ostringstream os;
    os << gamma << "," << std::scientific << error_energy << ",";
    return os.str();
}
