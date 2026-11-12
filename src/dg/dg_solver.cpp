#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include "dg_solver.h"
#include "quadrature.h"
using namespace std;

DGSolver::DGSolver(const Mesh &input_mesh,
                   size_t degree,
                   size_t input_n_fields,
                   double input_kappa,
                   BCType input_bc_type,
                   RhsFunction input_f_rhs,
                   RhsFunction input_u_exact,
                   RhsFunction input_du_exact)
    : mesh(input_mesh),
      ref_element(degree),
      n_fields(input_n_fields),
      kappa(input_kappa),
      bc_type(input_bc_type),
      f_rhs(input_f_rhs),
      u_exact(input_u_exact),
      du_exact(input_du_exact)
{
    /* ---------- MEMORY ALLOCATION ---------- */
    size_t n_loc = n_local_dofs();
    size_t total_dofs = n_fields * mesh.N * n_loc;

    A.resize(total_dofs, total_dofs);
    u.resize(total_dofs);
    b = Eigen::VectorXcd::Zero(total_dofs);

    /* volume block + 4 interface blocks per element and per pair of fields */
    tripletList.reserve(5 * n_fields * n_fields * mesh.N * n_loc * n_loc);

    /* ---------- TRACES OF THE REFERENCE BASIS ---------- */
    phi_L.resize(n_loc);
    phi_R.resize(n_loc);
    dphi_L.resize(n_loc);
    dphi_R.resize(n_loc);
    for (size_t i = 0; i < n_loc; i++)
    {
        phi_L[i] = ref_element.eval_basis(i, -1.0);
        phi_R[i] = ref_element.eval_basis(i, 1.0);
        dphi_L[i] = ref_element.eval_basis_deriv(i, -1.0);
        dphi_R[i] = ref_element.eval_basis_deriv(i, 1.0);
    }
}

/* ======================================================================= */
/*                                ASSEMBLY                                 */
/* ======================================================================= */

void DGSolver::assemble()
{
    assemble_volume();
    assemble_interfaces();
    assemble_boundaries();
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    tripletList.clear();

    assemble_rhs();
}

void DGSolver::assemble_rhs()
{
    size_t n_loc = n_local_dofs();
    double J = mesh.h / 2;
    QuadratureRule quad_rule = get_gauss_legendre(n_loc);

    for (size_t e = 0; e < mesh.N; e++)
    {
        double x_c = 0.5 * (mesh.positions[e] + mesh.positions[e + 1]);

        for (size_t q = 0; q < n_loc; q++)
        {
            double xi_q = quad_rule.points[q];
            double w_q = quad_rule.weights[q];
            complex<double> f_val = f_rhs(x_c + J * xi_q);

            for (size_t i = 0; i < n_loc; i++)
                b(local_to_global(e, U, i)) += J * w_q * f_val * ref_element.eval_basis(i, xi_q);
        }
    }
}

/* ======================================================================= */
/*                                 HELPERS                                 */
/* ======================================================================= */

size_t DGSolver::local_to_global(size_t element, Variable var, size_t local_dof) const
{
    return (n_fields * element + var) * n_local_dofs() + local_dof;
}

complex<double> DGSolver::eval_u(size_t element, double xi) const
{
    complex<double> val(0., 0.);
    for (size_t i = 0; i < n_local_dofs(); i++)
        val += u(local_to_global(element, U, i)) * ref_element.eval_basis(i, xi);
    return val;
}

complex<double> DGSolver::eval_du(size_t element, double xi) const
{
    complex<double> val(0., 0.);
    for (size_t i = 0; i < n_local_dofs(); i++)
        val += u(local_to_global(element, U, i)) * ref_element.eval_basis_deriv(i, xi);
    return 2. / mesh.h * val;
}

/* ======================================================================= */
/*                                  SOLVE                                  */
/* ======================================================================= */

void DGSolver::solve()
{
    std::cout << "LU factorization in progress..." << std::endl;
    Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>> linear_solver;
    linear_solver.compute(A);

    if (linear_solver.info() != Eigen::Success)
    {
        std::cerr << "FATAL ERROR: LU factorization has failed." << std::endl;
        return;
    }

    std::cout << "Resolution of the linear system..." << std::endl;
    u = linear_solver.solve(b);

    if (linear_solver.info() != Eigen::Success)
    {
        std::cerr << "FATAL ERROR: Linear resolution has failed." << std::endl;
        return;
    }
    std::cout << "Success!" << std::endl;
}

/* ======================================================================= */
/*                             POST-PROCESSING                             */
/* ======================================================================= */

void DGSolver::post_process_u()
{
    std::cout << "Exportation of the solution in solution_u.csv..." << std::endl;
    std::ofstream out("solution_u.csv");
    out << "x,u_real,u_imag,rhs_real,rhs_imag\n";

    size_t n_plot_points = 5; // per element
    for (size_t e = 0; e < mesh.N; e++)
    {
        double x_L = mesh.positions[e];
        double x_R = mesh.positions[e + 1];

        for (size_t p = 0; p <= n_plot_points; p++)
        {
            double xi = -1.0 + 2.0 * (double)p / n_plot_points;
            double x_phys = x_L + (xi + 1.0) * (x_R - x_L) / 2.0;

            complex<double> u_val = eval_u(e, xi);
            complex<double> rhs_val = f_rhs(x_phys);

            out << x_phys << ","
                << u_val.real() << "," << u_val.imag() << ","
                << rhs_val.real() << "," << rhs_val.imag() << "\n";
        }
    }
    std::cout << "Success!" << std::endl;
}

void DGSolver::post_process_u_exact()
{
    std::cout << "Exportation of the exact solution in solution_exact.csv..." << std::endl;
    std::ofstream out("solution_exact.csv");
    out << "x,u_real,u_imag,rhs_real,rhs_imag\n";

    size_t n_plot_points = 5; // per element
    for (size_t e = 0; e < mesh.N; e++)
    {
        double x_L = mesh.positions[e];
        double x_R = mesh.positions[e + 1];

        for (size_t p = 0; p <= n_plot_points; p++)
        {
            double xi = -1.0 + 2.0 * (double)p / n_plot_points;
            double x_phys = x_L + (xi + 1.0) * (x_R - x_L) / 2.0;

            complex<double> u_val = u_exact(x_phys);
            complex<double> rhs_val = f_rhs(x_phys);

            out << x_phys << ","
                << u_val.real() << "," << u_val.imag() << ","
                << rhs_val.real() << "," << rhs_val.imag() << "\n";
        }
    }
    std::cout << "Success!" << std::endl;
}

void DGSolver::plot_u()
{
    std::cout << "GNUPlot launching..." << std::endl;
    FILE *gnuplotPipe = popen("gnuplot -persistent", "w");

    if (gnuplotPipe)
    {
        fprintf(gnuplotPipe, "set title 'Solution de l\\'équation de Helmholtz (dG 1D)'\n");
        fprintf(gnuplotPipe, "set xlabel 'Position x'\n");
        fprintf(gnuplotPipe, "set ylabel 'Amplitude'\n");
        fprintf(gnuplotPipe, "set grid\n");
        fprintf(gnuplotPipe, "set datafile separator ','\n");

        fprintf(gnuplotPipe, "plot 'solution_u.csv' every ::1 using 1:2 with linespoints pt 7 ps 0.5 lc rgb 'blue' title 'Re(u)', ");
        fprintf(gnuplotPipe, "'' every ::1 using 1:3 with linespoints pt 7 ps 0.5 lc rgb 'orange' title 'Im(u)'\n");

        pclose(gnuplotPipe);
    }
    else
        std::cerr << "FATAL ERROR: unable to plot." << std::endl;
}

void DGSolver::plot_error()
{
    std::cout << "GNUPlot launching..." << std::endl;
    FILE *gnuplotPipe = popen("gnuplot -persistent", "w");

    if (gnuplotPipe)
    {
        fprintf(gnuplotPipe, "set title 'Erreur de l equation de Helmholtz (dG 1D)'\n");
        fprintf(gnuplotPipe, "set xlabel 'Position x'\n");
        fprintf(gnuplotPipe, "set ylabel 'Error'\n");
        fprintf(gnuplotPipe, "set grid\n");
        fprintf(gnuplotPipe, "set datafile separator ','\n");

        fprintf(gnuplotPipe, "plot 'error_helmholtz.csv' every ::1 using 1:4 with linespoints pt 7 ps 0.5 lc rgb 'blue' title 'Re(u)', ");

        pclose(gnuplotPipe);
    }
    else
        std::cerr << "FATAL ERROR: unable to plot." << std::endl;
}

/* ======================================================================= */
/*                                 ERRORS                                  */
/* ======================================================================= */

DGSolver::ErrorNorms DGSolver::integrate_errors() const
{
    ErrorNorms norms;

    double J = mesh.h / 2.0;
    QuadratureRule quad_rule = get_gauss_legendre(n_local_dofs() + 1);

    for (size_t e = 0; e < mesh.N; e++)
    {
        double x_L = mesh.positions[e];

        for (size_t q = 0; q < quad_rule.points.size(); q++)
        {
            double xi = quad_rule.points[q];
            double w_q = quad_rule.weights[q];
            double x_phys = x_L + (xi + 1.0) * J;

            complex<double> u_ex = u_exact(x_phys);
            norms.err_L2_sq += J * w_q * std::norm(eval_u(e, xi) - u_ex);
            norms.sol_L2_sq += J * w_q * std::norm(u_ex);

            if (du_exact)
            {
                complex<double> du_ex = du_exact(x_phys);
                norms.err_H1_sq += J * w_q * std::norm(eval_du(e, xi) - du_ex);
                norms.sol_H1_sq += J * w_q * std::norm(du_ex);
            }
        }
    }
    return norms;
}

void DGSolver::compute_errors(const std::string &filename)
{
    if (!u_exact)
    {
        std::cerr << "FATAL ERROR: no exact solution has been provided." << std::endl;
        return;
    }
    std::cout << "Error computation..." << std::endl;

    ErrorNorms norms = integrate_errors();
    double error_L2 = std::sqrt(norms.err_L2_sq / norms.sol_L2_sq);

    std::string file = filename.empty() ? default_error_file() : filename;
    std::ifstream check_file(file);
    bool file_exists = check_file.good();
    check_file.close();

    std::ofstream out(file, std::ios::app);
    if (!file_exists)
        out << "N,degree,kappa/Pi," << extra_error_header() << "error_L2\n";

    out << mesh.N << ","
        << ref_element.degree << ","
        << kappa / M_PI << ","
        << extra_error_columns(norms)
        << std::scientific << error_L2 << "\n";
}
