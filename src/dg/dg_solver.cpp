#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include "dg_solver.h"
#include "quadrature.h"
using namespace std;

static std::unique_ptr<ReferenceElement> make_reference_element(size_t dim, size_t degree)
{
    if (dim == 1)
        return std::make_unique<ReferenceInterval>(degree);
    throw std::invalid_argument("DGSolver: no reference element available in dimension " + std::to_string(dim));
}

DGSolver::DGSolver(const Mesh &input_mesh,
                   size_t degree,
                   size_t input_n_fields,
                   double input_kappa,
                   BCType input_bc_type,
                   ScalarFunction input_f_rhs,
                   ScalarFunction input_u_exact,
                   VectorFunction input_grad_u_exact)
    : mesh(input_mesh),
      ref(make_reference_element(input_mesh.dim, degree)),
      n_fields(input_n_fields),
      kappa(input_kappa),
      bc_type(input_bc_type),
      f_rhs(input_f_rhs),
      u_exact(input_u_exact),
      grad_u_exact(input_grad_u_exact)
{
    size_t n_loc = n_local_dofs();
    size_t total_dofs = n_fields * mesh.n_elements() * n_loc;

    A.resize(total_dofs, total_dofs);
    u.resize(total_dofs);
    b = Eigen::VectorXcd::Zero(total_dofs);

    /* one volume block per element + one (2 x 2 blocks) per face */
    size_t block = n_fields * n_loc;
    tripletList.reserve(mesh.n_elements() * block * block + 4 * mesh.faces.size() * block * block);
}

/* ======================================================================= */
/*                                ASSEMBLY                                 */
/* ======================================================================= */

void DGSolver::assemble()
{
    for (size_t e = 0; e < mesh.n_elements(); e++)
        assemble_element(e);

    for (const Face &F : mesh.faces)
    {
        if (F.boundary)
            assemble_boundary_face(F);
        else
            assemble_interface(F);
    }

    A.setFromTriplets(tripletList.begin(), tripletList.end());
    tripletList.clear();

    assemble_rhs();
}

void DGSolver::scatter(const std::vector<size_t> &dofs, const Eigen::MatrixXcd &A_loc)
{
    for (size_t a = 0; a < dofs.size(); a++)
        for (size_t c = 0; c < dofs.size(); c++)
            if (A_loc(a, c) != 0.)
                tripletList.push_back(TripletComplex(dofs[a], dofs[c], A_loc(a, c)));
}

void DGSolver::assemble_rhs()
{
    QuadratureRule quad = ref->quadrature(2 * ref->degree + 1);

    for (size_t e = 0; e < mesh.n_elements(); e++)
    {
        const Element &K = mesh.elements[e];
        for (size_t q = 0; q < quad.size(); q++)
        {
            complex<double> f_val = f_rhs(K.map(quad.points[q]));
            Eigen::VectorXd phi = basis_values(quad.points[q]);
            double w = K.det_J * quad.weights[q];

            for (size_t i = 0; i < n_local_dofs(); i++)
                b(local_to_global(e, U, i)) += w * f_val * phi(i);
        }
    }
}

/* ======================================================================= */
/*                          DOFS AND BASIS FUNCTIONS                       */
/* ======================================================================= */

size_t DGSolver::local_to_global(size_t element, size_t field, size_t local_dof) const
{
    return (n_fields * element + field) * n_local_dofs() + local_dof;
}

Eigen::VectorXd DGSolver::basis_values(const Point &xi) const
{
    Eigen::VectorXd phi(n_local_dofs());
    for (size_t i = 0; i < n_local_dofs(); i++)
        phi(i) = ref->eval_basis(i, xi);
    return phi;
}

Eigen::MatrixXd DGSolver::basis_grads(size_t element, const Point &xi) const
{
    const Element &K = mesh.elements[element];
    Eigen::MatrixXd grads(n_local_dofs(), dim());
    for (size_t i = 0; i < n_local_dofs(); i++)
        grads.row(i) = (K.J_inv_T * ref->eval_grad(i, xi)).transpose();
    return grads;
}

complex<double> DGSolver::eval_u(size_t element, const Point &xi) const
{
    complex<double> val(0., 0.);
    for (size_t i = 0; i < n_local_dofs(); i++)
        val += u(local_to_global(element, U, i)) * ref->eval_basis(i, xi);
    return val;
}

CPoint DGSolver::eval_grad_u(size_t element, const Point &xi) const
{
    Eigen::MatrixXd grads = basis_grads(element, xi);
    CPoint val = CPoint::Zero(dim());
    for (size_t i = 0; i < n_local_dofs(); i++)
        val += u(local_to_global(element, U, i)) * grads.row(i).transpose().cast<complex<double>>();
    return val;
}

/* ======================================================================= */
/*                             FACE QUADRATURE                             */
/* ======================================================================= */

std::vector<DGSolver::FaceQuadPoint> DGSolver::face_quadrature(const Face &F) const
{
    QuadratureRule quad = ref->face_quadrature(2 * ref->degree);
    std::vector<FaceQuadPoint> pts(quad.size());

    for (size_t q = 0; q < quad.size(); q++)
    {
        const Point &eta = quad.points[q];
        pts[q].xi[0] = ref->face_to_element(F.local_face[0], eta);
        pts[q].xi[1] = F.boundary ? pts[q].xi[0]
                                  : ref->face_to_element(F.local_face[1], F.flip ? Point(-eta) : eta);
        pts[q].x = mesh.elements[F.elem[0]].map(pts[q].xi[0]);
        pts[q].w = F.jac * quad.weights[q];
    }
    return pts;
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

/* Writes "x[,y[,z]],u_real,u_imag,rhs_real,rhs_imag" at sample points of every element */
static void write_samples(const std::string &filename, const Mesh &mesh, const ReferenceElement &ref,
                          const std::function<complex<double>(size_t, const Point &, const Point &)> &value,
                          const ScalarFunction &rhs)
{
    const char *coords[3] = {"x", "y", "z"};
    std::ofstream out(filename);
    for (size_t k = 0; k < mesh.dim; k++)
        out << coords[k] << ",";
    out << "u_real,u_imag,rhs_real,rhs_imag\n";

    std::vector<Point> samples = ref.sample_points(5);
    for (size_t e = 0; e < mesh.n_elements(); e++)
        for (const Point &xi : samples)
        {
            Point x = mesh.elements[e].map(xi);
            complex<double> val = value(e, xi, x);
            complex<double> rhs_val = rhs(x);
            for (size_t k = 0; k < mesh.dim; k++)
                out << x(k) << ",";
            out << val.real() << "," << val.imag() << "," << rhs_val.real() << "," << rhs_val.imag() << "\n";
        }
}

void DGSolver::post_process_u()
{
    std::cout << "Exportation of the solution in solution_u.csv..." << std::endl;
    write_samples("solution_u.csv", mesh, *ref,
                  [this](size_t e, const Point &xi, const Point &)
                  { return eval_u(e, xi); },
                  f_rhs);
    std::cout << "Success!" << std::endl;
}

void DGSolver::post_process_u_exact()
{
    std::cout << "Exportation of the exact solution in solution_exact.csv..." << std::endl;
    write_samples("solution_exact.csv", mesh, *ref,
                  [this](size_t, const Point &, const Point &x)
                  { return u_exact(x); },
                  f_rhs);
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
    QuadratureRule quad = ref->quadrature(2 * ref->degree + 3);

    for (size_t e = 0; e < mesh.n_elements(); e++)
    {
        const Element &K = mesh.elements[e];
        for (size_t q = 0; q < quad.size(); q++)
        {
            const Point &xi = quad.points[q];
            Point x = K.map(xi);
            double w = K.det_J * quad.weights[q];

            complex<double> u_ex = u_exact(x);
            norms.err_L2_sq += w * std::norm(eval_u(e, xi) - u_ex);
            norms.sol_L2_sq += w * std::norm(u_ex);

            if (grad_u_exact)
            {
                CPoint g_ex = grad_u_exact(x);
                norms.err_H1_sq += w * (eval_grad_u(e, xi) - g_ex).squaredNorm();
                norms.sol_H1_sq += w * g_ex.squaredNorm();
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

    out << mesh.n_elements() << ","
        << ref->degree << ","
        << kappa / M_PI << ","
        << extra_error_columns(norms)
        << std::scientific << error_L2 << "\n";
}
