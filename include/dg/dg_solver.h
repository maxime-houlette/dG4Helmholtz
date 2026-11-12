#pragma once

#include <complex>
#include <functional>
#include <string>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "mesh.h"
#include "reference_element.h"

using RhsFunction = std::function<std::complex<double>(double)>;
typedef Eigen::Triplet<std::complex<double>> TripletComplex;

inline const std::complex<double> I(0.0, 1.0);

enum BCType
{
    Dirichlet = 0,
    Neumann = 1,
    Robin = 2
};

/* Unknown fields. IPDG only has U, dGUF has U and Q. */
enum Variable
{
    U = 0,
    Q = 1
};

/*
 * Common base class of the 1D dG solvers (dGUF and IPDG) for the Helmholtz equation on (0, L).
 *
 * It owns the mesh, the reference element, the linear system A x = b and the discrete solution, and
 * implements everything that does not depend on the method: dof numbering, right-hand side, linear
 * solve, post-processing and error computation.
 *
 * A derived class only provides the bilinear form through
 *     assemble_volume(), assemble_interfaces(), assemble_boundaries(),
 * and must call assemble() at the end of its constructor.
 *
 * Dof numbering: element by element, and inside an element field by field (U first, then Q):
 *     global = (n_fields * e + var) * (p + 1) + i.
 */
class DGSolver
{
public:
    virtual ~DGSolver() = default;

    void solve();
    void post_process_u();
    void post_process_u_exact();
    void plot_u();
    void plot_error();

    /* Appends a line "N,degree,kappa/Pi,[method-specific columns,]error_L2" to `filename`
     * (default_error_file() if empty). error_L2 is the relative L2 error on u. */
    void compute_errors(const std::string &filename = "");

    const Eigen::VectorXcd &get_solution() const { return u; }

protected:
    DGSolver(const Mesh &input_mesh,
             size_t degree,
             size_t input_n_fields,
             double input_kappa,
             BCType input_bc_type,
             RhsFunction input_f_rhs,
             RhsFunction input_u_exact,
             RhsFunction input_du_exact);

    Mesh mesh;
    ReferenceElement ref_element;
    size_t n_fields;
    double kappa;
    BCType bc_type;

    Eigen::SparseMatrix<std::complex<double>> A;
    Eigen::VectorXcd b;
    Eigen::VectorXcd u;
    std::vector<TripletComplex> tripletList;

    RhsFunction f_rhs = nullptr;
    RhsFunction u_exact = nullptr;
    RhsFunction du_exact = nullptr;

    /* Reference basis functions and their (reference) derivatives at xi = -1 (L) and xi = +1 (R) */
    std::vector<double> phi_L, phi_R, dphi_L, dphi_R;

    /* --- Assembly --- */
    void assemble(); // volume, interfaces, boundaries, then rhs
    virtual void assemble_volume() = 0;
    virtual void assemble_interfaces() = 0;
    virtual void assemble_boundaries() = 0;
    void assemble_rhs(); // (f, phi_i) on the U rows

    /* --- Helpers --- */
    size_t n_local_dofs() const { return ref_element.degree + 1; }
    size_t local_to_global(size_t element, Variable var, size_t local_dof) const;
    std::complex<double> eval_u(size_t element, double xi) const;
    std::complex<double> eval_du(size_t element, double xi) const; // physical derivative

    /* --- Errors --- */
    struct ErrorNorms
    {
        double err_L2_sq = 0., sol_L2_sq = 0.; // ||u - u_h||^2, ||u||^2
        double err_H1_sq = 0., sol_H1_sq = 0.; // ||u' - u_h'||_{T_h}^2, ||u'||^2 (only if du_exact is provided)
    };
    ErrorNorms integrate_errors() const;

    virtual std::string default_error_file() const = 0;
    virtual std::string extra_error_header() const { return ""; }                    // e.g. "gamma,error_energy,"
    virtual std::string extra_error_columns(const ErrorNorms &) const { return ""; } // matching values
};
