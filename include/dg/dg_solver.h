#pragma once

#include <complex>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "types.h"
#include "mesh.h"
#include "reference_element.h"

/*
 * Common base class of the dG solvers (dGUF and IPDG) for the Helmholtz equation, in any dimension.
 *
 * It owns the mesh, the reference element, the linear system A x = b and the discrete solution, and
 * implements everything that does not depend on the method: dof numbering, assembly loops,
 * right-hand side, linear solve, post-processing and error computation.
 *
 * A derived class only provides the local contributions of the sesquilinear form:
 *     assemble_element(e)        volume terms on element e,
 *     assemble_interface(F)      terms on an interface F,
 *     assemble_boundary_face(F)  terms on a boundary face F,
 * and must call assemble() at the end of its constructor.
 *
 * Dof numbering: element by element, and inside an element field by field (u, then q_1, ..., q_d):
 *     global = (n_fields * e + field) * n_basis + i.
 */
class DGSolver
{
public:
    virtual ~DGSolver() = default;

    void solve();
    void post_process_u();
    void post_process_u_exact();
    void plot_u();     // 1D only (gnuplot)
    void plot_error(); // 1D only (gnuplot)

    /* Appends a line "N,degree,kappa/Pi,[method-specific columns,]error_L2" to `filename`
     * (default_error_file() if empty), with N the number of elements and error_L2 the relative L2 error on u. */
    void compute_errors(const std::string &filename = "");

    const Eigen::VectorXcd &get_solution() const { return u; }

protected:
    DGSolver(const Mesh &input_mesh,
             size_t degree,
             size_t input_n_fields,
             double input_kappa,
             BCType input_bc_type,
             ScalarFunction input_f_rhs,
             ScalarFunction input_u_exact,
             VectorFunction input_grad_u_exact);

    Mesh mesh;
    std::unique_ptr<ReferenceElement> ref;
    size_t n_fields;
    double kappa;
    BCType bc_type;

    Eigen::SparseMatrix<std::complex<double>> A;
    Eigen::VectorXcd b;
    Eigen::VectorXcd u;
    std::vector<TripletComplex> tripletList;

    ScalarFunction f_rhs = nullptr;
    ScalarFunction u_exact = nullptr;
    VectorFunction grad_u_exact = nullptr;

    /* --- Assembly --- */
    void assemble(); // elements, faces, then rhs
    virtual void assemble_element(size_t e) = 0;
    virtual void assemble_interface(const Face &F) = 0;
    virtual void assemble_boundary_face(const Face &F) = 0;
    void assemble_rhs(); // (f, phi_i) on the u rows

    /* Adds the dense local matrix A_loc, whose rows/columns correspond to `dofs`, to the global matrix */
    void scatter(const std::vector<size_t> &dofs, const Eigen::MatrixXcd &A_loc);

    /* --- Dofs and basis functions --- */
    size_t dim() const { return mesh.dim; }
    size_t n_local_dofs() const { return ref->n_basis(); }
    size_t local_to_global(size_t element, size_t field, size_t local_dof) const;

    Eigen::VectorXd basis_values(const Point &xi) const;                 // phi_i(xi), i = 0..n-1
    Eigen::MatrixXd basis_grads(size_t element, const Point &xi) const; // row i = physical grad phi_i at F_K(xi)
    std::complex<double> eval_u(size_t element, const Point &xi) const;
    CPoint eval_grad_u(size_t element, const Point &xi) const;

    /* --- Face quadrature --- */
    struct FaceQuadPoint
    {
        Point x;     // physical point
        double w;    // weight (including |F| / |F^|)
        Point xi[2]; // reference coordinates in elem[0] and elem[1] (= xi[0] on a boundary face)
    };
    std::vector<FaceQuadPoint> face_quadrature(const Face &F) const; // exact for degree 2p

    /* --- Errors --- */
    struct ErrorNorms
    {
        double err_L2_sq = 0., sol_L2_sq = 0.; // ||u - u_h||^2, ||u||^2
        double err_H1_sq = 0., sol_H1_sq = 0.; // ||grad(u - u_h)||_{T_h}^2, ||grad u||^2 (if grad_u_exact is provided)
    };
    ErrorNorms integrate_errors() const;

    virtual std::string default_error_file() const = 0;
    virtual std::string extra_error_header() const { return ""; }                    // e.g. "gamma,error_energy,"
    virtual std::string extra_error_columns(const ErrorNorms &) const { return ""; } // matching values
};
