#pragma once
#include <cstddef>
#include <utility>
#include <vector>
#include "types.h"
#include "quadrature.h"

/*
 * Reference element K^ with a polynomial basis of degree p.
 *
 * Everything the solvers need from the element type goes through this interface, so that adding a
 * new element (e.g. a triangle) only requires a new derived class:
 *   - the basis and its reference gradient,
 *   - quadrature rules on K^ and on its reference face F^ (exact for polynomials of a given degree),
 *   - the embedding of F^ into each local face of K^,
 *   - sample points for plotting.
 */
class ReferenceElement
{
public:
    explicit ReferenceElement(size_t p) : degree(p) {}
    virtual ~ReferenceElement() = default;

    size_t degree;

    virtual size_t dim() const = 0;
    virtual size_t n_basis() const = 0;
    virtual size_t n_faces() const = 0;

    virtual double eval_basis(size_t k, const Point &xi) const = 0;
    virtual Point eval_grad(size_t k, const Point &xi) const = 0; // reference gradient

    virtual QuadratureRule quadrature(size_t exactness) const = 0;      // on K^
    virtual QuadratureRule face_quadrature(size_t exactness) const = 0; // on F^ (dimension d - 1)
    virtual Point face_to_element(size_t local_face, const Point &eta) const = 0; // F^ -> local face of K^
    virtual std::vector<Point> sample_points(size_t n) const = 0; // n subdivisions per direction
};

/*
 * Reference interval K^ = [-1, 1] with the monomial basis xi^k, k = 0..p.
 * Local face 0 is xi = -1, local face 1 is xi = +1.
 */
class ReferenceInterval : public ReferenceElement
{
public:
    explicit ReferenceInterval(size_t p) : ReferenceElement(p) {}

    size_t dim() const override { return 1; }
    size_t n_basis() const override { return degree + 1; }
    size_t n_faces() const override { return 2; }

    double eval_basis(size_t k, const Point &xi) const override;
    Point eval_grad(size_t k, const Point &xi) const override;

    QuadratureRule quadrature(size_t exactness) const override;
    QuadratureRule face_quadrature(size_t exactness) const override;
    Point face_to_element(size_t local_face, const Point &eta) const override;
    std::vector<Point> sample_points(size_t n) const override;
};

/*
 * Reference triangle K^ with vertices v0 = (-1,-1), v1 = (1,-1), v2 = (-1,1)  (|K^| = 2),
 * with the orthonormal Dubiner basis of P_p (dimension (p+1)(p+2)/2):
 *
 *     psi_{ij}(xi, eta) = sqrt(2) P~_i^{(0,0)}(a) P~_j^{(2i+1,0)}(b) (1 - b)^i,   i + j <= p,
 *
 * in the collapsed coordinates a = 2 (1 + xi) / (1 - eta) - 1, b = eta, where P~ are the Jacobi polynomials
 * normalized in L2(-1, 1) with their weight (Hesthaven & Warburton, Nodal DG Methods, 2008, Sec. 6.1).
 * Basis index k runs over i = 0..p, then j = 0..p-i.
 *
 * Local faces (edges), each parametrized by eta in [-1, 1] from its first to its second vertex:
 *     face 0: v0 -> v1,   face 1: v1 -> v2,   face 2: v2 -> v0.
 */
class ReferenceTriangle : public ReferenceElement
{
public:
    explicit ReferenceTriangle(size_t p);

    size_t dim() const override { return 2; }
    size_t n_basis() const override { return (degree + 1) * (degree + 2) / 2; }
    size_t n_faces() const override { return 3; }

    double eval_basis(size_t k, const Point &xi) const override;
    Point eval_grad(size_t k, const Point &xi) const override;

    QuadratureRule quadrature(size_t exactness) const override;
    QuadratureRule face_quadrature(size_t exactness) const override;
    Point face_to_element(size_t local_face, const Point &eta) const override;
    std::vector<Point> sample_points(size_t n) const override;

    static const double vertices[3][2];

private:
    std::vector<std::pair<size_t, size_t>> ij; // basis index k -> (i, j)
};

/* Jacobi polynomial P_n^{(alpha,beta)}(x) normalized in L2((-1,1), (1-x)^alpha (1+x)^beta), and its derivative */
double jacobi_p(double x, double alpha, double beta, size_t n);
double grad_jacobi_p(double x, double alpha, double beta, size_t n);
