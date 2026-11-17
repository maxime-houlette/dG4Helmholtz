#pragma once
#include <cstddef>
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
