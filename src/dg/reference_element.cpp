#include <cmath>
#include "reference_element.h"

double ReferenceInterval::eval_basis(size_t k, const Point &xi) const
{
    if (k == 0)
        return 1.0;
    if (k == 1)
        return xi(0);
    return std::pow(xi(0), k);
}

Point ReferenceInterval::eval_grad(size_t k, const Point &xi) const
{
    Point g(1);
    if (k == 0)
        g(0) = 0.0;
    else if (k == 1)
        g(0) = 1.0;
    else
        g(0) = k * std::pow(xi(0), k - 1);
    return g;
}

/* Gauss-Legendre with n points is exact up to degree 2n - 1 */
QuadratureRule ReferenceInterval::quadrature(size_t exactness) const
{
    return get_gauss_legendre(exactness / 2 + 1);
}

QuadratureRule ReferenceInterval::face_quadrature(size_t) const
{
    return get_point_rule();
}

Point ReferenceInterval::face_to_element(size_t local_face, const Point &) const
{
    return make_point(local_face == 0 ? -1.0 : 1.0);
}

std::vector<Point> ReferenceInterval::sample_points(size_t n) const
{
    std::vector<Point> pts;
    for (size_t p = 0; p <= n; p++)
        pts.push_back(make_point(-1.0 + 2.0 * (double)p / n));
    return pts;
}
