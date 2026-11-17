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

/* ======================================================================= */
/*                            JACOBI POLYNOMIALS                           */
/* ======================================================================= */

/* Three-term recurrence for the normalized Jacobi polynomials (Hesthaven & Warburton, Appendix A, JacobiP) */
double jacobi_p(double x, double alpha, double beta, size_t n)
{
    double gamma0 = std::pow(2., alpha + beta + 1.) / (alpha + beta + 1.) *
                    std::tgamma(alpha + 1.) * std::tgamma(beta + 1.) / std::tgamma(alpha + beta + 1.);
    double p_prev = 1. / std::sqrt(gamma0);
    if (n == 0)
        return p_prev;

    double gamma1 = (alpha + 1.) * (beta + 1.) / (alpha + beta + 3.) * gamma0;
    double p_curr = ((alpha + beta + 2.) * x / 2. + (alpha - beta) / 2.) / std::sqrt(gamma1);
    if (n == 1)
        return p_curr;

    double a_old = 2. / (2. + alpha + beta) * std::sqrt((alpha + 1.) * (beta + 1.) / (alpha + beta + 3.));
    for (size_t i = 1; i < n; i++)
    {
        double h1 = 2. * i + alpha + beta;
        double a_new = 2. / (h1 + 2.) * std::sqrt((i + 1.) * (i + 1. + alpha + beta) * (i + 1. + alpha) * (i + 1. + beta) /
                                                  (h1 + 1.) / (h1 + 3.));
        double b_new = -(alpha * alpha - beta * beta) / h1 / (h1 + 2.);
        double p_next = 1. / a_new * (-a_old * p_prev + (x - b_new) * p_curr);
        p_prev = p_curr;
        p_curr = p_next;
        a_old = a_new;
    }
    return p_curr;
}

/* d/dx P~_n^{(alpha,beta)} = sqrt(n (n + alpha + beta + 1)) P~_{n-1}^{(alpha+1,beta+1)} */
double grad_jacobi_p(double x, double alpha, double beta, size_t n)
{
    if (n == 0)
        return 0.;
    return std::sqrt(n * (n + alpha + beta + 1.)) * jacobi_p(x, alpha + 1., beta + 1., n - 1);
}

/* ======================================================================= */
/*                            REFERENCE TRIANGLE                           */
/* ======================================================================= */

const double ReferenceTriangle::vertices[3][2] = {{-1., -1.}, {1., -1.}, {-1., 1.}};

ReferenceTriangle::ReferenceTriangle(size_t p) : ReferenceElement(p)
{
    for (size_t i = 0; i <= p; i++)
        for (size_t j = 0; j + i <= p; j++)
            ij.emplace_back(i, j);
}

/* Collapsed coordinates (a, b) of (xi, eta); a is set to -1 at the singular vertex eta = 1 */
static void collapse(const Point &xi, double &a, double &b)
{
    b = xi(1);
    a = (std::abs(1. - b) > 1e-14) ? 2. * (1. + xi(0)) / (1. - b) - 1. : -1.;
}

double ReferenceTriangle::eval_basis(size_t k, const Point &xi) const
{
    auto [i, j] = ij[k];
    double a, b;
    collapse(xi, a, b);
    return std::sqrt(2.) * jacobi_p(a, 0., 0., i) * jacobi_p(b, 2. * i + 1., 0., j) * std::pow(1. - b, (double)i);
}

/* Gradient with respect to (xi, eta) (Hesthaven & Warburton, GradSimplex2DP) */
Point ReferenceTriangle::eval_grad(size_t k, const Point &xi) const
{
    auto [i, j] = ij[k];
    double a, b;
    collapse(xi, a, b);

    double fa = jacobi_p(a, 0., 0., i), dfa = grad_jacobi_p(a, 0., 0., i);
    double gb = jacobi_p(b, 2. * i + 1., 0., j), dgb = grad_jacobi_p(b, 2. * i + 1., 0., j);
    double c = 0.5 * (1. - b);

    double d_xi = dfa * gb;
    double d_eta = dfa * gb * 0.5 * (1. + a);
    if (i > 0)
    {
        d_xi *= std::pow(c, i - 1.);
        d_eta *= std::pow(c, i - 1.);
    }
    double tmp = dgb * std::pow(c, (double)i);
    if (i > 0)
        tmp -= 0.5 * i * gb * std::pow(c, i - 1.);
    d_eta += fa * tmp;

    double scale = std::pow(2., i + 0.5);
    Point g(2);
    g << scale * d_xi, scale * d_eta;
    return g;
}

/*
 * Collapsed Gauss-Legendre rule: (a, b) in [-1,1]^2 -> xi = (1 + a)(1 - b)/2 - 1, eta = b, with Jacobian (1 - b)/2.
 * A polynomial of total degree n becomes of degree n in a and n + 1 in b (Jacobian), hence
 * (n + 2)/2 points in a and (n + 3)/2 points in b.
 */
QuadratureRule ReferenceTriangle::quadrature(size_t exactness) const
{
    QuadratureRule ga = get_gauss_legendre(exactness / 2 + 1);
    QuadratureRule gb = get_gauss_legendre((exactness + 3) / 2);

    QuadratureRule rule;
    for (size_t qa = 0; qa < ga.size(); qa++)
        for (size_t qb = 0; qb < gb.size(); qb++)
        {
            double a = ga.points[qa](0), b = gb.points[qb](0);
            Point p(2);
            p << (1. + a) * (1. - b) / 2. - 1., b;
            rule.points.push_back(p);
            rule.weights.push_back(ga.weights[qa] * gb.weights[qb] * (1. - b) / 2.);
        }
    return rule;
}

/* Edges are segments: Gauss-Legendre on [-1, 1] */
QuadratureRule ReferenceTriangle::face_quadrature(size_t exactness) const
{
    return get_gauss_legendre(exactness / 2 + 1);
}

Point ReferenceTriangle::face_to_element(size_t local_face, const Point &eta) const
{
    const double *va = vertices[local_face];
    const double *vb = vertices[(local_face + 1) % 3];
    double t = eta(0);
    Point p(2);
    p << va[0] * (1. - t) / 2. + vb[0] * (1. + t) / 2.,
        va[1] * (1. - t) / 2. + vb[1] * (1. + t) / 2.;
    return p;
}

/* Equispaced lattice with n subdivisions per edge: (n+1)(n+2)/2 points */
std::vector<Point> ReferenceTriangle::sample_points(size_t n) const
{
    std::vector<Point> pts;
    for (size_t r = 0; r <= n; r++)
        for (size_t s = 0; s + r <= n; s++)
        {
            Point p(2);
            p << -1. + 2. * r / n, -1. + 2. * s / n;
            pts.push_back(p);
        }
    return pts;
}
