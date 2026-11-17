#pragma once
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "types.h"

/* Quadrature rule on a reference domain (element or face): sum_q weights[q] * f(points[q]). */
struct QuadratureRule
{
    std::vector<Point> points;
    std::vector<double> weights;
    size_t size() const { return weights.size(); }
};

/* Gauss-Legendre rule with num_points points on [-1, 1] (exact for polynomials of degree 2 num_points - 1),
 * computed with the Golub-Welsch algorithm. */
inline QuadratureRule get_gauss_legendre(int num_points)
{
    QuadratureRule rule;
    rule.points.assign(num_points, Point(1));
    rule.weights.resize(num_points);

    if (num_points == 1)
    {
        rule.points[0](0) = 0.0;
        rule.weights[0] = 2.0;
        return rule;
    }

    /* Jacobi matrix (tridiagonal) */
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(num_points, num_points);
    for (int i = 0; i < num_points - 1; ++i)
    {
        double beta = (i + 1.0) / std::sqrt(4.0 * (i + 1.0) * (i + 1.0) - 1.0);
        J(i, i + 1) = beta;
        J(i + 1, i) = beta;
    }

    /* Eigenvalues = points, weights = 2 * (first component of the eigenvectors)^2 */
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(J);
    for (int i = 0; i < num_points; ++i)
    {
        rule.points[i](0) = solver.eigenvalues()(i);
        rule.weights[i] = 2.0 * std::pow(solver.eigenvectors()(0, i), 2);
    }
    return rule;
}

/* Rule on a 0-dimensional reference face (a point): one point, weight 1. */
inline QuadratureRule get_point_rule()
{
    QuadratureRule rule;
    rule.points.assign(1, Point(0));
    rule.weights.assign(1, 1.0);
    return rule;
}
