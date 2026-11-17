#pragma once

#include <complex>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>

/*
 * Basic types shared by the whole library.
 *
 * Points and small matrices have a dynamic size (the space dimension d) bounded by MAX_DIM, so that
 * Eigen stores them on the stack (no heap allocation) and the same code works for d = 1, 2, 3.
 */
constexpr int MAX_DIM = 3;

using Point = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MAX_DIM, 1>;                       // x or xi in R^d
using CPoint = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1, 0, MAX_DIM, 1>;        // complex vector of R^d
using SmallMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, MAX_DIM, MAX_DIM>; // d x d

using ScalarFunction = std::function<std::complex<double>(const Point &)>; // f, u
using VectorFunction = std::function<CPoint(const Point &)>;               // grad u

typedef Eigen::Triplet<std::complex<double>> TripletComplex;

inline const std::complex<double> I(0.0, 1.0);

enum BCType
{
    Dirichlet = 0,
    Neumann = 1,
    Robin = 2
};

/* Unknown fields: u, then the components q_1, ..., q_d of q (dGUF only), i.e. field Q + k for q_{k+1}. */
enum Variable
{
    U = 0,
    Q = 1
};

/* Helpers to build points / complex vectors */
inline Point make_point(double x)
{
    Point p(1);
    p << x;
    return p;
}
inline CPoint make_cpoint(std::complex<double> x)
{
    CPoint p(1);
    p << x;
    return p;
}
