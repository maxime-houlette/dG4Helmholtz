/*
 * Regression test of the 1D solvers against tests/reference/, generated on 2026-11-14 with the 1D-only
 * implementation (before the dimension-independent rewrite).
 *
 * For every configuration (dGUF: Dirichlet, Neumann, Robin with/without stabilization, L2 projection;
 * IPDG: Dirichlet, Robin; p = 1, 2, 3; N = 7, 40), it checks
 *   - the solution vector:  max_i |u_i - u_i^ref| <= TOL_VEC * max_i |u_i^ref|,
 *   - the error columns of the CSV line (relative difference <= TOL_ERR, the reference has 7 digits).
 *
 * Usage: ./regression_test   (returns 0 if all checks pass)
 */
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "dguf_solver.h"
#include "ipdg_solver.h"

#ifndef REFERENCE_DIR
#define REFERENCE_DIR "tests/reference"
#endif

using C = std::complex<double>;
const double TOL_VEC = 1e-10;
const double TOL_ERR = 1e-5;

static bool compare_vector(const std::string &name, const Eigen::VectorXcd &u, double &rel)
{
    std::ifstream in(std::string(REFERENCE_DIR) + "/vec_" + name + ".txt");
    std::vector<C> ref;
    double re, im;
    while (in >> re >> im)
        ref.emplace_back(re, im);
    if (ref.size() != (size_t)u.size())
    {
        std::cout << "  size mismatch: " << u.size() << " vs reference " << ref.size() << "\n";
        return false;
    }
    double max_diff = 0., max_ref = 0.;
    for (size_t i = 0; i < ref.size(); i++)
    {
        max_diff = std::max(max_diff, std::abs(u(i) - ref[i]));
        max_ref = std::max(max_ref, std::abs(ref[i]));
    }
    rel = max_diff / max_ref;
    return rel <= TOL_VEC;
}

static std::vector<std::string> last_line_fields(const std::string &file)
{
    std::ifstream in(file);
    std::string line, last;
    while (std::getline(in, line))
        if (!line.empty())
            last = line;
    std::vector<std::string> fields;
    std::stringstream ss(last);
    std::string f;
    while (std::getline(ss, f, ','))
        fields.push_back(f);
    return fields;
}

static bool compare_errors(const std::string &name)
{
    std::vector<std::string> ref = last_line_fields(std::string(REFERENCE_DIR) + "/err_" + name + ".csv");
    std::vector<std::string> now = last_line_fields("err_" + name + ".csv");
    if (ref.size() != now.size())
        return false;
    for (size_t k = 0; k < ref.size(); k++)
    {
        double a = std::stod(ref[k]), b = std::stod(now[k]);
        if (std::abs(a - b) > TOL_ERR * std::abs(a))
        {
            std::cout << "  error column " << k << ": " << now[k] << " vs reference " << ref[k] << "\n";
            return false;
        }
    }
    return true;
}

int main()
{
    const double L = 1.0, kappa = 10.1 * M_PI;
    const int n = 3;
    auto x_of = [](const Point &X)
    { return X(0); };

    /* dGUF test cases */
    ScalarFunction fD = [=](const Point &X)
    { return C(0., 1.) * (kappa - n * n * M_PI * M_PI / kappa) * std::sin(n * M_PI * x_of(X)); };
    ScalarFunction uD = [=](const Point &X)
    { return C(std::sin(n * M_PI * x_of(X)), 0.); };
    ScalarFunction fN = [=](const Point &X)
    { double x = x_of(X); return C(0.0, std::sin(kappa) * (kappa * kappa * x * x / 2.0 + 1.0)); };
    ScalarFunction uN = [=](const Point &X)
    { double x = x_of(X); return C(std::cos(kappa * x) + kappa * std::sin(kappa) / 2.0 * x * x, 0.0); };
    ScalarFunction fR = [](const Point &)
    { return C(1.); };
    ScalarFunction uR = [=](const Point &X)
    { double x = x_of(X); return I / (2. * kappa) * (cos(kappa * x) + cos(kappa * (L - x)) - 2.) + 1. / (2. * kappa) * (sin(kappa * x) + sin(kappa * (L - x))); };
    VectorFunction duR = [=](const Point &X)
    { double x = x_of(X); return make_cpoint(I / 2. * (-sin(kappa * x) + sin(kappa * (L - x))) + 1. / 2. * (cos(kappa * x) - cos(kappa * (L - x)))); };

    /* IPDG test cases */
    ScalarFunction gR = [=](const Point &)
    { return I * kappa; };
    double c = std::cos(kappa * L / 2.);
    ScalarFunction g1 = [](const Point &)
    { return C(1.); };
    ScalarFunction u1 = [=](const Point &X)
    { return C((std::cos(kappa * (x_of(X) - L / 2.)) / c - 1.) / (kappa * kappa)); };
    VectorFunction du1 = [=](const Point &X)
    { return make_cpoint(C(-std::sin(kappa * (x_of(X) - L / 2.)) / (kappa * c))); };

    int n_fail = 0, n_tests = 0;
    std::streambuf *cout_buf = std::cout.rdbuf();

    for (size_t N : {7, 40})
        for (size_t p : {1, 2, 3})
        {
            Mesh mesh(N, L);
            std::string suffix = "_N" + std::to_string(N) + "_p" + std::to_string(p);
            double gamma = 2. * (p + 1) * (p + 1);

            auto run = [&](const std::string &base, DGSolver &solver)
            {
                std::string name = base + suffix;
                std::remove(("err_" + name + ".csv").c_str());

                std::ostringstream silent; // silence the solver messages
                std::cout.rdbuf(silent.rdbuf());
                solver.solve();
                solver.compute_errors("err_" + name + ".csv");
                std::cout.rdbuf(cout_buf);

                double rel = 0.;
                bool ok = compare_vector(name, solver.get_solution(), rel);
                ok = compare_errors(name) && ok;
                std::remove(("err_" + name + ".csv").c_str());

                n_tests++;
                if (!ok)
                    n_fail++;
                std::printf("%-4s %-32s vector rel. diff = %.2e\n", ok ? "OK" : "FAIL", name.c_str(), rel);
            };

            { DGUFSolver S(mesh, p, kappa, Dirichlet, fD, uD, nullptr, 1., true, Helmholtz); run("dguf_dirichlet", S); }
            { DGUFSolver S(mesh, p, kappa, Neumann, fN, uN, nullptr, 1., true, Helmholtz); run("dguf_neumann", S); }
            { DGUFSolver S(mesh, p, kappa, Robin, fR, uR, nullptr, 1., false, Helmholtz); run("dguf_robin_nostab", S); }
            { DGUFSolver S(mesh, p, kappa, Robin, fR, uR, nullptr, 1., true, Helmholtz); run("dguf_robin_stab", S); }
            { DGUFSolver S(mesh, p, kappa, Robin, fR, uR, nullptr, 1., true, L2); run("dguf_l2proj", S); }
            { IPDGSolver S(mesh, p, kappa, Dirichlet, g1, u1, du1, gamma); run("ipdg_dirichlet", S); }
            { IPDGSolver S(mesh, p, kappa, Robin, gR, uR, duR, gamma); run("ipdg_robin", S); }
        }

    std::printf("\n%d / %d configurations passed.\n", n_tests - n_fail, n_tests);
    return n_fail == 0 ? 0 : 1;
}
