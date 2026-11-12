#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "quadrature.h"
#include "reference_element.h"

/* Local stiffness matrix S_loc(i,j) = int_K phi_i' phi_j' dx
 * on a physical element of size h (x = x_c + h/2 xi, so d/dx = 2/h d/dxi). */
inline void stiffness(const ReferenceElement &ref_el, double h, Eigen::MatrixXcd &S_loc)
{
    size_t n_dofs = ref_el.degree + 1;
    S_loc.setZero();

    QuadratureRule quad_rule = get_gauss_legendre(n_dofs);

    for (size_t q = 0; q < n_dofs; q++)
    {
        double point_q = quad_rule.points[q];
        double weight_q = quad_rule.weights[q];
        for (size_t i = 0; i < n_dofs; i++)
            for (size_t j = 0; j < n_dofs; j++)
            {
                double d_i = ref_el.eval_basis_deriv(i, point_q);
                double d_j = ref_el.eval_basis_deriv(j, point_q);
                S_loc(i, j) += 2. / h * weight_q * d_i * d_j;
            }
    }
}
