#pragma once
#include <Eigen/Dense>
#include "mesh.h"
#include "reference_element.h"

/*
 * Element matrices on a physical element K = F_K(K^), computed by quadrature on K^:
 *     int_K g dx = sum_q w_q |det J_K| g(F_K(xi_q)),      grad phi = J_K^{-T} grad^ phi^.
 */

/* M(i,j) = (phi_j, phi_i)_K */
inline void local_mass(const ReferenceElement &ref, const Element &K, Eigen::MatrixXcd &M)
{
    size_t n = ref.n_basis();
    M.setZero(n, n);
    QuadratureRule quad = ref.quadrature(2 * ref.degree + 1);

    for (size_t q = 0; q < quad.size(); q++)
    {
        double w = K.det_J * quad.weights[q];
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                M(i, j) += w * ref.eval_basis(i, quad.points[q]) * ref.eval_basis(j, quad.points[q]);
    }
}

/* S(i,j) = (grad phi_j, grad phi_i)_K */
inline void local_stiffness(const ReferenceElement &ref, const Element &K, Eigen::MatrixXcd &S)
{
    size_t n = ref.n_basis();
    S.setZero(n, n);
    QuadratureRule quad = ref.quadrature(2 * ref.degree + 1);

    for (size_t q = 0; q < quad.size(); q++)
    {
        double w = K.det_J * quad.weights[q];
        for (size_t i = 0; i < n; i++)
        {
            Point g_i = K.J_inv_T * ref.eval_grad(i, quad.points[q]);
            for (size_t j = 0; j < n; j++)
            {
                Point g_j = K.J_inv_T * ref.eval_grad(j, quad.points[q]);
                S(i, j) += w * g_i.dot(g_j);
            }
        }
    }
}

/* H(i,j) = (phi_i, d_k phi_j)_K  ("half stiffness" in direction k) */
inline void local_half_stiffness(const ReferenceElement &ref, const Element &K, size_t k, Eigen::MatrixXcd &H)
{
    size_t n = ref.n_basis();
    H.setZero(n, n);
    QuadratureRule quad = ref.quadrature(2 * ref.degree + 1);

    for (size_t q = 0; q < quad.size(); q++)
    {
        double w = K.det_J * quad.weights[q];
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
            {
                Point g_j = K.J_inv_T * ref.eval_grad(j, quad.points[q]);
                H(i, j) += w * ref.eval_basis(i, quad.points[q]) * g_j(k);
            }
    }
}
