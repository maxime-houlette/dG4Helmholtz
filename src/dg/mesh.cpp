#include <algorithm>
#include "mesh.h"

Mesh::Mesh(size_t N, double L) : dim(1)
{
    double hK = L / N;
    h = hK;

    /* Vertices */
    for (size_t i = 0; i < N + 1; i++)
        vertices.push_back(make_point(i * hK));

    /* Elements K_e = [x_e, x_{e+1}], F_K(xi) = (x_e + x_{e+1}) / 2 + h/2 xi */
    for (size_t e = 0; e < N; e++)
    {
        Element K;
        K.vertices = {e, e + 1};
        K.x0 = make_point(0.5 * (vertices[e](0) + vertices[e + 1](0)));
        K.J = SmallMatrix::Constant(1, 1, hK / 2.);
        K.J_inv_T = SmallMatrix::Constant(1, 1, 2. / hK);
        K.det_J = hK / 2.;
        K.h = hK;
        elements.push_back(K);
    }

    /* Interfaces x_{e+1}: K_+ = K_e (left, outward normal +1, local face 1), K_- = K_{e+1} (local face 0) */
    for (size_t e = 0; e + 1 < N; e++)
    {
        Face F;
        F.elem[0] = e;
        F.elem[1] = e + 1;
        F.local_face[0] = 1;
        F.local_face[1] = 0;
        F.boundary = false;
        F.tag = -1;
        F.normal = make_point(1.);
        F.jac = 1.;
        F.h = std::min(elements[e].h, elements[e + 1].h);
        F.flip = false;
        faces.push_back(F);
    }

    /* Boundary faces x = 0 (tag 0, n = -1) and x = L (tag 1, n = +1) */
    Face F0;
    F0.elem[0] = F0.elem[1] = 0;
    F0.local_face[0] = F0.local_face[1] = 0;
    F0.boundary = true;
    F0.tag = 0;
    F0.normal = make_point(-1.);
    F0.jac = 1.;
    F0.h = elements[0].h;
    F0.flip = false;
    faces.push_back(F0);

    Face FL = F0;
    FL.elem[0] = FL.elem[1] = N - 1;
    FL.local_face[0] = FL.local_face[1] = 1;
    FL.tag = 1;
    FL.normal = make_point(1.);
    FL.h = elements[N - 1].h;
    faces.push_back(FL);
}
