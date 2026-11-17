#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
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

/* ======================================================================= */
/*                              TRIANGLE MESHES                            */
/* ======================================================================= */

Mesh::Mesh(const std::vector<Point> &input_vertices,
           const std::vector<std::array<size_t, 3>> &triangles,
           const std::function<int(const Point &)> &boundary_tag)
    : dim(2), vertices(input_vertices), h(0.)
{
    /* Elements: F_K maps v0 = (-1,-1), v1 = (1,-1), v2 = (-1,1) to P0, P1, P2 (counterclockwise),
     * i.e. F_K(xi) = P0 + J (xi - v0) with J = [ (P1 - P0)/2 | (P2 - P0)/2 ]. */
    for (const auto &tri : triangles)
    {
        std::array<size_t, 3> v = tri;
        Point P0 = vertices[v[0]], P1 = vertices[v[1]], P2 = vertices[v[2]];
        double det = (P1(0) - P0(0)) * (P2(1) - P0(1)) - (P1(1) - P0(1)) * (P2(0) - P0(0));
        if (det < 0.) // enforce counterclockwise orientation
        {
            std::swap(v[1], v[2]);
            std::swap(P1, P2);
        }

        Element K;
        K.vertices = {v[0], v[1], v[2]};
        K.J = SmallMatrix(2, 2);
        K.J.col(0) = (P1 - P0) / 2.;
        K.J.col(1) = (P2 - P0) / 2.;
        Point ones(2);
        ones << 1., 1.;
        K.x0 = P0 + K.J * ones;
        K.det_J = std::abs(K.J.determinant());
        K.J_inv_T = K.J.inverse().transpose();
        K.h = std::max({(P1 - P0).norm(), (P2 - P1).norm(), (P0 - P2).norm()});
        h = std::max(h, K.h);
        elements.push_back(K);
    }

    /* Edges: local face f of element e goes from vertex f to vertex (f+1)%3 (see ReferenceTriangle).
     * The first element met is K_+ (its outward normal is n_F), the second one K_-. */
    std::map<std::pair<size_t, size_t>, size_t> edge_to_face; // sorted vertex pair -> index in `all`
    std::vector<Face> all;
    for (size_t e = 0; e < elements.size(); e++)
        for (size_t f = 0; f < 3; f++)
        {
            size_t a = elements[e].vertices[f], b = elements[e].vertices[(f + 1) % 3];
            auto key = std::make_pair(std::min(a, b), std::max(a, b));
            auto it = edge_to_face.find(key);

            if (it == edge_to_face.end())
            {
                Point d = vertices[b] - vertices[a];
                double length = d.norm();
                Face F;
                F.elem[0] = F.elem[1] = e;
                F.local_face[0] = F.local_face[1] = f;
                F.boundary = true;
                F.tag = -1;
                F.normal = Point(2);
                F.normal << d(1) / length, -d(0) / length; // outward for a counterclockwise triangle
                F.jac = length / 2.;                        // |F^| = 2 (parameter eta in [-1, 1])
                F.h = elements[e].h;
                F.flip = false;
                edge_to_face[key] = all.size();
                all.push_back(F);
            }
            else
            {
                Face &F = all[it->second];
                F.elem[1] = e;
                F.local_face[1] = f;
                F.boundary = false;
                F.h = std::min(F.h, elements[e].h);
                /* same parametrization iff both elements run along the edge from the same vertex */
                F.flip = (elements[F.elem[0]].vertices[F.local_face[0]] != a);
            }
        }

    /* Interfaces first, then boundary faces (tagged from their midpoint) */
    for (const Face &F : all)
        if (!F.boundary)
            faces.push_back(F);
    for (Face F : all)
        if (F.boundary)
        {
            const Element &K = elements[F.elem[0]];
            Point mid = (vertices[K.vertices[F.local_face[0]]] + vertices[K.vertices[(F.local_face[0] + 1) % 3]]) / 2.;
            F.tag = boundary_tag(mid);
            faces.push_back(F);
        }
}

Mesh Mesh::rectangle(size_t nx, size_t ny, double Lx, double Ly)
{
    std::vector<Point> vertices;
    for (size_t j = 0; j <= ny; j++)
        for (size_t i = 0; i <= nx; i++)
        {
            Point p(2);
            p << Lx * i / nx, Ly * j / ny;
            vertices.push_back(p);
        }

    auto id = [nx](size_t i, size_t j)
    { return j * (nx + 1) + i; };

    std::vector<std::array<size_t, 3>> triangles;
    for (size_t j = 0; j < ny; j++)
        for (size_t i = 0; i < nx; i++)
        {
            triangles.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1)});
            triangles.push_back({id(i, j), id(i + 1, j + 1), id(i, j + 1)});
        }

    double tol = 1e-12 * std::max(Lx, Ly);
    auto tag = [Lx, Ly, tol](const Point &x)
    {
        if (std::abs(x(1)) < tol)
            return 0;
        if (std::abs(x(0) - Lx) < tol)
            return 1;
        if (std::abs(x(1) - Ly) < tol)
            return 2;
        return 3;
    };
    return Mesh(vertices, triangles, tag);
}
