#pragma once
#include <array>
#include <cstddef>
#include <functional>
#include <vector>
#include "types.h"

/*
 * Affine element K = F_K(K^), with F_K(xi) = x0 + J xi.
 */
struct Element
{
    std::vector<size_t> vertices;
    Point x0;            // F_K(0)
    SmallMatrix J;       // Jacobian of F_K
    SmallMatrix J_inv_T; // J^{-T}, maps reference gradients to physical gradients
    double det_J;        // |det J|, volume ratio |K| / |K^|
    double h;            // diameter h_K

    Point map(const Point &xi) const { return x0 + J * xi; }
};

/*
 * Mesh face F. Following the paper, an interface F = dK_+ cap dK_- is oriented by n_F := n_{K_+}:
 *   elem[0] = K_+ (owner of the normal), elem[1] = K_-  (unused on a boundary face),
 *   local_face[s] = index of F among the faces of elem[s] (see ReferenceElement::face_to_element),
 * and [v] = v_{K_+} - v_{K_-}, {v} = (v_{K_+} + v_{K_-}) / 2. On a boundary face, n_F is the outward normal.
 */
struct Face
{
    size_t elem[2];
    size_t local_face[2];
    bool boundary;
    int tag;      // boundary tag (to select boundary conditions), -1 for interfaces
    Point normal; // n_F
    double jac;   // |F| / |F^|, scaling of the reference face quadrature weights (1 in 1D)
    double h;     // h_F = min of the diameters of the adjacent elements
    bool flip;    // the reference face parametrization of elem[1] is reversed w.r.t. that of elem[0]
};

struct Mesh
{
    size_t dim = 0;
    std::vector<Point> vertices;
    std::vector<Element> elements;
    std::vector<Face> faces; // interfaces first, then boundary faces
    double h = 0.;           // max_K h_K

    size_t n_elements() const { return elements.size(); }

    Mesh() = default;

    /* Uniform mesh of (0, L) with N elements. Boundary tags: 0 at x = 0, 1 at x = L. */
    Mesh(size_t N, double L);

    /* Triangular mesh from a list of vertices and triangles (vertex indices, any orientation).
     * boundary_tag(x_mid) gives the tag of a boundary edge from its midpoint. */
    Mesh(const std::vector<Point> &input_vertices,
         const std::vector<std::array<size_t, 3>> &triangles,
         const std::function<int(const Point &)> &boundary_tag);

    /* Structured triangulation of (0, Lx) x (0, Ly): nx * ny rectangles, each cut along its diagonal
     * (x_i, y_j) -- (x_{i+1}, y_{j+1}) into 2 triangles.
     * Boundary tags: 0 bottom (y = 0), 1 right (x = Lx), 2 top (y = Ly), 3 left (x = 0). */
    static Mesh rectangle(size_t nx, size_t ny, double Lx, double Ly);
};
