#ifndef _MESH_H_
#define _MESH_H_

#include "em_defs.h"
#include "tetgen.h"

#include <mpi.h>
#include <nanoflann.hpp>

#include <map>
#include <memory>
#include <vector>

const int VERTICES_PER_EDGE = 2;
const int VERTICES_PER_TET = 4;
const int VERTICES_PER_TRIANGLE = 3;
const int EDGES_PER_TET = 6;
const int EDGES_PER_TRIANGLE = 3;
const int TRIANGLES_PER_TET = 4;

class TetAccessor;

class Mesh {
public:
  Mesh(MPI_Comm comm) : comm_(comm), tio_(NULL) {
    MPI_Comm_size(comm_, &comm_size_);
    MPI_Comm_rank(comm_, &comm_rank_);
  }

  ~Mesh() {
    if (tio_) {
      delete tio_;
    }
  }

  void create(const std::vector<Point> &, const Eigen::MatrixXi &, const std::vector<int> &);
  void create_from_tetgen(tetgenio *);

  void copy(const Mesh &);

  int n_vertices() const { return vertices_.size(); }
  int n_tets() const { return tetrahedra_.rows(); }
  int n_edges() const { return edges_.rows(); }

  void refine_tetgen(const std::vector<bool> &);

  void get_vertex_owners(std::vector<int> &);

  void get_edge_owners(std::vector<int> &);

  void save_vtk(const char *, const std::map<std::string, Eigen::VectorXd> &);

public:
  typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, Mesh>, Mesh, 3>
      KDTree;

  size_t kdtree_get_point_count() const { return (size_t)n_vertices(); }

  double kdtree_distance(const double *p, const size_t idx, size_t) const {
    const double *ptr = &vertices_[idx][0];

    return std::sqrt((p[0] - ptr[0]) * (p[0] - ptr[0]) + (p[1] - ptr[1]) * (p[1] - ptr[1]) +
                     (p[2] - ptr[2]) * (p[2] - ptr[2]));
  }

  double kdtree_get_pt(const size_t idx, int dim) const { return vertices_[idx][dim]; }

  template <typename BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }

  void init_kd_tree();
  std::tuple<Point, int> find_closest_vertex(const Point &);
  int find_cell_around_point(const Point &);

private:
  void partition();
  void extract_ghost_cells();
  void compute_new_vertex_indices_();
  void compute_new_edge_indices_();

private:
  void build_vertex_info();
  void build_edge_info();
  void build_neighbor_info();
  void build_boundary_info();

private:
  MPI_Comm comm_;
  int comm_rank_, comm_size_;

  tetgenio *tio_;

  std::vector<Point> vertices_;
  std::vector<bool> vertex_bdr_marker_;

  Eigen::MatrixXi tetrahedra_;
  std::vector<int> tet_attributes_, subdomain_;
  std::vector<bool> ghost_flag_;

  Eigen::MatrixXi tet_neighbors_, tet_neighbor_of_neighbor_;

  Eigen::MatrixXi edges_, tet_edges_;
  std::vector<int> new_edge_indices_, new_vertex_indices_;
  std::vector<bool> edge_bdr_marker_;

  std::vector<int> vertex_to_tet_;
  std::shared_ptr<KDTree> kd_index_;

  friend class TetAccessor;
};

class TetAccessor {
public:
  TetAccessor() : m_(NULL), tidx_(-1) {}
  TetAccessor(const Mesh *m, int tidx) : m_(m), tidx_(tidx) {}

  int index() const { return tidx_; }

  Point vertex(int v) const {
    assert(v < VERTICES_PER_TET);
    return m_->vertices_[m_->tetrahedra_(tidx_, v)];
  }

  int vertex_index(int v) const {
    assert(v < VERTICES_PER_TET);
    return m_->new_vertex_indices_[m_->tetrahedra_(tidx_, v)];
  }

  bool vertex_on_boundary(int v) const {
    assert(v < VERTICES_PER_TET);
    return m_->vertex_bdr_marker_[m_->tetrahedra_(tidx_, v)];
  }

  int edge_index(int e) const {
    assert(e < EDGES_PER_TET);
    return m_->new_edge_indices_[m_->tet_edges_(tidx_, e)];
  }

  int edge_end_point(int e, int ep) const {
    int v, eep;

    assert(e < EDGES_PER_TET && ep < 2);

    eep = -1;
    for (v = 0; v < VERTICES_PER_TET; ++v) {
      if (m_->edges_(m_->tet_edges_(tidx_, e), ep) == m_->tetrahedra_(tidx_, v)) {
        eep = v;
        break;
      }
    }

    assert(eep >= 0);

    return eep;
  }

  bool edge_on_boundary(int e) const {
    assert(e < EDGES_PER_TET);
    return m_->edge_bdr_marker_[m_->tet_edges_(tidx_, e)];
  }

  int neighbor(int n) const {
    assert(n < TRIANGLES_PER_TET);
    return m_->tet_neighbors_(tidx_, n);
  }

  int neighbor_of_neighbor(int n) const {
    assert(n < TRIANGLES_PER_TET);
    return m_->tet_neighbor_of_neighbor_(tidx_, n);
  }

  bool face_on_boundary(int f) const {
    assert(f < TRIANGLES_PER_TET);
    return m_->tet_neighbors_(tidx_, f) < 0;
  }

  int attribute() const { return m_->tet_attributes_[tidx_]; }

  double volume() const {
    int i;
    Eigen::Matrix4d jac;

    for (i = 0; i < 4; ++i) {
      jac(0, i) = 1;
      jac.col(i).tail(3) = vertex(i);
    }

    return jac.determinant() / 6.0;
  }

  bool contains_point(const Point &p) const {
    double d[4];
    int i, gt_zero;
    Point vertices[4];

    for (i = 0; i < 4; ++i) {
      vertices[i] = vertex(i);
    }

    d[0] = orient3d(&vertices[0][0], &vertices[1][0], &vertices[3][0], const_cast<double *>(&p[0]));
    d[1] = orient3d(&vertices[1][0], &vertices[2][0], &vertices[3][0], const_cast<double *>(&p[0]));
    d[2] = orient3d(&vertices[1][0], &vertices[0][0], &vertices[2][0], const_cast<double *>(&p[0]));
    d[3] = orient3d(&vertices[0][0], &vertices[3][0], &vertices[2][0], const_cast<double *>(&p[0]));

    gt_zero = 0;
    for (i = 0; i < 4; ++i) {
      if (d[i] >= 0.0) {
        ++gt_zero;
      }
    }

    return (gt_zero == 4);
  }

  double diameter() const {
    int i;
    Point vertices[VERTICES_PER_TET];
    double a, b, c, aa, bb, cc, la, lb, lc, s;

    for (i = 0; i < VERTICES_PER_TET; ++i) {
      vertices[i] = vertex(i);
    }

    a = (vertices[1] - vertices[2]).norm();
    b = (vertices[0] - vertices[2]).norm();
    c = (vertices[0] - vertices[1]).norm();
    aa = (vertices[0] - vertices[3]).norm();
    bb = (vertices[1] - vertices[3]).norm();
    cc = (vertices[2] - vertices[3]).norm();

    la = a * aa;
    lb = b * bb;
    lc = c * cc;
    s = 0.5 * (la + lb + lc);

    return std::sqrt(s * (s - la) * (s - lb) * (s - lc)) / (6.0 * volume());
  }

  double face_diameter(int f) const {
    int i;
    double a, b, c, area;
    Point vertices[VERTICES_PER_TRIANGLE];

    assert(f < TRIANGLES_PER_TET);

    for (i = 0; i < VERTICES_PER_TRIANGLE; ++i) {
      vertices[i] = vertex((f + i + 1) % VERTICES_PER_TET);
    }

    a = (vertices[1] - vertices[2]).norm();
    b = (vertices[0] - vertices[2]).norm();
    c = (vertices[0] - vertices[1]).norm();

    area = 0.25 * std::sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c));

    return a * b * c / (4.0 * area);
  }

  bool is_locally_owned() const { return m_->subdomain_[tidx_] == m_->comm_rank_; }

  bool is_ghost() const { return m_->ghost_flag_[tidx_]; }

private:
  const Mesh *m_;
  int tidx_;
};

#endif
