#include "em_fe.h"

#include <iostream>
#include <unordered_map>

void calculate_ref_face_quadrature_points(int p, std::vector<Point> &qpoints,
                                          std::vector<double> &weights) {
  switch (p) {
  case 1: {
    qpoints.resize(1);
    weights.resize(1);

    qpoints[0](0) = 1.0 / 3.0;
    qpoints[0](1) = 1.0 / 3.0;
    qpoints[0](2) = 1.0 / 3.0;

    weights[0] = 0.5;
    break;
  }
  case 2: {
    qpoints.resize(3);
    weights.resize(3);

    qpoints[0](0) = 2.0 / 3;
    qpoints[0](1) = 1.0 / 6;
    qpoints[0](2) = 1.0 / 6;

    qpoints[1](0) = 1.0 / 6;
    qpoints[1](1) = 2.0 / 3;
    qpoints[1](2) = 1.0 / 6;

    qpoints[2](0) = 1.0 / 6;
    qpoints[2](1) = 1.0 / 6;
    qpoints[2](2) = 2.0 / 3;

    weights[0] = 1.0 / 6.0;
    weights[1] = weights[0];
    weights[2] = weights[0];
    break;
  }
  default:
    break;
  }
}

void transform_face_quadrature_points(const TetAccessor &tet, int f,
                                      const std::vector<Point> &ref_qpoints,
                                      const std::vector<double> &weights,
                                      std::vector<Point> &qpoints, std::vector<double> &jxw) {
  int i, j;
  double det_jac, s;

  Point p;
  Eigen::Matrix4d jac, j_inv;
  Eigen::Vector4d natual_coords;

  for (i = 0; i < VERTICES_PER_TET; ++i) {
    jac(0, i) = 1;
    jac.col(i).tail(3) = tet.vertex(i);
  }

  jxw.resize(weights.size());
  qpoints.resize(ref_qpoints.size());

  j_inv = jac.inverse();
  det_jac = jac.determinant();

  s = (j_inv.row(f).tail(3) * det_jac * 6).norm();

  for (i = 0; i < (int)qpoints.size(); ++i) {
    p = ref_qpoints[i];
    natual_coords[f] = 0.0;
    for (j = 0; j < 3; ++j) {
      natual_coords[(f + j + 1) % 4] = p[j];
    }
    qpoints[i] = (jac * natual_coords).tail(3);

    jxw[i] = s * weights[i];
  }
}

void calculate_ref_cell_quadrature_points(int p, std::vector<Point> &qpoints,
                                          std::vector<double> &weights) {
  switch (p) {
  case 1: {
    qpoints.resize(1);
    weights.resize(1);

    qpoints[0](0) = 0.25;
    qpoints[0](1) = 0.25;
    qpoints[0](2) = 0.25;

    weights[0] = 1.0 / 6.0;
    break;
  }
  case 2: {
    const double a = 0.585410196624969;
    const double b = 0.138196601125011;

    qpoints.resize(4);
    weights.resize(4);

    qpoints[0](0) = a;
    qpoints[0](1) = b;
    qpoints[0](2) = b;

    qpoints[1](0) = b;
    qpoints[1](1) = a;
    qpoints[1](2) = b;

    qpoints[2](0) = b;
    qpoints[2](1) = b;
    qpoints[2](2) = a;

    qpoints[3](0) = b;
    qpoints[3](1) = b;
    qpoints[3](2) = b;

    weights[0] = 1.0 / 24.0;
    weights[1] = weights[0];
    weights[2] = weights[0];
    weights[3] = weights[0];
    break;
  }
  default:
    break;
  }
}

void transform_cell_quadrature_points(const TetAccessor &tet, const std::vector<Point> &ref_qpoints,
                                      const std::vector<double> &weights,
                                      std::vector<Point> &qpoints, std::vector<double> &jxw) {
  int i;
  double det_jac;

  Point p;
  Eigen::Matrix4d jac;

  for (i = 0; i < VERTICES_PER_TET; ++i) {
    jac(0, i) = 1;
    jac.col(i).tail(3) = tet.vertex(i);
  }

  jxw.resize(weights.size());
  qpoints.resize(ref_qpoints.size());

  det_jac = jac.determinant();
  for (i = 0; i < (int)qpoints.size(); ++i) {
    p = ref_qpoints[i];
    qpoints[i] = (jac * Eigen::Vector4d(1 - p[0] - p[1] - p[2], p[0], p[1], p[2])).tail(3);

    jxw[i] = det_jac * weights[i];
  }
}

FEValues::FEValues() {
}

void FEValues::reinit(const TetAccessor &tet) {
  tet_ = tet;

  generate_meta_info();
  calculate_coefs();
}

double FEValues::value_q(int n, const Point &p) const {
  assert(n < VERTICES_PER_TET);
  return a_[n] + p[0] * b_[n] + p[1] * c_[n] + p[2] * d_[n];
}

VectorD FEValues::grad_q(int n, const Point &) const {
  assert(n < VERTICES_PER_TET);
  return VectorD(b_[n], c_[n], d_[n]);
}

VectorD FEValues::value_nedelec(int e, const Point &p) const {
  assert(e < EDGES_PER_TET);
  return (value_q(ev_[e][0], p) * grad_q(ev_[e][1], p) -
          value_q(ev_[e][1], p) * grad_q(ev_[e][0], p));
}

VectorD FEValues::curl_nedelec(int e, const Point &p) const {
  assert(e < EDGES_PER_TET);
  return 2.0 * grad_q(ev_[e][0], p).cross(grad_q(ev_[e][1], p));
}

double FEValues::face_value_q(int f, int n, const Point &p) const {
  assert(f < TRIANGLES_PER_TET && n < VERTICES_PER_TRIANGLE);
  return value_q(fv_[f][n], p);
}

VectorD FEValues::face_grad_q(int f, int n, const Point &p) const {
  assert(f < TRIANGLES_PER_TET && n < VERTICES_PER_TRIANGLE);
  return grad_q(fv_[f][n], p);
}

VectorD FEValues::face_value_nedelec(int f, int e, const Point &p) const {
  assert(f < TRIANGLES_PER_TET && e < EDGES_PER_TRIANGLE);
  return value_nedelec(fe_[f][e], p);
}

VectorD FEValues::face_curl_nedelec(int f, int e, const Point &p) const {
  assert(f < TRIANGLES_PER_TET && e < EDGES_PER_TRIANGLE);
  return curl_nedelec(fe_[f][e], p);
}

VectorD FEValues::face_normal(int f) const {
  assert(f < TRIANGLES_PER_TET);
  return (-VectorD(b_[f], c_[f], d_[f])).normalized();
}

int FEValues::edge_face_to_cell_index(int f, int i) {
  assert(f < TRIANGLES_PER_TET && i < EDGES_PER_TRIANGLE);

  return fe_[f][i];
}

void FEValues::get_cell_dof_indices(std::vector<PetscInt> &indices) const {
  int e;

  indices.resize(EDGES_PER_TET);
  for (e = 0; e < EDGES_PER_TET; ++e) {
    indices[e] = tet_.edge_index(e);
  }
}

void FEValues::get_face_dof_indices(int f, std::vector<PetscInt> &indices) const {
  int e;

  assert(f < TRIANGLES_PER_TET);

  indices.resize(EDGES_PER_TRIANGLE);
  for (e = 0; e < EDGES_PER_TRIANGLE; ++e) {
    indices[e] = tet_.edge_index(fe_[f][e]);
  }
}

void FEValues::generate_meta_info() {
  int i, j, k;

  for (i = 0; i < EDGES_PER_TET; ++i) {
    for (j = 0; j < VERTICES_PER_EDGE; ++j) {
      for (k = 0; k < VERTICES_PER_TET; ++k) {
        if (tet_.edge_end_point(i, j) == k) {
          ev_[i][j] = k;
          break;
        }
      }
    }
  }

  for (i = 0; i < TRIANGLES_PER_TET; ++i) {
    for (j = 0; j < VERTICES_PER_TRIANGLE; ++j) {
      fv_[i][j] = (i + j + 1) % VERTICES_PER_TET;
    }
  }

  for (i = 0; i < TRIANGLES_PER_TET; ++i) {
    for (j = 0; j < EDGES_PER_TRIANGLE; ++j) {
      for (k = 0; k < EDGES_PER_TET; ++k) {
        if ((fv_[i][j] == ev_[k][0] && fv_[i][(j + 1) % 3] == ev_[k][1]) ||
            (fv_[i][j] == ev_[k][1] && fv_[i][(j + 1) % 3] == ev_[k][0])) {
          fe_[i][j] = k;
        }
      }
    }
  }
}

void FEValues::calculate_coefs() {
  int i;
  Eigen::Matrix4d jac;

  for (i = 0; i < VERTICES_PER_TET; ++i) {
    jac(0, i) = 1.0;
    jac.col(i).tail(3) = tet_.vertex(i);
  }

  jac = jac.inverse().eval();

  a_ = jac.col(0);
  b_ = jac.col(1);
  c_ = jac.col(2);
  d_ = jac.col(3);
}
