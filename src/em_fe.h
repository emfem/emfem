#ifndef _EM_FE_H_
#define _EM_FE_H_ 1

#include "em_defs.h"
#include "em_mesh.h"

class FEValues {
public:
  FEValues();

  void reinit(const TetAccessor &tet);
  double value_q(int, const Point &) const;
  VectorD grad_q(int, const Point &) const;
  VectorD value_nedelec(int, const Point &) const;
  VectorD curl_nedelec(int, const Point &) const;

  double face_value_q(int, int, const Point &) const;
  VectorD face_grad_q(int, int, const Point &) const;
  VectorD face_value_nedelec(int, int, const Point &) const;
  VectorD face_curl_nedelec(int, int, const Point &) const;

  VectorD face_normal(int) const;

  int edge_face_to_cell_index(int, int);

  void get_cell_dof_indices(std::vector<PetscInt> &) const;
  void get_face_dof_indices(int, std::vector<PetscInt> &) const;

private:
  void calculate_coefs();
  void generate_meta_info();

private:
  TetAccessor tet_;
  Eigen::Vector4d a_, b_, c_, d_;
  int ev_[EDGES_PER_TET][VERTICES_PER_EDGE], fv_[TRIANGLES_PER_TET][VERTICES_PER_TRIANGLE],
      fe_[TRIANGLES_PER_TET][EDGES_PER_TRIANGLE];
};

void calculate_ref_face_quadrature_points(int, std::vector<Point> &, std::vector<double> &);
void transform_face_quadrature_points(const TetAccessor &, int, const std::vector<Point> &,
                                      const std::vector<double> &, std::vector<Point> &,
                                      std::vector<double> &);

void calculate_ref_cell_quadrature_points(int, std::vector<Point> &, std::vector<double> &);
void transform_cell_quadrature_points(const TetAccessor &, const std::vector<Point> &,
                                      const std::vector<double> &, std::vector<Point> &,
                                      std::vector<double> &);

#endif
