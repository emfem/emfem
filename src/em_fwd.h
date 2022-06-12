#ifndef _EM_FWD_H_
#define _EM_FWD_H_ 1

#include "em_defs.h"
#include "em_mesh.h"
#include "em_pw.h"

struct EMContext;

PetscErrorCode get_cell_attribute(EMContext *, const TetAccessor &, Tensor &);

PetscErrorCode extract_locally_owned_vertices(EMContext *);

PetscErrorCode extract_locally_owned_edges(EMContext *);
PetscErrorCode extract_locally_relevant_edges(EMContext *);

PetscErrorCode make_sparsity_patterns(EMContext *, std::vector<PetscInt> &,
                                      std::vector<PetscInt> &);

PetscErrorCode create_linear_system(EMContext *, int);
PetscErrorCode destroy_linear_system(EMContext *);

PetscErrorCode generate_layered_model(EMContext *, const Point &, const Point &, const Point &,
                                      Eigen::VectorXd &, Eigen::VectorXd &);
PetscErrorCode update_background_model(EMContext *);

PetscErrorCode calculate_boundary_values(EMContext *, int, int, std::map<int, Complex> &);

PetscErrorCode divide_line_source(const double *, int,
                                  std::vector<std::tuple<Point, VectorD, double, double>> &);

PetscErrorCode assemble_matrix(EMContext *, int);
PetscErrorCode assemble_rhs_mt(EMContext *, int, int);
PetscErrorCode assemble_rhs_csem(EMContext *, int, int);
PetscErrorCode assemble_rhs_dual(EMContext *);

PetscErrorCode interpolate_fields(EMContext *, const TetAccessor &, const Point &, double, int,
                                  VectorZ &, VectorZ &);
PetscErrorCode calculate_response(EMContext *, int);

PetscErrorCode refine_tx_area(EMContext *, int);
PetscErrorCode refine_rx_area(EMContext *);

PetscErrorCode integrate_over_cell(EMContext *, const TetAccessor &, int ,
                                   const std::vector<PETScBlockVector *> &, std::vector<Eigen::VectorXd> &);
PetscErrorCode integrate_over_face(EMContext *, const TetAccessor &, int ,
                                   const std::vector<PETScBlockVector *> &, std::vector<Eigen::MatrixXd> &);
PetscErrorCode estimate_error(EMContext *, int, int);

PetscErrorCode refine_fixed_fraction(const Eigen::VectorXd &, std::vector<bool> &, double);
PetscErrorCode refine_fixed_number(const Eigen::VectorXd &, std::vector<bool> &, double);
PetscErrorCode refine_mesh(EMContext *, int);

PetscErrorCode em_forward(EMContext *);

#endif
