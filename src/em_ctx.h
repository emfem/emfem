#ifndef _EM_CTX_H_
#define _EM_CTX_H_ 1

#include "em_defs.h"

#include <petsc.h>

#include <set>
#include <map>
#include <memory>
#include <vector>

class Mesh;

struct EMContext {
  MPI_Comm world_comm, group_comm;
  int world_size, world_rank, group_size, group_rank, group_id;

  std::vector<Point> rx;
  std::vector<double> tx;
  std::vector<double> freqs;

  Eigen::VectorXcd rsp, obs, obserr;
  Eigen::VectorXi otype, fidx, tidx, ridx;

  Point top_corners[4];
  Eigen::VectorXd ztop[4], lsig[4];

  std::shared_ptr<Mesh> original_mesh, mesh;

  std::vector<int> relevant_edges;
  std::pair<int, int> local_vertices, local_edges;

  std::set<int> bdr_cells;

  int aniso_form;
  Eigen::MatrixXd rho;
  Eigen::VectorXd lb, ub;
  std::vector<std::string> rho_name;

  Vec w;
  Mat C, M, A, B;
  KSP A_ksp, B_ksp;
  PETScBlockVector s, csem_e, dual_e;
  std::map<int, PETScBlockVector> mt_e;

  PetscViewer LS_log;

  Eigen::VectorXd csem_error;
  std::map<int, Eigen::VectorXd> mt_error;

  Mat G;
  std::vector<PetscReal> v_coords;

  PetscBool use_ams;
  char iprefix[256], oprefix[256];
  PetscReal max_rx_edge_length, refine_fraction, e_rtol, dual_rtol;
  PetscInt max_adaptive_refinements, max_dofs, refine_strategy, n_groups, K_max_it, pc_threshold, inner_pc_type, direct_solver_type;

  PetscClassId EMCTX_ID;
  PetscLogEvent CreateLS, AssembleMat, AssembleRHS, SetupAMS, CreatePC, SolveLS, EstimateError,
      RefineMesh, CalculateRSP;
};

PetscErrorCode create_context(EMContext *);
PetscErrorCode destroy_context(EMContext *);
PetscErrorCode process_options(EMContext *);

#endif
