#include "em_ctx.h"
#include "em_mesh.h"
#include "em_utils.h"

#include <petsc.h>

PetscErrorCode create_context(EMContext *ctx) {
  PetscInt i;
  std::vector<PetscInt> np_per_group, proc_range;

  PetscFunctionBegin;

  PetscCall(MPI_Comm_dup(MPI_COMM_WORLD, &ctx->world_comm));
  PetscCall(MPI_Comm_size(ctx->world_comm, &ctx->world_size));
  PetscCall(MPI_Comm_rank(ctx->world_comm, &ctx->world_rank));

  if (ctx->n_groups > ctx->world_size) {
    SETERRQ(ctx->world_comm, EM_ERR_USER, "Number of groups should be less than the total number of MPI processes.");
  }

  proc_range.resize(ctx->n_groups + 1);
  np_per_group.resize(ctx->n_groups);

  proc_range[0] = 0;
  for (i = 0; i < ctx->n_groups; ++i) {
    np_per_group[i] = ctx->world_size / ctx->n_groups;
    if (i < (ctx->world_size % ctx->n_groups)) {
      np_per_group[i] += 1;
    }
    proc_range[i + 1] = proc_range[i] + np_per_group[i];
  }

  ctx->group_id = 0;
  for (i = 0; i < ctx->n_groups; ++i) {
    if (ctx->world_rank >= proc_range[i] && ctx->world_rank < proc_range[i + 1]) {
      ctx->group_id = i;
      break;
    }
  }
  PetscCall(MPI_Comm_split(ctx->world_comm, ctx->group_id, ctx->world_rank, &ctx->group_comm));

  PetscCall(MPI_Comm_size(ctx->group_comm, &ctx->group_size));
  PetscCall(MPI_Comm_rank(ctx->group_comm, &ctx->group_rank));

  ctx->coarse_mesh.reset(new Mesh(ctx->group_comm));
  ctx->mesh.reset(new Mesh(ctx->group_comm));

  ctx->C = NULL;
  ctx->M = NULL;
  ctx->A = NULL;
  ctx->B = NULL;
  ctx->A_ksp = NULL;
  ctx->B_ksp = NULL;

  ctx->G = NULL;

  ctx->s.re = NULL;
  ctx->s.im = NULL;
  ctx->csem_e.re = NULL;
  ctx->csem_e.im = NULL;
  ctx->dual_e.re = NULL;
  ctx->dual_e.im = NULL;
  ctx->mt_e[XY_POLAR].re = NULL;
  ctx->mt_e[XY_POLAR].im = NULL;
  ctx->mt_e[YX_POLAR].re = NULL;
  ctx->mt_e[YX_POLAR].im = NULL;
  ctx->w = NULL;

  PetscCall(PetscClassIdRegister("EMCTX", &ctx->EMCTX_ID));
  PetscCall(PetscLogEventRegister("CreateLS", ctx->EMCTX_ID, &ctx->CreateLS));
  PetscCall(PetscLogEventRegister("AssembleMat", ctx->EMCTX_ID, &ctx->AssembleMat));
  PetscCall(PetscLogEventRegister("AssembleRHS", ctx->EMCTX_ID, &ctx->AssembleRHS));
  PetscCall(PetscLogEventRegister("SetupAMS", ctx->EMCTX_ID, &ctx->SetupAMS));
  PetscCall(PetscLogEventRegister("CreatePC", ctx->EMCTX_ID, &ctx->CreatePC));
  PetscCall(PetscLogEventRegister("SolveLS", ctx->EMCTX_ID, &ctx->SolveLS));
  PetscCall(PetscLogEventRegister("EstimateError", ctx->EMCTX_ID, &ctx->EstimateError));
  PetscCall(PetscLogEventRegister("RefineMesh", ctx->EMCTX_ID, &ctx->RefineMesh));
  PetscCall(PetscLogEventRegister("CalculateRSP", ctx->EMCTX_ID, &ctx->CalculateRSP));
  PetscCall(PetscLogDefaultBegin());

  PetscCall(PetscViewerASCIIOpen(ctx->group_comm, string_format("%s-group-%03d.log", ctx->oprefix, ctx->group_id).c_str(), &ctx->LS_log));

  PetscFunctionReturn(0);
}

PetscErrorCode destroy_context(EMContext *ctx) {
  PetscFunctionBegin;

  ctx->mesh = NULL;

  PetscCall(PetscLogView(ctx->LS_log));
  PetscCall(PetscViewerDestroy(&ctx->LS_log));

  PetscCall(MPI_Comm_free(&ctx->group_comm));
  PetscCall(MPI_Comm_free(&ctx->world_comm));

  PetscFunctionReturn(0);
}

PetscErrorCode process_options(EMContext *ctx) {
  PetscBool flg;
  const char *MeshFormat[] = {"mdl", "mesh"};
  const char *InnerPCType[] = {"mixed", "ams", "direct"};
  const char *DirectSolverType[] = {"mumps", "superlu_dist"};
  const char *RefineStrategy[] = {"fixed_number", "fixed_fraction"};

  PetscFunctionBegin;

  PetscOptionsBegin(MPI_COMM_WORLD, "", "EMFEM options", "");

  PetscCall(PetscOptionsGetString(NULL, NULL, "-iprefix", ctx->iprefix, sizeof(ctx->iprefix), &flg));
  if (!flg) {
    SETERRQ(MPI_COMM_WORLD, EM_ERR_USER, "Plase specify the prefix of the input file.");
  }

  PetscCall(PetscOptionsGetString(NULL, NULL, "-oprefix", ctx->oprefix, sizeof(ctx->oprefix), &flg));
  if (!flg) {
    SETERRQ(MPI_COMM_WORLD, EM_ERR_USER, "Plase specify the prefix of the output file.");
  }

  ctx->n_groups = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_groups", &ctx->n_groups, &flg));

  ctx->max_tx_edge_length = -1;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-max_tx_edge_length", &ctx->max_tx_edge_length, &flg));

  ctx->max_rx_edge_length = -1;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-max_rx_edge_length", &ctx->max_rx_edge_length, &flg));

  ctx->mesh_format = MDL;
  PetscCall(PetscOptionsEList("-mesh_format", "", "", MeshFormat, sizeof(MeshFormat) / sizeof(MeshFormat[0]), MeshFormat[ctx->mesh_format], &ctx->mesh_format, &flg));

  ctx->inner_pc_type = Mixed;
  PetscCall(PetscOptionsEList("-inner_pc_type", "", "", InnerPCType, sizeof(InnerPCType) / sizeof(InnerPCType[0]), InnerPCType[ctx->inner_pc_type], &ctx->inner_pc_type, &flg));

  ctx->pc_threshold = 500000;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-pc_threshold", &ctx->pc_threshold, &flg));

  ctx->direct_solver_type = MUMPS;
  PetscCall(PetscOptionsEList("-direct_solver_type", "", "", DirectSolverType, sizeof(DirectSolverType) / sizeof(DirectSolverType[0]), DirectSolverType[ctx->direct_solver_type], &ctx->direct_solver_type, &flg));

  ctx->K_max_it = 100;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-K_max_it", &ctx->K_max_it, &flg));

  ctx->e_rtol = 1.0E-8;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-e_rtol", &ctx->e_rtol, &flg));

  ctx->dual_rtol = 1.0E-6;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dual_rtol", &ctx->dual_rtol, &flg));

  ctx->max_adaptive_refinements = 0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_adaptive_refinements", &ctx->max_adaptive_refinements, &flg));

  ctx->n_uniform_refinements = 0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_uniform_refinements", &ctx->n_uniform_refinements, &flg));

  ctx->max_dofs = 1000000;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_dofs", &ctx->max_dofs, &flg));

  ctx->refine_fraction = 0.1;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-refine_fraction", &ctx->refine_fraction, &flg));

  ctx->refine_strategy = FixedFraction;
  PetscCall(PetscOptionsEList("-refine_strategy", "", "", RefineStrategy, sizeof(RefineStrategy) / sizeof(RefineStrategy[0]), RefineStrategy[ctx->refine_strategy], &ctx->refine_strategy, &flg));

  ctx->save_mesh = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-save_mesh", &ctx->save_mesh, &flg));

  ctx->n_tx_divisions = 20;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_tx_divisions", &ctx->n_tx_divisions, &flg));

  PetscOptionsEnd();

  PetscFunctionReturn(0);
}
