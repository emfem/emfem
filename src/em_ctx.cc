#include "em_ctx.h"
#include "em_mesh.h"
#include "em_utils.h"

#include <petsc.h>

PetscErrorCode create_context(EMContext *ctx) {
  PetscInt i;
  PetscErrorCode ierr;
  std::vector<PetscInt> np_per_group, proc_range;

  PetscFunctionBegin;

  ierr = MPI_Comm_dup(MPI_COMM_WORLD, &ctx->world_comm); CHKERRQ(ierr);
  ierr = MPI_Comm_size(ctx->world_comm, &ctx->world_size); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(ctx->world_comm, &ctx->world_rank); CHKERRQ(ierr);

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
  ierr = MPI_Comm_split(ctx->world_comm, ctx->group_id, ctx->world_rank, &ctx->group_comm); CHKERRQ(ierr);

  ierr = MPI_Comm_size(ctx->group_comm, &ctx->group_size); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(ctx->group_comm, &ctx->group_rank); CHKERRQ(ierr);

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

  ierr = PetscClassIdRegister("EMCTX", &ctx->EMCTX_ID); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CreateLS", ctx->EMCTX_ID, &ctx->CreateLS); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("AssembleMat", ctx->EMCTX_ID, &ctx->AssembleMat); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("AssembleRHS", ctx->EMCTX_ID, &ctx->AssembleRHS); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SetupAMS", ctx->EMCTX_ID, &ctx->SetupAMS); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CreatePC", ctx->EMCTX_ID, &ctx->CreatePC); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SolveLS", ctx->EMCTX_ID, &ctx->SolveLS); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("EstimateError", ctx->EMCTX_ID, &ctx->EstimateError); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("RefineMesh", ctx->EMCTX_ID, &ctx->RefineMesh); CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CalculateRSP", ctx->EMCTX_ID, &ctx->CalculateRSP); CHKERRQ(ierr);
  ierr = PetscLogDefaultBegin(); CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(ctx->group_comm, string_format("%s-group-%03d.log", ctx->oprefix, ctx->group_id).c_str(), &ctx->LS_log); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode destroy_context(EMContext *ctx) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ctx->mesh = NULL;

  ierr = PetscLogView(ctx->LS_log); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ctx->LS_log); CHKERRQ(ierr);

  ierr = MPI_Comm_free(&ctx->group_comm); CHKERRQ(ierr);
  ierr = MPI_Comm_free(&ctx->world_comm); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode process_options(EMContext *ctx) {
  PetscBool flg;
  PetscErrorCode ierr;
  const char *MeshFormat[] = {"mdl", "mesh"};
  const char *InnerPCType[] = {"mixed", "ams", "direct"};
  const char *DirectSolverType[] = {"mumps", "superlu_dist"};
  const char *RefineStrategy[] = {"fixed_number", "fixed_fraction"};

  PetscFunctionBegin;

  PetscOptionsBegin(MPI_COMM_WORLD, "", "EMFEM options", "");

  ierr = PetscOptionsGetString(NULL, NULL, "-iprefix", ctx->iprefix, sizeof(ctx->iprefix), &flg); CHKERRQ(ierr);
  if (!flg) {
    SETERRQ(MPI_COMM_WORLD, EM_ERR_USER, "Plase specify the prefix of the input file.");
  }

  ierr = PetscOptionsGetString(NULL, NULL, "-oprefix", ctx->oprefix, sizeof(ctx->oprefix), &flg); CHKERRQ(ierr);
  if (!flg) {
    SETERRQ(MPI_COMM_WORLD, EM_ERR_USER, "Plase specify the prefix of the output file.");
  }

  ctx->n_groups = 1;
  ierr = PetscOptionsGetInt(NULL, NULL, "-n_groups", &ctx->n_groups, &flg); CHKERRQ(ierr);

  ctx->max_tx_edge_length = -1;
  ierr = PetscOptionsGetReal(NULL, NULL, "-max_tx_edge_length", &ctx->max_tx_edge_length, &flg); CHKERRQ(ierr);

  ctx->max_rx_edge_length = -1;
  ierr = PetscOptionsGetReal(NULL, NULL, "-max_rx_edge_length", &ctx->max_rx_edge_length, &flg); CHKERRQ(ierr);

  ctx->mesh_format = MDL;
  ierr = PetscOptionsEList("-mesh_format", "", "", MeshFormat, sizeof(MeshFormat) / sizeof(MeshFormat[0]), MeshFormat[ctx->mesh_format], &ctx->mesh_format, &flg); CHKERRQ(ierr);

  ctx->inner_pc_type = Mixed;
  ierr = PetscOptionsEList("-inner_pc_type", "", "", InnerPCType, sizeof(InnerPCType) / sizeof(InnerPCType[0]), InnerPCType[ctx->inner_pc_type], &ctx->inner_pc_type, &flg); CHKERRQ(ierr);

  ctx->pc_threshold = 500000;
  ierr = PetscOptionsGetInt(NULL, NULL, "-pc_threshold", &ctx->pc_threshold, &flg); CHKERRQ(ierr);

  ctx->direct_solver_type = MUMPS;
  ierr = PetscOptionsEList("-direct_solver_type", "", "", DirectSolverType, sizeof(DirectSolverType) / sizeof(DirectSolverType[0]), DirectSolverType[ctx->direct_solver_type], &ctx->direct_solver_type, &flg); CHKERRQ(ierr);

  ctx->K_max_it = 100;
  ierr = PetscOptionsGetInt(NULL, NULL, "-K_max_it", &ctx->K_max_it, &flg); CHKERRQ(ierr);

  ctx->e_rtol = 1.0E-8;
  ierr = PetscOptionsGetReal(NULL, NULL, "-e_rtol", &ctx->e_rtol, &flg); CHKERRQ(ierr);

  ctx->dual_rtol = 1.0E-6;
  ierr = PetscOptionsGetReal(NULL, NULL, "-dual_rtol", &ctx->dual_rtol, &flg); CHKERRQ(ierr);

  ctx->max_adaptive_refinements = 0;
  ierr = PetscOptionsGetInt(NULL, NULL, "-max_adaptive_refinements", &ctx->max_adaptive_refinements, &flg); CHKERRQ(ierr);

  ctx->n_uniform_refinements = 0;
  ierr = PetscOptionsGetInt(NULL, NULL, "-n_uniform_refinements", &ctx->n_uniform_refinements, &flg); CHKERRQ(ierr);

  ctx->max_dofs = 1000000;
  ierr = PetscOptionsGetInt(NULL, NULL, "-max_dofs", &ctx->max_dofs, &flg); CHKERRQ(ierr);

  ctx->refine_fraction = 0.1;
  ierr = PetscOptionsGetReal(NULL, NULL, "-refine_fraction", &ctx->refine_fraction, &flg); CHKERRQ(ierr);

  ctx->refine_strategy = FixedFraction;
  ierr = PetscOptionsEList("-refine_strategy", "", "", RefineStrategy, sizeof(RefineStrategy) / sizeof(RefineStrategy[0]), RefineStrategy[ctx->refine_strategy], &ctx->refine_strategy, &flg); CHKERRQ(ierr);

  ctx->save_mesh = PETSC_TRUE;
  ierr = PetscOptionsGetBool(NULL, NULL, "-save_mesh", &ctx->save_mesh, &flg); CHKERRQ(ierr);

  ctx->n_tx_divisions = 20;
  ierr = PetscOptionsGetInt(NULL, NULL, "-n_tx_divisions", &ctx->n_tx_divisions, &flg); CHKERRQ(ierr);

  PetscOptionsEnd();

  PetscFunctionReturn(0);
}
