#include "em_utils.h"
#include "em_ctx.h"
#include "em_la.h"
#include "tetgen.h"
#include "em_mesh.h"
#include "em_fwd.h"

PetscErrorCode access_vec(Vec v, std::vector<PetscInt> &ghosts, int idx, double *value) {
  Vec lv;
  PetscReal *ptr;
  PetscErrorCode ierr;
  PetscInt begin, end, gidx;

  PetscFunctionBegin;

  ierr = VecGetOwnershipRange(v, &begin, &end);

  ierr = VecGhostGetLocalForm(v, &lv); CHKERRQ(ierr);
  ierr = VecGetArray(lv, &ptr); CHKERRQ(ierr);

  if (idx >= begin && idx < end) {
    *value = ptr[idx - begin];
  } else {
    gidx = std::lower_bound(ghosts.begin(), ghosts.end(), idx) - ghosts.begin();
    if (gidx >= (int)ghosts.size()) {
      SETERRQ(PetscObjectComm((PetscObject)v), EM_ERR_USER, string_format("Index %d is not stored in this Vec.", idx).c_str());
    }
    *value = ptr[end - begin + gidx];
  }

  ierr = VecRestoreArray(lv, &ptr); CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(v, &lv); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode solve_linear_system(EMContext *ctx, const PETScBlockVector &s, PETScBlockVector &e, PetscInt max_it, PetscReal rtol) {
  PetscErrorCode ierr;
  Vec xx[2], bb[2], x, b;

  PetscFunctionBegin;

  LogEventHelper leh(ctx->SolveLS);

  xx[0] = e.re;
  xx[1] = e.im;
  ierr = VecCreateNest(ctx->group_comm, 2, NULL, xx, &x); CHKERRQ(ierr);
  bb[0] = s.re;
  bb[1] = s.im;
  ierr = VecCreateNest(ctx->group_comm, 2, NULL, bb, &b); CHKERRQ(ierr);

  ierr = VecZeroEntries(x); CHKERRQ(ierr);

  ierr = KSPSetTolerances(ctx->A_ksp, rtol, PETSC_DEFAULT, PETSC_DEFAULT, max_it); CHKERRQ(ierr);
  ierr = KSPSolve(ctx->A_ksp, b, x); CHKERRQ(ierr);

  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = VecDestroy(&b); CHKERRQ(ierr);

  ierr = VecGhostUpdateBegin(e.re, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(e.re, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(e.im, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(e.im, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode matshell_mult_a(Mat A, Vec x, Vec y) {
  EMContext *ctx;
  Vec xr, xi, yr, yi;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = VecNestGetSubVec(x, 0, &xr); CHKERRQ(ierr);
  ierr = VecNestGetSubVec(x, 1, &xi); CHKERRQ(ierr);
  ierr = VecNestGetSubVec(y, 0, &yr); CHKERRQ(ierr);
  ierr = VecNestGetSubVec(y, 1, &yi); CHKERRQ(ierr);

  ierr = MatShellGetContext(A, &ctx); CHKERRQ(ierr);

  ierr = MatMult(ctx->C, xr, yr); CHKERRQ(ierr);
  ierr = MatMult(ctx->M, xi, ctx->w); CHKERRQ(ierr);
  ierr = VecAXPY(yr, -1.0, ctx->w); CHKERRQ(ierr);
  ierr = MatMult(ctx->M, xr, yi); CHKERRQ(ierr);
  ierr = VecScale(yi, -1.0); CHKERRQ(ierr);
  ierr = MatMult(ctx->C, xi, ctx->w); CHKERRQ(ierr);
  ierr = VecAXPY(yi, -1.0, ctx->w); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode matshell_createvecs_a(Mat A, Vec *right, Vec *left) {
  Vec xx[2], x;
  EMContext *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = MatShellGetContext(A, &ctx); CHKERRQ(ierr);

  xx[0] = ctx->w;
  xx[1] = ctx->w;
  ierr = VecCreateNest(ctx->group_comm, 2, NULL, xx, &x); CHKERRQ(ierr);

  if (right) {
    ierr = VecDuplicate(x, right); CHKERRQ(ierr);
  }
  if (left) {
    ierr = VecDuplicate(x, left); CHKERRQ(ierr);
  }

  ierr = VecDestroy(&x); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode create_pc(EMContext *ctx) {
  PetscInt nrow;
  PC A_pc, B_pc;
  PetscErrorCode ierr;
  PetscViewerAndFormat *vf;

  PetscFunctionBegin;

  LogEventHelper leh(ctx->CreatePC);

  ierr = MatDuplicate(ctx->C, MAT_COPY_VALUES, &ctx->B); CHKERRQ(ierr);
  ierr = MatAXPY(ctx->B, 1.0, ctx->M, SAME_NONZERO_PATTERN); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(ctx->B), "B"); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(ctx->B, "B_"); CHKERRQ(ierr);
  ierr = MatSetOption(ctx->B, MAT_SPD, PETSC_TRUE); CHKERRQ(ierr);
  ierr = MatSetFromOptions(ctx->B); CHKERRQ(ierr);

  ierr = KSPCreate(ctx->group_comm, &ctx->B_ksp); CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ctx->B_ksp, "B_"); CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->B_ksp, ctx->B, ctx->B); CHKERRQ(ierr);

  if (ctx->use_ams) {
    ierr = KSPSetNormType(ctx->B_ksp, KSP_NORM_UNPRECONDITIONED); CHKERRQ(ierr);
  }

  ierr = KSPGetPC(ctx->B_ksp, &B_pc); CHKERRQ(ierr);
  if (ctx->use_ams) {
    ierr = setup_ams(ctx); CHKERRQ(ierr);
    ierr = KSPSetType(ctx->B_ksp, KSPFCG); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ctx->B_ksp, 1E-2, PETSC_DEFAULT, PETSC_DEFAULT, 100); CHKERRQ(ierr);
    ierr = PCSetType(B_pc, PCHYPRE); CHKERRQ(ierr);
    ierr = PCHYPRESetType(B_pc, "ams"); CHKERRQ(ierr);
    ierr = PCHYPRESetDiscreteGradient(B_pc, ctx->G); CHKERRQ(ierr);
    ierr = PCSetCoordinates(B_pc, 3, (PetscInt)ctx->v_coords.size() / 3, &ctx->v_coords[0]); CHKERRQ(ierr);
  } else {
    ierr = KSPSetType(ctx->B_ksp, KSPPREONLY); CHKERRQ(ierr);
    ierr = PCSetType(B_pc, PCCHOLESKY); CHKERRQ(ierr);
    if (ctx->direct_solver_type == MUMPS) {
      ierr = PCFactorSetMatSolverType(B_pc, MATSOLVERMUMPS); CHKERRQ(ierr);
    } else if (ctx->direct_solver_type == SUPERLUDIST) {
      ierr = PCFactorSetMatSolverType(B_pc, MATSOLVERSUPERLU_DIST); CHKERRQ(ierr);
    }
  }
  ierr = PCSetFromOptions(B_pc); CHKERRQ(ierr);
  ierr = PCSetUp(B_pc); CHKERRQ(ierr);

  if (ctx->use_ams) {
    ierr = PetscViewerAndFormatCreate(ctx->LS_log, PETSC_VIEWER_DEFAULT, &vf); CHKERRQ(ierr);
    ierr = KSPMonitorSet(ctx->B_ksp, (PetscErrorCode(*)(KSP, PetscInt, PetscReal, void *))KSPMonitorTrueResidual, vf, (PetscErrorCode(*)(void **))PetscViewerAndFormatDestroy); CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ctx->B_ksp); CHKERRQ(ierr);
  ierr = KSPSetUp(ctx->B_ksp); CHKERRQ(ierr);

  nrow = ctx->local_edges.second - ctx->local_edges.first;
  ierr = MatCreateShell(ctx->group_comm, nrow * 2, nrow * 2, PETSC_DECIDE, PETSC_DECIDE, ctx, &ctx->A);
  ierr = MatShellSetOperation(ctx->A, MATOP_MULT, (void(*)(void))matshell_mult_a); CHKERRQ(ierr);
  ierr = MatShellSetOperation(ctx->A, MATOP_MULT_TRANSPOSE, (void(*)(void))matshell_mult_a); CHKERRQ(ierr);
  ierr = MatShellSetOperation(ctx->A, MATOP_CREATE_VECS, (void(*)(void))matshell_createvecs_a); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(ctx->A), "A"); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(ctx->A, "A_"); CHKERRQ(ierr);
  ierr = MatSetFromOptions(ctx->A); CHKERRQ(ierr);

  ierr = KSPCreate(ctx->group_comm, &ctx->A_ksp); CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ctx->A_ksp, "A_"); CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->A_ksp, ctx->A, ctx->A); CHKERRQ(ierr);
  ierr = KSPSetType(ctx->A_ksp, KSPFGMRES); CHKERRQ(ierr);
  ierr = KSPGetPC(ctx->A_ksp, &A_pc); CHKERRQ(ierr);
  ierr = PCSetType(A_pc, PCSHELL); CHKERRQ(ierr);
  ierr = PCShellSetContext(A_pc, ctx); CHKERRQ(ierr);
  ierr = PCShellSetApply(A_pc, pc_apply_b); CHKERRQ(ierr);

  ierr = PetscViewerAndFormatCreate(ctx->LS_log, PETSC_VIEWER_DEFAULT, &vf); CHKERRQ(ierr);
  ierr = KSPMonitorSet(ctx->A_ksp, (PetscErrorCode(*)(KSP, PetscInt, PetscReal, void *))KSPMonitorTrueResidual, vf, (PetscErrorCode(*)(void **))PetscViewerAndFormatDestroy); CHKERRQ(ierr);

  ierr = KSPSetFromOptions(ctx->A_ksp); CHKERRQ(ierr);
  ierr = KSPSetUp(ctx->A_ksp); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode destroy_pc(EMContext *ctx) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (ctx->use_ams) {
    ierr = destroy_ams(ctx); CHKERRQ(ierr);
  }

  ierr = MatDestroy(&ctx->A); CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->B); CHKERRQ(ierr);
  ierr = KSPDestroy(&ctx->A_ksp); CHKERRQ(ierr);
  ierr = KSPDestroy(&ctx->B_ksp); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode pc_apply_b(PC pc, Vec b, Vec x) {
  EMContext *ctx;
  Vec br, bi, xr, xi;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PCShellGetContext(pc, (void **)&ctx); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(ctx->LS_log); CHKERRQ(ierr);

  ierr = VecNestGetSubVec(b, 0, &br); CHKERRQ(ierr);
  ierr = VecNestGetSubVec(b, 1, &bi); CHKERRQ(ierr);
  ierr = VecNestGetSubVec(x, 0, &xr); CHKERRQ(ierr);
  ierr = VecNestGetSubVec(x, 1, &xi); CHKERRQ(ierr);

  ierr = KSPSolve(ctx->B_ksp, br, xr); CHKERRQ(ierr);
  ierr = KSPSolve(ctx->B_ksp, bi, xi); CHKERRQ(ierr);

  ierr = PetscViewerASCIIPopTab(ctx->LS_log); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode setup_ams(EMContext *ctx) {
  TetAccessor tet;
  PetscErrorCode ierr;
  std::vector<PetscReal> vals;
  std::vector<PetscInt> row_ptr, col_idx;
  PetscInt t, e, eidx, v, vidx, n_local_edges, n_edges, n_local_vertices, n_vertices, begin, end;

  PetscFunctionBegin;

  LogEventHelper leh(ctx->SetupAMS);

  n_edges = ctx->mesh->n_edges();
  n_local_edges = ctx->local_edges.second - ctx->local_edges.first;

  n_vertices = ctx->mesh->n_vertices();

  row_ptr.resize(n_local_edges + 1);
  col_idx.resize(n_local_edges * 2);
  vals.resize(n_local_edges * 2);

  std::fill(row_ptr.begin(), row_ptr.end(), -1);

  begin = ctx->local_edges.first;
  end = ctx->local_edges.second;

  for (t = 0; t < ctx->mesh->n_tets(); ++t) {
    tet = TetAccessor(ctx->mesh.get(), t);
    if (!tet.is_locally_owned()) {
      continue;
    }

    for (e = 0; e < EDGES_PER_TET; ++e) {
      eidx = tet.edge_index(e);
      if (eidx >= begin && eidx < end) {
        row_ptr[eidx - begin] = (eidx - begin) * 2;
        col_idx[(eidx - begin) * 2 + 0] = tet.vertex_index(tet.edge_end_point(e, 0));
        col_idx[(eidx - begin) * 2 + 1] = tet.vertex_index(tet.edge_end_point(e, 1));
        vals[(eidx - begin) * 2 + 0] = -1;
        vals[(eidx - begin) * 2 + 1] = 1;
      }
    }
  }
  row_ptr[n_local_edges] = n_local_edges * 2;

  ierr = MatCreateMPIAIJWithArrays(ctx->group_comm, n_local_edges, PETSC_DECIDE, n_edges, n_vertices, &row_ptr[0], &col_idx[0], &vals[0], &ctx->G); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(ctx->G), "G_"); CHKERRQ(ierr);

  ierr = MatGetOwnershipRangeColumn(ctx->G, &begin, &end); CHKERRQ(ierr);
  n_local_vertices = end - begin;

  ctx->v_coords.resize(n_local_vertices * 3);

  for (t = 0; t < ctx->mesh->n_tets(); ++t) {
    tet = TetAccessor(ctx->mesh.get(), t);
    for (v = 0; v < VERTICES_PER_TET; ++v) {
      vidx = tet.vertex_index(v);
      if (vidx >= begin && vidx < end) {
        ctx->v_coords[(vidx - begin) * 3 + 0] = tet.vertex(v)[0];
        ctx->v_coords[(vidx - begin) * 3 + 1] = tet.vertex(v)[1];
        ctx->v_coords[(vidx - begin) * 3 + 2] = tet.vertex(v)[2];
      }
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode destroy_ams(EMContext *ctx) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = MatDestroy(&ctx->G); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
