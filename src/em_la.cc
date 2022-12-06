#include "em_utils.h"
#include "em_ctx.h"
#include "em_la.h"
#include "em_mesh.h"
#include "em_fwd.h"

PetscErrorCode access_vec(Vec v, std::vector<PetscInt> &ghosts, int idx, double *value) {
  Vec lv;
  PetscReal *ptr;
  PetscInt begin, end, gidx;

  PetscFunctionBegin;

  PetscCall(VecGetOwnershipRange(v, &begin, &end));

  PetscCall(VecGhostGetLocalForm(v, &lv));
  PetscCall(VecGetArray(lv, &ptr));

  if (idx >= begin && idx < end) {
    *value = ptr[idx - begin];
  } else {
    gidx = std::lower_bound(ghosts.begin(), ghosts.end(), idx) - ghosts.begin();
    if (gidx >= (int)ghosts.size()) {
      SETERRQ(PetscObjectComm((PetscObject)v), EM_ERR_USER, string_format("Index %d is not stored in this Vec.", idx).c_str());
    }
    *value = ptr[end - begin + gidx];
  }

  PetscCall(VecRestoreArray(lv, &ptr));
  PetscCall(VecGhostRestoreLocalForm(v, &lv));

  PetscFunctionReturn(0);
}

PetscErrorCode solve_linear_system(EMContext *ctx, const PETScBlockVector &s, PETScBlockVector &e, PetscInt max_it, PetscReal rtol) {
  Vec xx[2], bb[2], x, b;

  PetscFunctionBegin;

  LogEventHelper leh(ctx->SolveLS);

  xx[0] = e.re;
  xx[1] = e.im;
  PetscCall(VecCreateNest(ctx->group_comm, 2, NULL, xx, &x));
  bb[0] = s.re;
  bb[1] = s.im;
  PetscCall(VecCreateNest(ctx->group_comm, 2, NULL, bb, &b));

  PetscCall(VecZeroEntries(x));

  PetscCall(KSPSetTolerances(ctx->A_ksp, rtol, PETSC_DEFAULT, PETSC_DEFAULT, max_it));
  PetscCall(KSPSolve(ctx->A_ksp, b, x));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));

  PetscCall(VecGhostUpdateBegin(e.re, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGhostUpdateEnd(e.re, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGhostUpdateBegin(e.im, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGhostUpdateEnd(e.im, INSERT_VALUES, SCATTER_FORWARD));

  PetscFunctionReturn(0);
}

PetscErrorCode matshell_mult_a(Mat A, Vec x, Vec y) {
  EMContext *ctx;
  Vec xr, xi, yr, yi;

  PetscFunctionBegin;

  PetscCall(VecNestGetSubVec(x, 0, &xr));
  PetscCall(VecNestGetSubVec(x, 1, &xi));
  PetscCall(VecNestGetSubVec(y, 0, &yr));
  PetscCall(VecNestGetSubVec(y, 1, &yi));

  PetscCall(MatShellGetContext(A, &ctx));

  PetscCall(MatMult(ctx->C, xr, yr));
  PetscCall(MatMult(ctx->M, xi, ctx->w));
  PetscCall(VecAXPY(yr, -1.0, ctx->w));
  PetscCall(MatMult(ctx->M, xr, yi));
  PetscCall(VecScale(yi, -1.0));
  PetscCall(MatMult(ctx->C, xi, ctx->w));
  PetscCall(VecAXPY(yi, -1.0, ctx->w));

  PetscFunctionReturn(0);
}

PetscErrorCode matshell_createvecs_a(Mat A, Vec *right, Vec *left) {
  Vec xx[2], x;
  EMContext *ctx;

  PetscFunctionBegin;

  PetscCall(MatShellGetContext(A, &ctx));

  xx[0] = ctx->w;
  xx[1] = ctx->w;
  PetscCall(VecCreateNest(ctx->group_comm, 2, NULL, xx, &x));

  if (right) {
    PetscCall(VecDuplicate(x, right));
  }
  if (left) {
    PetscCall(VecDuplicate(x, left));
  }

  PetscCall(VecDestroy(&x));

  PetscFunctionReturn(0);
}

PetscErrorCode create_pc(EMContext *ctx) {
  PetscInt nrow;
  PC A_pc, B_pc;
  PetscViewerAndFormat *vf;

  PetscFunctionBegin;

  LogEventHelper leh(ctx->CreatePC);

  PetscCall(MatDuplicate(ctx->C, MAT_SHARE_NONZERO_PATTERN, &ctx->B));
  PetscCall(MatCopy(ctx->C, ctx->B, SAME_NONZERO_PATTERN));
  PetscCall(MatAXPY(ctx->B, 1.0, ctx->M, SAME_NONZERO_PATTERN));
  PetscCall(PetscObjectSetName((PetscObject)(ctx->B), "B"));
  PetscCall(MatSetOptionsPrefix(ctx->B, "B_"));
  PetscCall(MatSetOption(ctx->B, MAT_SPD, PETSC_TRUE));
  PetscCall(MatSetFromOptions(ctx->B));

  PetscCall(KSPCreate(ctx->group_comm, &ctx->B_ksp));
  PetscCall(KSPSetOptionsPrefix(ctx->B_ksp, "B_"));
  PetscCall(KSPSetOperators(ctx->B_ksp, ctx->B, ctx->B));

  if (ctx->use_ams) {
    PetscCall(KSPSetNormType(ctx->B_ksp, KSP_NORM_UNPRECONDITIONED));
  }

  PetscCall(KSPGetPC(ctx->B_ksp, &B_pc));
  if (ctx->use_ams) {
    PetscCall(setup_ams(ctx));
    PetscCall(KSPSetType(ctx->B_ksp, KSPFCG));
    PetscCall(KSPSetTolerances(ctx->B_ksp, 1E-2, PETSC_DEFAULT, PETSC_DEFAULT, 100));
    PetscCall(PCSetType(B_pc, PCHYPRE));
    PetscCall(PCHYPRESetType(B_pc, "ams"));
    PetscCall(PCHYPRESetDiscreteGradient(B_pc, ctx->G));
    PetscCall(PCSetCoordinates(B_pc, 3, (PetscInt)ctx->v_coords.size() / 3, &ctx->v_coords[0]));
  } else {
    PetscCall(KSPSetType(ctx->B_ksp, KSPPREONLY));
    PetscCall(PCSetType(B_pc, PCCHOLESKY));
    if (ctx->direct_solver_type == MUMPS) {
      PetscCall(PCFactorSetMatSolverType(B_pc, MATSOLVERMUMPS));
    } else if (ctx->direct_solver_type == SUPERLUDIST) {
      PetscCall(PCFactorSetMatSolverType(B_pc, MATSOLVERSUPERLU_DIST));
    }
  }
  PetscCall(PCSetFromOptions(B_pc));
  PetscCall(PCSetUp(B_pc));

  if (ctx->use_ams) {
    PetscCall(PetscViewerAndFormatCreate(ctx->LS_log, PETSC_VIEWER_DEFAULT, &vf));
    PetscCall(KSPMonitorSet(ctx->B_ksp, (PetscErrorCode(*)(KSP, PetscInt, PetscReal, void *))KSPMonitorTrueResidual, vf, (PetscErrorCode(*)(void **))PetscViewerAndFormatDestroy));
  }
  PetscCall(KSPSetFromOptions(ctx->B_ksp));
  PetscCall(KSPSetUp(ctx->B_ksp));

  nrow = ctx->local_edges.second - ctx->local_edges.first;
  PetscCall(MatCreateShell(ctx->group_comm, nrow * 2, nrow * 2, PETSC_DECIDE, PETSC_DECIDE, ctx, &ctx->A));
  PetscCall(MatShellSetOperation(ctx->A, MATOP_MULT, (void(*)(void))matshell_mult_a));
  PetscCall(MatShellSetOperation(ctx->A, MATOP_MULT_TRANSPOSE, (void(*)(void))matshell_mult_a));
  PetscCall(MatShellSetOperation(ctx->A, MATOP_CREATE_VECS, (void(*)(void))matshell_createvecs_a));
  PetscCall(PetscObjectSetName((PetscObject)(ctx->A), "A"));
  PetscCall(MatSetOptionsPrefix(ctx->A, "A_"));
  PetscCall(MatSetFromOptions(ctx->A));

  PetscCall(KSPCreate(ctx->group_comm, &ctx->A_ksp));
  PetscCall(KSPSetOptionsPrefix(ctx->A_ksp, "A_"));
  PetscCall(KSPSetOperators(ctx->A_ksp, ctx->A, ctx->A));
  PetscCall(KSPSetType(ctx->A_ksp, KSPFGMRES));
  PetscCall(KSPGetPC(ctx->A_ksp, &A_pc));
  PetscCall(PCSetType(A_pc, PCSHELL));
  PetscCall(PCShellSetContext(A_pc, ctx));
  PetscCall(PCShellSetApply(A_pc, pc_apply_b));

  PetscCall(PetscViewerAndFormatCreate(ctx->LS_log, PETSC_VIEWER_DEFAULT, &vf));
  PetscCall(KSPMonitorSet(ctx->A_ksp, (PetscErrorCode(*)(KSP, PetscInt, PetscReal, void *))KSPMonitorTrueResidual, vf, (PetscErrorCode(*)(void **))PetscViewerAndFormatDestroy));

  PetscCall(KSPSetFromOptions(ctx->A_ksp));
  PetscCall(KSPSetUp(ctx->A_ksp));

  PetscFunctionReturn(0);
}

PetscErrorCode destroy_pc(EMContext *ctx) {
  PetscFunctionBegin;

  if (ctx->use_ams) {
    PetscCall(destroy_ams(ctx));
  }

  PetscCall(MatDestroy(&ctx->A));
  PetscCall(MatDestroy(&ctx->B));
  PetscCall(KSPDestroy(&ctx->A_ksp));
  PetscCall(KSPDestroy(&ctx->B_ksp));

  PetscFunctionReturn(0);
}

PetscErrorCode pc_apply_b(PC pc, Vec b, Vec x) {
  EMContext *ctx;
  Vec br, bi, xr, xi;

  PetscFunctionBegin;

  PetscCall(PCShellGetContext(pc, (void **)&ctx));
  PetscCall(PetscViewerASCIIPushTab(ctx->LS_log));

  PetscCall(VecNestGetSubVec(b, 0, &br));
  PetscCall(VecNestGetSubVec(b, 1, &bi));
  PetscCall(VecNestGetSubVec(x, 0, &xr));
  PetscCall(VecNestGetSubVec(x, 1, &xi));

  PetscCall(KSPSolve(ctx->B_ksp, br, xr));
  PetscCall(KSPSolve(ctx->B_ksp, bi, xi));

  PetscCall(PetscViewerASCIIPopTab(ctx->LS_log));

  PetscFunctionReturn(0);
}

PetscErrorCode setup_ams(EMContext *ctx) {
  TetAccessor tet;
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

  begin = ctx->local_vertices.first;
  end = ctx->local_vertices.second;
  n_local_vertices = end - begin;

  PetscCall(MatCreateMPIAIJWithArrays(ctx->group_comm, n_local_edges, n_local_vertices, n_edges, n_vertices, &row_ptr[0], &col_idx[0], &vals[0], &ctx->G));
  PetscCall(PetscObjectSetName((PetscObject)(ctx->G), "G_"));

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
  PetscFunctionBegin;

  PetscCall(MatDestroy(&ctx->G));

  PetscFunctionReturn(0);
}
