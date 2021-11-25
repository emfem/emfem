#include "em_pw.h"
#include "em_io.h"
#include "em_ctx.h"
#include "em_utils.h"
#include "em_la.h"

#include "em_fe.h"

#include <map>
#include <set>
#include <algorithm>
#include <functional>

PetscErrorCode get_cell_attribute(EMContext *ctx, const TetAccessor &cell, Tensor &sigma) {
  int idx;
  Tensor x_rot, y_rot, z_rot;

  PetscFunctionBegin;

  idx = cell.attribute();

  sigma.setZero();
  switch (ctx->aniso_form) {
  case Isotropic:
    sigma(0, 0) = 1.0 / ctx->rho(idx, 0);
    sigma(1, 1) = 1.0 / ctx->rho(idx, 0);
    sigma(2, 2) = 1.0 / ctx->rho(idx, 0);
    break;
  case Vertical:
    sigma(0, 0) = 1.0 / ctx->rho(idx, 0);
    sigma(1, 1) = 1.0 / ctx->rho(idx, 0);
    sigma(2, 2) = 1.0 / ctx->rho(idx, 1);
    break;
  case Triaxial:
    sigma(0, 0) = 1.0 / ctx->rho(idx, 0);
    sigma(1, 1) = 1.0 / ctx->rho(idx, 1);
    sigma(2, 2) = 1.0 / ctx->rho(idx, 2);
    break;
  case Arbitrary:
    x_rot.setZero();
    y_rot.setZero();
    z_rot.setZero();

    sigma(0, 0) = 1.0 / ctx->rho(idx, 0);
    sigma(1, 1) = 1.0 / ctx->rho(idx, 1);
    sigma(2, 2) = 1.0 / ctx->rho(idx, 2);

    x_rot(0, 0) = 1.0;
    x_rot(1, 1) = x_rot(2, 2) = std::cos(ctx->rho(idx, 3) * DTOR);
    x_rot(1, 2) = std::sin(ctx->rho(idx, 3) * DTOR);
    x_rot(2, 1) = -std::sin(ctx->rho(idx, 3) * DTOR);

    y_rot(1, 1) = 1.0;
    y_rot(0, 0) = y_rot(2, 2) = std::cos(ctx->rho(idx, 4) * DTOR);
    y_rot(0, 2) = -std::sin(ctx->rho(idx, 4) * DTOR);
    y_rot(2, 0) = std::sin(ctx->rho(idx, 4) * DTOR);

    z_rot(2, 2) = 1.0;
    z_rot(0, 0) = z_rot(1, 1) = std::cos(ctx->rho(idx, 5) * DTOR);
    z_rot(0, 1) = std::sin(ctx->rho(idx, 5) * DTOR);
    z_rot(1, 0) = -std::sin(ctx->rho(idx, 5) * DTOR);

    sigma = z_rot * y_rot * x_rot * sigma * x_rot.transpose() * y_rot.transpose() *
            z_rot.transpose();
    break;
  default:
    SETERRQ(ctx->world_comm, EM_ERR_USER, string_format("Unsupported anisotropy form %d.", ctx->aniso_form).c_str());
    break;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode generate_layered_model(EMContext *ctx, const Point &tl, const Point &br,
                                      const Point &p, Eigen::VectorXd &ztop,
                                      Eigen::VectorXd &lsig) {
  Point fp;
  Tensor sigma;
  int v, t, i, n;
  TetAccessor cell;
  PetscErrorCode ierr;
  std::vector<double> z, sig;

  PetscFunctionBegin;

  fp = p;

  while (fp[2] < br[2]) {
    v = ctx->original_mesh->find_closest_vertex(fp);
    t = ctx->original_mesh->find_cell_around_point(Point(fp[0], fp[1], fp[2] + EPS));

    cell = TetAccessor(ctx->original_mesh.get(), t);

    ierr = get_cell_attribute(ctx, cell, sigma); CHKERRQ(ierr);

    z.push_back(fp[2]);
    sig.push_back(sigma.diagonal().mean());

    for (i = 0; i < VERTICES_PER_TET; ++i) {
      if (cell.vertex_index(i) != v && std::abs(cell.vertex(i)[0] - p[0]) < EPS &&
          std::abs(cell.vertex(i)[1] - p[1]) < EPS) {
        fp = cell.vertex(i);
        break;
      }
    }
    assert(i < VERTICES_PER_TET);
  }

  n = 1;
  for (i = 0; i < (int)z.size() - 1; ++i) {
    if (sig[i] != sig[i + 1]) {
      ++n;
    }
  }

  ztop.resize(n);
  lsig.resize(n);

  n = 1;
  ztop[0] = z[0];
  lsig[0] = sig[0];

  for (i = 0; i < (int)z.size() - 1; ++i) {
    if (sig[i] != sig[i + 1]) {
      ztop[n] = z[i + 1];
      lsig[n] = sig[i + 1];
      ++n;
    }
  }

  (void)tl;

  PetscFunctionReturn(0);
}

PetscErrorCode update_background_model(EMContext *ctx) {
  PetscErrorCode ierr;
  Point p, top_left, bottom_right;

  PetscFunctionBegin;

  p[0] = p[1] = p[2] = -1.0E+15;
  top_left = ctx->original_mesh->vertex(ctx->original_mesh->find_closest_vertex(p));

  p[0] = p[1] = p[2] = 1.0E+15;
  bottom_right = ctx->original_mesh->vertex(ctx->original_mesh->find_closest_vertex(p));

  ctx->top_corners[0] = Point(top_left[0], top_left[1], top_left[2]);
  ierr = generate_layered_model(ctx, top_left, bottom_right, ctx->top_corners[0], ctx->ztop[0], ctx->lsig[0]); CHKERRQ(ierr);

  ctx->top_corners[1] = Point(bottom_right[0], top_left[1], top_left[2]);
  ierr = generate_layered_model(ctx, top_left, bottom_right, ctx->top_corners[1], ctx->ztop[1], ctx->lsig[1]); CHKERRQ(ierr);

  ctx->top_corners[2] = Point(bottom_right[0], bottom_right[1], top_left[2]);
  ierr = generate_layered_model(ctx, top_left, bottom_right, ctx->top_corners[2], ctx->ztop[2], ctx->lsig[2]); CHKERRQ(ierr);

  ctx->top_corners[3] = Point(top_left[0], bottom_right[1], top_left[2]);
  ierr = generate_layered_model(ctx, top_left, bottom_right, ctx->top_corners[3], ctx->ztop[3], ctx->lsig[3]); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode extract_locally_owned_edges(EMContext *ctx) {
  int e, begin, end;
  std::vector<int> owners;

  PetscFunctionBegin;

  ctx->mesh->get_edge_owners(owners);

  begin = std::numeric_limits<int>::max();
  end = std::numeric_limits<int>::min();

  for (e = 0; e < ctx->mesh->n_edges(); ++e) {
    if (owners[e] == ctx->group_rank) {
      begin = std::min(begin, e);
      end = std::max(end, e);
    }
  }

  if (begin > end) {
    begin = end = -1;
  } else {
    end += 1;
  }

  ctx->local_edges.first = begin;
  ctx->local_edges.second = end;

  PetscFunctionReturn(0);
}

PetscErrorCode extract_locally_relevant_edges(EMContext *ctx) {
  int t, e;
  TetAccessor cell;
  std::vector<int>::iterator vit;
  std::vector<int> edges_on_ghosts;

  PetscFunctionBegin;

  for (t = 0; t < ctx->mesh->n_tets(); ++t) {
    cell = TetAccessor(ctx->mesh.get(), t);
    if (!cell.is_ghost()) {
      continue;
    }

    for (e = 0; e < EDGES_PER_TET; ++e) {
      if (!(cell.edge_index(e) >= ctx->local_edges.first &&
            cell.edge_index(e) < ctx->local_edges.second)) {
        edges_on_ghosts.push_back(cell.edge_index(e));
      }
    }
  }

  std::sort(edges_on_ghosts.begin(), edges_on_ghosts.end());
  vit = std::unique(edges_on_ghosts.begin(), edges_on_ghosts.end());
  ctx->relevant_edges.resize(vit - edges_on_ghosts.begin());
  std::copy(edges_on_ghosts.begin(), vit, ctx->relevant_edges.begin());

  PetscFunctionReturn(0);
}

PetscErrorCode make_sparsity_patterns(EMContext *ctx, std::vector<PetscInt> &rptr,
                                      std::vector<PetscInt> &cidx) {
  TetAccessor cell;
  int t, i, j, nrow, nnz, eidx, begin, end;
  std::vector<std::set<int> > nnz_patterns;

  PetscFunctionBegin;

  begin = ctx->local_edges.first;
  end = ctx->local_edges.second;

  nrow = end - begin;
  nnz_patterns.resize(nrow);

  for (t = 0; t < ctx->mesh->n_tets(); ++t) {
    cell = TetAccessor(ctx->mesh.get(), t);
    if (cell.is_locally_owned() || cell.is_ghost()) {
      for (i = 0; i < EDGES_PER_TET; ++i) {
        eidx = cell.edge_index(i);
        if (eidx >= begin && eidx < end) {
          for (j = 0; j < EDGES_PER_TET; ++j) {
            nnz_patterns[eidx - begin].insert(cell.edge_index(j));
          }
        }
      }
    }
  }

  nnz = 0;
  for (i = begin; i < end; ++i) {
    nnz += nnz_patterns[i - begin].size();
  }

  rptr.resize(nrow + 1);
  cidx.resize(nnz);

  nnz = 0;
  for (i = begin; i < end; ++i) {
    rptr[i - begin] = nnz;
    std::copy(nnz_patterns[i - begin].begin(), nnz_patterns[i - begin].end(), cidx.begin() + nnz);
    nnz += std::distance(nnz_patterns[i - begin].begin(), nnz_patterns[i - begin].end());
  }
  rptr[i - begin] = nnz;

  PetscFunctionReturn(0);
}

PetscErrorCode create_linear_system(EMContext *ctx, int tidx) {
  int nrow;
  PetscErrorCode ierr;
  std::set<int> ghost_cells;
  std::vector<PetscInt> rptr, cidx;

  PetscFunctionBegin;

  LogEventHelper leh(ctx->CreateLS);

  ierr = extract_locally_owned_edges(ctx); CHKERRQ(ierr);
  ierr = extract_locally_relevant_edges(ctx); CHKERRQ(ierr);
  ierr = make_sparsity_patterns(ctx, rptr, cidx); CHKERRQ(ierr);

  nrow = ctx->local_edges.second - ctx->local_edges.first;

  ierr = MatCreateMPIAIJWithArrays(ctx->group_comm, nrow, nrow, PETSC_DECIDE, PETSC_DECIDE, &rptr[0], &cidx[0], NULL, &ctx->C); CHKERRQ(ierr);
  ierr = MatCreateMPIAIJWithArrays(ctx->group_comm, nrow, nrow, PETSC_DECIDE, PETSC_DECIDE, &rptr[0], &cidx[0], NULL, &ctx->M); CHKERRQ(ierr);

  ierr = VecCreateMPI(ctx->group_comm, nrow, PETSC_DECIDE, &ctx->s.re); CHKERRQ(ierr);
  ierr = VecCreateMPI(ctx->group_comm, nrow, PETSC_DECIDE, &ctx->s.im); CHKERRQ(ierr);
  ierr = VecCreateGhost(ctx->group_comm, nrow, PETSC_DECIDE, (PetscInt)ctx->relevant_edges.size(), &ctx->relevant_edges[0], &ctx->dual_e.re); CHKERRQ(ierr);
  ierr = VecCreateGhost(ctx->group_comm, nrow, PETSC_DECIDE, (PetscInt)ctx->relevant_edges.size(), &ctx->relevant_edges[0], &ctx->dual_e.im); CHKERRQ(ierr);
  if (tidx < 0) {
    ierr = VecCreateGhost(ctx->group_comm, nrow, PETSC_DECIDE, (PetscInt)ctx->relevant_edges.size(), &ctx->relevant_edges[0], &ctx->mt_e[XY_POLAR].re); CHKERRQ(ierr);
    ierr = VecCreateGhost(ctx->group_comm, nrow, PETSC_DECIDE, (PetscInt)ctx->relevant_edges.size(), &ctx->relevant_edges[0], &ctx->mt_e[XY_POLAR].im); CHKERRQ(ierr);
    ierr = VecCreateGhost(ctx->group_comm, nrow, PETSC_DECIDE, (PetscInt)ctx->relevant_edges.size(), &ctx->relevant_edges[0], &ctx->mt_e[YX_POLAR].re); CHKERRQ(ierr);
    ierr = VecCreateGhost(ctx->group_comm, nrow, PETSC_DECIDE, (PetscInt)ctx->relevant_edges.size(), &ctx->relevant_edges[0], &ctx->mt_e[YX_POLAR].im); CHKERRQ(ierr);
  } else {
    ierr = VecCreateGhost(ctx->group_comm, nrow, PETSC_DECIDE, (PetscInt)ctx->relevant_edges.size(), &ctx->relevant_edges[0], &ctx->csem_e.re); CHKERRQ(ierr);
    ierr = VecCreateGhost(ctx->group_comm, nrow, PETSC_DECIDE, (PetscInt)ctx->relevant_edges.size(), &ctx->relevant_edges[0], &ctx->csem_e.im); CHKERRQ(ierr);
  }
  ierr = VecCreateMPI(ctx->group_comm, nrow, PETSC_DECIDE, &ctx->w); CHKERRQ(ierr);

  if ((InnerPCType)ctx->inner_pc_type == Mixed) {
    if (ctx->mesh->n_edges() > ctx->pc_threshold) {
      ctx->use_ams = PETSC_TRUE;
    } else {
      ctx->use_ams = PETSC_FALSE;
    }
  } else if ((InnerPCType)ctx->inner_pc_type == AMS) {
    ctx->use_ams = PETSC_TRUE;
  } else {
    ctx->use_ams = PETSC_FALSE;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode destroy_linear_system(EMContext *ctx) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = MatDestroy(&ctx->C); CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->M); CHKERRQ(ierr);

  ierr = VecDestroy(&ctx->s.re); CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->s.im); CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->csem_e.re); CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->csem_e.im); CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->dual_e.re); CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->dual_e.im); CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->mt_e[XY_POLAR].re); CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->mt_e[XY_POLAR].im); CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->mt_e[YX_POLAR].re); CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->mt_e[YX_POLAR].im); CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->w); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode assemble_matrix(EMContext *ctx, int fidx) {
  double omega;
  Tensor sigma;
  int i, j, t, q;
  PetscErrorCode ierr;
  std::vector<PetscInt> dof_indices;
  Eigen::Matrix<PetscReal, EDGES_PER_TET, EDGES_PER_TET> c_cell_matrix, m_cell_matrix;

  FEValues fv;
  TetAccessor cell;
  std::vector<double> weights, jxw;
  std::vector<Point> ref_qpoints, qpoints;

  PetscFunctionBegin;

  LogEventHelper leh(ctx->AssembleMat);

  ctx->bdr_cells.clear();

  omega = 2 * PI * ctx->freqs[fidx];

  calculate_ref_cell_quadrature_points(2, ref_qpoints, weights);

  ierr = MatZeroEntries(ctx->C); CHKERRQ(ierr);
  ierr = MatZeroEntries(ctx->M); CHKERRQ(ierr);

  for (t = 0; t < ctx->mesh->n_tets(); ++t) {
    cell = TetAccessor(ctx->mesh.get(), t);
    if (!cell.is_locally_owned()) {
      continue;
    }

    get_cell_attribute(ctx, cell, sigma);

    fv.reinit(cell);
    fv.get_cell_dof_indices(dof_indices);

    transform_cell_quadrature_points(cell, ref_qpoints, weights, qpoints, jxw);

    c_cell_matrix.setZero();
    m_cell_matrix.setZero();
    for (i = 0; i < EDGES_PER_TET; ++i) {
      for (j = 0; j <= i; ++j) {
        for (q = 0; q < (int)qpoints.size(); ++q) {
          c_cell_matrix(i, j) +=
              (1.0 / MU) *
              fv.curl_nedelec(i, qpoints[q]).dot(fv.curl_nedelec(j, qpoints[q])) *
              jxw[q];
          m_cell_matrix(i, j) += omega *
                                 (fv.value_nedelec(i, qpoints[q])
                                      .dot(sigma * fv.value_nedelec(j, qpoints[q]))) *
                                 jxw[q];
        }
        if (i != j) {
          c_cell_matrix(j, i) = c_cell_matrix(i, j);
          m_cell_matrix(j, i) = m_cell_matrix(i, j);
        }
      }
    }

    for (i = 0; i < EDGES_PER_TET; ++i) {
      if (cell.edge_on_boundary(i)) {
        for (j = 0; j < EDGES_PER_TET; ++j) {
          if (j != i) {
            c_cell_matrix(i, j) = c_cell_matrix(j, i) = 0.0;
          }
          m_cell_matrix(i, j) = m_cell_matrix(j, i) = 0.0;
        }

        ctx->bdr_cells.insert(t);
      }
    }

    ierr = MatSetValues(ctx->C, EDGES_PER_TET, &dof_indices[0], EDGES_PER_TET, &dof_indices[0], &c_cell_matrix(0, 0), ADD_VALUES); CHKERRQ(ierr);
    ierr = MatSetValues(ctx->M, EDGES_PER_TET, &dof_indices[0], EDGES_PER_TET, &dof_indices[0], &m_cell_matrix(0, 0), ADD_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(ctx->C, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->C, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(ctx->M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode calculate_boundary_values(EMContext *ctx, int fidx, int mode,
                                         std::map<int, Complex> &bdr_values) {
  int i;
  Point ec;
  VectorD ed;
  FEValues fv;
  TetAccessor cell;
  Complex dof_value;
  double coeffs[4], area;
  VectorZ ep_values[4], ep;
  PlaneWaveFunction pwf[4];
  std::set<int>::iterator it;
  std::vector<PetscInt> dof_indices;

  PetscFunctionBegin;

  pwf[0].initialize(ctx->ztop[0], ctx->lsig[0], ctx->freqs[fidx]);
  pwf[1].initialize(ctx->ztop[1], ctx->lsig[1], ctx->freqs[fidx]);
  pwf[2].initialize(ctx->ztop[2], ctx->lsig[2], ctx->freqs[fidx]);
  pwf[3].initialize(ctx->ztop[3], ctx->lsig[3], ctx->freqs[fidx]);

  bdr_values.clear();
  for (it = ctx->bdr_cells.begin(); it != ctx->bdr_cells.end(); ++it) {
    cell = TetAccessor(ctx->mesh.get(), *it);

    fv.reinit(cell);
    fv.get_cell_dof_indices(dof_indices);

    for (i = 0; i < EDGES_PER_TET; ++i) {
      if (!cell.edge_on_boundary(i)) {
        continue;
      }

      ec = (cell.vertex(cell.edge_end_point(i, 0)) + cell.vertex(cell.edge_end_point(i, 1))) / 2.0;
      ed = (cell.vertex(cell.edge_end_point(i, 1)) - cell.vertex(cell.edge_end_point(i, 0)));

      if (std::abs(ec[0] - ctx->top_corners[0][0]) < EPS) {
        coeffs[0] = 1.0 - (1.0 - std::cos(PI * (ec[1] - ctx->top_corners[0][1]) / (ctx->top_corners[2][1] - ctx->top_corners[0][1]))) / 2.0;
        coeffs[1] = 0.0;
        coeffs[2] = 0.0;
        coeffs[3] = (1.0 - std::cos(PI * (ec[1] - ctx->top_corners[0][1]) / (ctx->top_corners[2][1] - ctx->top_corners[0][1]))) / 2.0;
      } else if (std::abs(ec[1] - ctx->top_corners[0][1]) < EPS) {
        coeffs[0] = 1.0 - (1.0 - std::cos(PI * (ec[0] - ctx->top_corners[0][0]) / (ctx->top_corners[2][0] - ctx->top_corners[0][0]))) / 2.0;
        coeffs[1] = (1.0 - std::cos(PI * (ec[0] - ctx->top_corners[0][0]) / (ctx->top_corners[2][0] - ctx->top_corners[0][0]))) / 2.0;
        coeffs[2] = 0.0;
        coeffs[3] = 0.0;
      } else if (std::abs(ec[0] - ctx->top_corners[2][0]) < EPS) {
        coeffs[0] = 0.0;
        coeffs[1] = 1.0 - (1.0 - std::cos(PI * (ec[1] - ctx->top_corners[0][1]) / (ctx->top_corners[2][1] - ctx->top_corners[0][1]))) / 2.0;
        coeffs[2] = (1.0 - std::cos(PI * (ec[1] - ctx->top_corners[0][1]) / (ctx->top_corners[2][1] - ctx->top_corners[0][1]))) / 2.0;
        coeffs[3] = 0.0;
      } else if (std::abs(ec[1] - ctx->top_corners[2][1]) < EPS) {
        coeffs[0] = 0.0;
        coeffs[1] = 0.0;
        coeffs[2] = (1.0 - std::cos(PI * (ec[0] - ctx->top_corners[0][0]) / (ctx->top_corners[2][0] - ctx->top_corners[0][0]))) / 2.0;
        coeffs[3] = 1.0 - (1.0 - std::cos(PI * (ec[0] - ctx->top_corners[0][0]) / (ctx->top_corners[2][0] - ctx->top_corners[0][0]))) / 2.0;
      } else {
        area = (ctx->top_corners[2][0] - ctx->top_corners[0][0]) * (ctx->top_corners[2][1] - ctx->top_corners[0][1]);
        coeffs[0] = ((ctx->top_corners[2][0] - ec[0]) * (ctx->top_corners[2][1] - ec[1])) / area;
        coeffs[1] = ((ec[0] - ctx->top_corners[3][0]) * (ctx->top_corners[3][1] - ec[1])) / area;
        coeffs[2] = ((ec[0] - ctx->top_corners[0][0]) * (ec[1] - ctx->top_corners[0][1])) / area;
        coeffs[3] = ((ctx->top_corners[1][0] - ec[0]) * (ec[1] - ctx->top_corners[1][1])) / area;
      }

      pwf[0].calculate_efield(ec, mode, ep_values[0]);
      pwf[1].calculate_efield(ec, mode, ep_values[1]);
      pwf[2].calculate_efield(ec, mode, ep_values[2]);
      pwf[3].calculate_efield(ec, mode, ep_values[3]);

      ep = coeffs[0] * ep_values[0] + coeffs[1] * ep_values[1] + coeffs[2] * ep_values[2] + coeffs[3] * ep_values[3];

      dof_value = ed.dot(ep) / ed.dot(fv.value_nedelec(i, ec));

      if (bdr_values.find(dof_indices[i]) == bdr_values.end()) {
        if (std::abs(dof_value) > 1E-13) {
          bdr_values[dof_indices[i]] = dof_value;
        } else {
          bdr_values[dof_indices[i]] = Complex(0.0);
        }
      }
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode assemble_rhs_mt(EMContext *ctx, int fidx, int mode) {
  int i, j, q;
  Tensor sigma;
  PetscErrorCode ierr;
  double omega, bv_re, bv_im;
  std::set<int>::iterator it;
  std::map<int, Complex> bdr_values;
  std::vector<PetscInt> dof_indices;
  Eigen::Matrix<PetscReal, EDGES_PER_TET, 1> b_re, b_im;
  Eigen::Matrix<PetscReal, EDGES_PER_TET, EDGES_PER_TET> c_cell_matrix, m_cell_matrix;

  FEValues fv;
  TetAccessor cell;
  std::vector<double> weights, jxw;
  std::vector<Point> ref_qpoints, qpoints;

  PetscFunctionBegin;

  LogEventHelper leh(ctx->AssembleRHS);

  omega = 2 * PI * ctx->freqs[fidx];

  calculate_ref_cell_quadrature_points(2, ref_qpoints, weights);

  ierr = calculate_boundary_values(ctx, fidx, mode, bdr_values); CHKERRQ(ierr);

  ierr = VecZeroEntries(ctx->s.re); CHKERRQ(ierr);
  ierr = VecZeroEntries(ctx->s.im); CHKERRQ(ierr);

  for (it = ctx->bdr_cells.begin(); it != ctx->bdr_cells.end(); ++it) {
    cell = TetAccessor(ctx->mesh.get(), *it);

    get_cell_attribute(ctx, cell, sigma);

    fv.reinit(cell);
    fv.get_cell_dof_indices(dof_indices);

    transform_cell_quadrature_points(cell, ref_qpoints, weights, qpoints, jxw);

    c_cell_matrix.setZero();
    m_cell_matrix.setZero();
    for (i = 0; i < EDGES_PER_TET; ++i) {
      for (j = 0; j <= i; ++j) {
        for (q = 0; q < (int)qpoints.size(); ++q) {
          c_cell_matrix(i, j) +=
              (1.0 / MU) *
              fv.curl_nedelec(i, qpoints[q]).dot(fv.curl_nedelec(j, qpoints[q])) *
              jxw[q];
          m_cell_matrix(i, j) += omega *
                                 (fv.value_nedelec(i, qpoints[q])
                                      .dot(sigma * fv.value_nedelec(j, qpoints[q]))) *
                                 jxw[q];
        }
        if (i != j) {
          c_cell_matrix(j, i) = c_cell_matrix(i, j);
          m_cell_matrix(j, i) = m_cell_matrix(i, j);
        }
      }
    }

    b_re.setZero();
    b_im.setZero();

    for (i = 0; i < EDGES_PER_TET; ++i) {
      if (!cell.edge_on_boundary(i)) {
        continue;
      }

      bv_re = bdr_values[dof_indices[i]].real();
      bv_im = bdr_values[dof_indices[i]].imag();

      b_re[i] = bv_re * c_cell_matrix(i, i);
      b_im[i] = -bv_im * c_cell_matrix(i, i);
      for (j = 0; j < EDGES_PER_TET; ++j) {
        if (!cell.edge_on_boundary(j)) {
          b_re[j] -= c_cell_matrix(j, i) * bv_re;
          b_im[j] += m_cell_matrix(j, i) * bv_re;
          b_re[j] += m_cell_matrix(j, i) * bv_im;
          b_im[j] += c_cell_matrix(j, i) * bv_im;
        }
      }
    }

    ierr = VecSetValues(ctx->s.re, EDGES_PER_TET, &dof_indices[0], &b_re[0], ADD_VALUES); CHKERRQ(ierr);
    ierr = VecSetValues(ctx->s.im, EDGES_PER_TET, &dof_indices[0], &b_im[0], ADD_VALUES); CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(ctx->s.re); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(ctx->s.re); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(ctx->s.im); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(ctx->s.im); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode assemble_rhs_csem(EMContext *ctx, int fidx, int tidx) {
  int i, t;
  Point tx_coord;
  PetscErrorCode ierr;
  VectorD tx_direction;
  PetscReal omega, current;
  std::vector<PetscInt> dof_indices;
  Eigen::Matrix<PetscReal, EDGES_PER_TET, 1> b_re, b_im;

  FEValues fv;
  TetAccessor cell;

  PetscFunctionBegin;

  LogEventHelper leh(ctx->AssembleRHS);

  omega = 2 * PI * ctx->freqs[fidx];
  current = ctx->tx[tidx * TX_SIZE + 5];

  tx_coord[0] = ctx->tx[tidx * TX_SIZE + 0];
  tx_coord[1] = ctx->tx[tidx * TX_SIZE + 1];
  tx_coord[2] = ctx->tx[tidx * TX_SIZE + 2];

  tx_direction[0] = std::cos(ctx->tx[tidx * TX_SIZE + 3] * DTOR) * std::cos(ctx->tx[tidx * TX_SIZE + 4] * DTOR);
  tx_direction[1] = std::sin(ctx->tx[tidx * TX_SIZE + 3] * DTOR) * std::cos(ctx->tx[tidx * TX_SIZE + 4] * DTOR);
  tx_direction[2] = std::sin(ctx->tx[tidx * TX_SIZE + 4] * DTOR);

  ierr = VecZeroEntries(ctx->s.re); CHKERRQ(ierr);
  ierr = VecZeroEntries(ctx->s.im); CHKERRQ(ierr);

  t = ctx->mesh->find_cell_around_point(tx_coord);
  if (t < 0) {
    SETERRQ(ctx->world_comm, EM_ERR_USER, string_format("Transmitter %d is not found in mesh.", tidx).c_str());
  }
  cell = TetAccessor(ctx->mesh.get(), t);

  if (cell.is_locally_owned()) {
    fv.reinit(cell);
    fv.get_cell_dof_indices(dof_indices);

    b_re.setZero();
    b_im.setZero();
    for (i = 0; i < EDGES_PER_TET; ++i) {
      b_im[i] = omega * current * fv.value_nedelec(i, tx_coord).dot(tx_direction);
    }

    ierr = VecSetValues(ctx->s.re, EDGES_PER_TET, &dof_indices[0], &b_re[0], ADD_VALUES); CHKERRQ(ierr);
    ierr = VecSetValues(ctx->s.im, EDGES_PER_TET, &dof_indices[0], &b_im[0], ADD_VALUES); CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(ctx->s.re); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(ctx->s.re); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(ctx->s.im); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(ctx->s.im); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode assemble_rhs_dual(EMContext *ctx) {
  int i, j, t;
  PetscErrorCode ierr;
  std::vector<PetscInt> dof_indices;
  Eigen::Matrix<PetscReal, EDGES_PER_TET, 1> b_re, b_im;

  FEValues fv;
  TetAccessor cell;

  PetscFunctionBegin;

  LogEventHelper leh(ctx->AssembleRHS);

  ierr = VecZeroEntries(ctx->s.re); CHKERRQ(ierr);
  ierr = VecZeroEntries(ctx->s.im); CHKERRQ(ierr);

  for (i = 0; i < (int)ctx->rx.size(); ++i) {
    t = ctx->mesh->find_cell_around_point(ctx->rx[i]);
    if (t < 0) {
      SETERRQ(ctx->world_comm, EM_ERR_USER, string_format("Receiver %d is not found in mesh.", i).c_str());
    }
    cell = TetAccessor(ctx->mesh.get(), t);

    if (!cell.is_locally_owned()) {
      continue;
    }

    fv.reinit(cell);
    fv.get_cell_dof_indices(dof_indices);

    for (j = 0; j < EDGES_PER_TET; ++j) {
      b_re[j] = fv.value_nedelec(j, ctx->rx[i]).sum();
      b_im[j] = fv.value_nedelec(j, ctx->rx[i]).sum();
    }

    ierr = VecSetValues(ctx->s.re, EDGES_PER_TET, &dof_indices[0], &b_re[0], ADD_VALUES); CHKERRQ(ierr);
    ierr = VecSetValues(ctx->s.im, EDGES_PER_TET, &dof_indices[0], &b_im[0], ADD_VALUES); CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(ctx->s.re); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(ctx->s.re); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(ctx->s.im); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(ctx->s.im); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode interpolate_fields(EMContext *ctx, const TetAccessor &cell, const Point &p,
                                  double freq, const PETScBlockVector &e, VectorZ &es, VectorZ &hs) {
  FEValues fv;
  int i, eidx;
  PetscErrorCode ierr;
  PetscReal v_re, v_im;
  std::vector<PetscInt> indices;

  PetscFunctionBegin;

  es.setZero();
  hs.setZero();

  fv.reinit(cell);
  fv.get_cell_dof_indices(indices);

  for (i = 0; i < EDGES_PER_TET; ++i) {
    eidx = indices[i];
    ierr = access_vec(e.re, ctx->relevant_edges, eidx, &v_re); CHKERRQ(ierr);
    ierr = access_vec(e.im, ctx->relevant_edges, eidx, &v_im); CHKERRQ(ierr);

    es += Complex(v_re, v_im) * fv.value_nedelec(i, p);
    hs += -1.0 / (II * 2.0 * PI * freq * MU) * Complex(v_re, v_im) * fv.curl_nedelec(i, p);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode calculate_response_mt(EMContext *ctx, int fidx) {
  int o, ridx;
  VectorZ e_xy, h_xy, e_yx, h_yx;

  TetAccessor cell;

  PetscFunctionBegin;

  LogEventHelper leh(ctx->CalculateRSP);

  for (ridx = 0; ridx < (int)ctx->rx.size(); ++ridx) {
    cell = TetAccessor(ctx->mesh.get(), ctx->mesh->find_cell_around_point(ctx->rx[ridx]));
    if (!cell.is_locally_owned()) {
      continue;
    }

    interpolate_fields(ctx, cell, ctx->rx[ridx], ctx->freqs[fidx], ctx->mt_e[XY_POLAR], e_xy, h_xy);
    interpolate_fields(ctx, cell, ctx->rx[ridx], ctx->freqs[fidx], ctx->mt_e[YX_POLAR], e_yx, h_yx);

    for (o = 0; o < ctx->rsp.size(); ++o) {
      if (ctx->fidx[o] != fidx || ctx->tidx[o] != -3 || ctx->ridx[o] != ridx) {
        continue;
      }

      switch (ctx->otype[o]) {
      case Z_XX_RI:
        ctx->rsp[o] =
            (e_xy[0] * h_yx[1] - e_yx[0] * h_xy[1]) / (h_xy[0] * h_yx[1] - h_yx[0] * h_xy[1]);
        break;
      case Z_XY_RI:
        ctx->rsp[o] =
            (e_yx[0] * h_xy[0] - e_xy[0] * h_yx[0]) / (h_xy[0] * h_yx[1] - h_yx[0] * h_xy[1]);
        break;
      case Z_YX_RI:
        ctx->rsp[o] =
            (e_xy[1] * h_yx[1] - e_yx[1] * h_xy[1]) / (h_xy[0] * h_yx[1] - h_yx[0] * h_xy[1]);
        break;
      case Z_YY_RI:
        ctx->rsp[o] =
            (e_yx[1] * h_xy[0] - e_xy[1] * h_yx[0]) / (h_xy[0] * h_yx[1] - h_yx[0] * h_xy[1]);
        break;
      case T_ZX_RI:
        ctx->rsp[o] =
            (h_xy[2] * h_yx[1] - h_yx[2] * h_xy[1]) / (h_xy[0] * h_yx[1] - h_yx[0] * h_xy[1]);
        break;
      case T_ZY_RI:
        ctx->rsp[o] =
            (h_yx[2] * h_xy[0] - h_xy[2] * h_yx[0]) / (h_xy[0] * h_yx[1] - h_yx[0] * h_xy[1]);
        break;
      default:
        SETERRQ(ctx->world_comm, EM_ERR_USER, string_format("Data type %d not implemented.", ctx->otype[o]).c_str());
      }
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode calculate_response_csem(EMContext *ctx, int fidx, int tidx) {
  int o, ridx;
  VectorZ e, h;

  TetAccessor cell;

  PetscFunctionBegin;

  LogEventHelper leh(ctx->CalculateRSP);

  for (ridx = 0; ridx < (int)ctx->rx.size(); ++ridx) {
    cell = TetAccessor(ctx->mesh.get(), ctx->mesh->find_cell_around_point(ctx->rx[ridx]));
    if (!cell.is_locally_owned()) {
      continue;
    }

    interpolate_fields(ctx, cell, ctx->rx[ridx], ctx->freqs[fidx], ctx->csem_e, e, h);

    for (o = 0; o < ctx->rsp.size(); ++o) {
      if (ctx->fidx[o] != fidx || ctx->tidx[o] != tidx || ctx->ridx[o] != ridx) {
        continue;
      }

      switch (ctx->otype[o]) {
      case F_EX_RI:
        ctx->rsp[o] = e[0];
        break;
      case F_EY_RI:
        ctx->rsp[o] = e[1];
        break;
      case F_EZ_RI:
        ctx->rsp[o] = e[2];
        break;
      case F_HX_RI:
        ctx->rsp[o] = h[0];
        break;
      case F_HY_RI:
        ctx->rsp[o] = h[1];
        break;
      case F_HZ_RI:
        ctx->rsp[o] = h[2];
        break;
      case F_EX_AP:
        ctx->rsp[o] = std::log(e[0]);
        break;
      case F_EY_AP:
        ctx->rsp[o] = std::log(e[1]);
        break;
      case F_EZ_AP:
        ctx->rsp[o] = std::log(e[2]);
        break;
      case F_HX_AP:
        ctx->rsp[o] = std::log(h[0]);
        break;
      case F_HY_AP:
        ctx->rsp[o] = std::log(h[1]);
        break;
      case F_HZ_AP:
        ctx->rsp[o] = std::log(h[2]);
        break;
      default:
        SETERRQ(ctx->world_comm, EM_ERR_USER, string_format("Data type %d for tx %d is not implemented.", ctx->otype[o], ctx->tidx[o]).c_str());
      }
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode refine_receiving_area(EMContext *ctx) {
  bool flg;
  int i, t;
  std::vector<bool> refine_flag;

  PetscFunctionBegin;

  while (true) {
    if (ctx->max_rx_edge_length <= 0.0) {
      break;
    }

    refine_flag.resize(ctx->original_mesh->n_tets());
    std::fill(refine_flag.begin(), refine_flag.end(), false);

    flg = false;
    for (i = 0; i < (int)ctx->rx.size(); ++i) {
      t = ctx->original_mesh->find_cell_around_point(ctx->rx[i]);
      if (t < 0) {
        SETERRQ(ctx->world_comm, EM_ERR_USER, string_format("Receiver %d is not found in mesh.", i).c_str());
      }
      if (TetAccessor(ctx->original_mesh.get(), t).volume() > std::pow(ctx->max_rx_edge_length, 3)) {
        flg = true;
        refine_flag[t] = true;
      }
    }

    if (!flg) {
      break;
    }

    ctx->original_mesh->refine_tetgen(refine_flag);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode calculate_function_value(EMContext *ctx, const FEValues &fv, const Point &p, const PETScBlockVector &f, VectorZ &e) {
  int i, eidx;
  double v_re, v_im;
  PetscErrorCode ierr;
  std::vector<PetscInt> indices;

  PetscFunctionBegin;

  fv.get_cell_dof_indices(indices);

  e.setZero();
  for (i = 0; i < EDGES_PER_TET; ++i) {
    eidx = indices[i];
    ierr = access_vec(f.re, ctx->relevant_edges, eidx, &v_re); CHKERRQ(ierr);
    ierr = access_vec(f.im, ctx->relevant_edges, eidx, &v_im); CHKERRQ(ierr);

    e += Complex(v_re, v_im) * fv.value_nedelec(i, p);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode calculate_function_curl(EMContext *ctx, const FEValues &fv, const Point &p, const PETScBlockVector &f, VectorZ &ce) {
  int i, eidx;
  double v_re, v_im;
  PetscErrorCode ierr;
  std::vector<PetscInt> indices;

  PetscFunctionBegin;

  fv.get_cell_dof_indices(indices);

  ce.setZero();
  for (i = 0; i < EDGES_PER_TET; ++i) {
    eidx = indices[i];
    ierr = access_vec(f.re, ctx->relevant_edges, eidx, &v_re); CHKERRQ(ierr);
    ierr = access_vec(f.im, ctx->relevant_edges, eidx, &v_im); CHKERRQ(ierr);

    ce += Complex(v_re, v_im) * fv.curl_nedelec(i, p);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode integrate_over_cell(EMContext *ctx, const TetAccessor &cell, int fidx,
                                   const std::vector<PETScBlockVector *> &solutions, std::vector<Eigen::VectorXd> &errors) {
  int i, q;
  VectorZ e;
  FEValues fv;
  Tensor sigma;
  PetscErrorCode ierr;
  double diameter, omega;
  std::vector<Point> ref_qpoints, qpoints;
  std::vector<double> weights, jxw, residuals;

  PetscFunctionBegin;

  omega = 2 * PI * ctx->freqs[fidx];

  ierr = get_cell_attribute(ctx, cell, sigma);

  calculate_ref_cell_quadrature_points(2, ref_qpoints, weights);
  transform_cell_quadrature_points(cell, ref_qpoints, weights, qpoints, jxw);

  fv.reinit(cell);

  residuals.resize(solutions.size());

  for (i = 0; i < (int)solutions.size(); ++i) {
    residuals[i] = 0.0;
    for (q = 0; q < (int)qpoints.size(); ++q) {
      ierr = calculate_function_value(ctx, fv, qpoints[q], *(solutions[i]), e); CHKERRQ(ierr);
      residuals[i] += (II * omega * (sigma * e)).squaredNorm() * jxw[q];
    }
  }

  diameter = cell.diameter();

  for (i = 0; i < (int)solutions.size(); ++i) {
    errors[i][cell.index()] += residuals[i] * diameter * diameter;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode integrate_over_face(EMContext *ctx, const TetAccessor &cell, int f,
                                   const std::vector<PETScBlockVector *> &solutions, std::vector<Eigen::MatrixXd> &face_errors) {
  VectorD fn;
  double diameter;
  int i, q, n, non;
  FEValues fv, fv_n;
  TetAccessor cell_n;
  PetscErrorCode ierr;
  Tensor sigma, sigma_n;
  VectorZ e, e_n, ce, ce_n;
  std::vector<double> weights, jxw, dotj, crossh;
  std::vector<Point> ref_qpoints, qpoints;

  PetscFunctionBegin;

  n = cell.neighbor(f);
  non = cell.neighbor_of_neighbor(f);

  cell_n = TetAccessor(ctx->mesh.get(), n);

  ierr = get_cell_attribute(ctx, cell, sigma);
  ierr = get_cell_attribute(ctx, cell_n, sigma_n);

  fv.reinit(cell);
  fv_n.reinit(cell_n);

  fn = fv.face_normal(f);

  calculate_ref_face_quadrature_points(2, ref_qpoints, weights);
  transform_face_quadrature_points(cell, f, ref_qpoints, weights, qpoints, jxw);

  dotj.resize(solutions.size());
  crossh.resize(solutions.size());

  for (i = 0; i < (int)solutions.size(); ++i) {
    dotj[i] = 0.0;
    crossh[i] = 0.0;
    for (q = 0; q < (int)qpoints.size(); ++q) {
      ierr = calculate_function_value(ctx, fv, qpoints[q], *(solutions[i]), e); CHKERRQ(ierr);
      ierr = calculate_function_value(ctx, fv_n, qpoints[q], *(solutions[i]), e_n); CHKERRQ(ierr);
      dotj[i] += std::norm(((sigma * e) - (sigma_n * e_n)).dot(fn)) * jxw[q];

      ierr = calculate_function_curl(ctx, fv, qpoints[q], *(solutions[i]), ce); CHKERRQ(ierr);
      ierr = calculate_function_curl(ctx, fv_n, qpoints[q], *(solutions[i]), ce_n); CHKERRQ(ierr);
      crossh[i] += (ce - ce_n).cross(fn).squaredNorm() * jxw[q];
    }
  }

  diameter = cell.face_diameter(f);

  for (i = 0; i < (int)solutions.size(); ++i) {
    face_errors[i](cell.index(), f) = (dotj[i] + crossh[i]) * diameter / 2.0;
    face_errors[i](n, non) = face_errors[i](cell.index(), f);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode estimate_error(EMContext *ctx, int fidx, int tidx) {
  int i, t, f;
  TetAccessor cell;
  PetscErrorCode ierr;
  std::vector<Eigen::VectorXd> errors;
  std::vector<Eigen::MatrixXd> face_errors;
  std::vector<PETScBlockVector *> solutions;

  PetscFunctionBegin;

  LogEventHelper leh(ctx->EstimateError);

  if (tidx < 0) {
    errors.resize(3);
    for (i = 0; i < 3; ++i) {
      errors[i].resize(ctx->mesh->n_tets());
      errors[i].setZero();
    }
    solutions.resize(3);
    solutions[0] = &ctx->mt_e[XY_POLAR];
    solutions[1] = &ctx->mt_e[YX_POLAR];
    solutions[2] = &ctx->dual_e;
  } else {
    errors.resize(2);
    for (i = 0; i < 2; ++i) {
      errors[i].resize(ctx->mesh->n_tets());
      errors[i].setZero();
    }
    solutions.resize(2);
    solutions[0] = &ctx->csem_e;
    solutions[1] = &ctx->dual_e;
  }

  face_errors.resize(solutions.size());
  for (i = 0; i < (int)solutions.size(); ++i) {
    face_errors[i].resize(ctx->mesh->n_tets(), TRIANGLES_PER_TET);
    face_errors[i].setConstant(-1.0);
  }

  for (t = 0; t < ctx->mesh->n_tets(); ++t) {
    cell = TetAccessor(ctx->mesh.get(), t);
    if (!cell.is_locally_owned()) {
      continue;
    }

    ierr = integrate_over_cell(ctx, cell, fidx, solutions, errors); CHKERRQ(ierr);

    for (f = 0; f < TRIANGLES_PER_TET; ++f) {
      if (cell.neighbor(f) < 0) {
        for (i = 0; i < (int)solutions.size(); ++i) {
          face_errors[i](t, f) = 0.0;
        }
      }

      if (face_errors[0](t, f) >= 0.0) {
        continue;
      }
      ierr = integrate_over_face(ctx, cell, f, solutions, face_errors); CHKERRQ(ierr);
    }

    for (i = 0; i < (int)solutions.size(); ++i) {
      for (f = 0; f < TRIANGLES_PER_TET; ++f) {
        errors[i][t] += face_errors[i](t, f);
      }
    }
  }

  for (i = 0; i < (int)solutions.size(); ++i) {
    ierr = MPI_Allreduce(MPI_IN_PLACE, &errors[i][0], errors[i].size(), MPI_DOUBLE, MPI_SUM, ctx->group_comm); CHKERRQ(ierr);
  }

  if (tidx < 0) {
    ctx->mt_error[XY_POLAR] = (errors[0].array() * errors[2].array()).sqrt();
    ctx->mt_error[YX_POLAR] = (errors[1].array() * errors[2].array()).sqrt();
  } else {
    ctx->csem_error = (errors[0].array() * errors[1].array()).sqrt();
  }

  PetscFunctionReturn(0);
}

PetscErrorCode refine_fixed_fraction(const Eigen::VectorXd &error, std::vector<bool> &flag,
                                     double fraction) {
  int i;
  Eigen::VectorXd sorted_error;
  double total_error, sum, threshold;

  PetscFunctionBegin;

  assert(error.size() == (int)flag.size());

  total_error = error.sum();

  sorted_error = error;
  std::sort(sorted_error.data(), sorted_error.data() + sorted_error.size(), std::greater<double>());

  sum = 0.0;
  threshold = 0.0;
  for (i = 0; i < sorted_error.size(); ++i) {
    if (sum < fraction * total_error) {
      sum += sorted_error[i];
    } else {
      threshold = sorted_error[i];
      break;
    }
  }

  for (i = 0; i < error.size(); ++i) {
    if (error[i] > threshold) {
      flag[i] = true;
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode refine_fixed_number(const Eigen::VectorXd &error, std::vector<bool> &flag,
                                   double fraction) {
  double threshold;
  Eigen::VectorXd tmp;
  int i, n_refine_cells;

  PetscFunctionBegin;

  assert(error.size() == (int)flag.size());

  n_refine_cells = (int)error.size() * fraction;

  tmp = error;
  std::nth_element(tmp.data(), tmp.data() + n_refine_cells, tmp.data() + tmp.size(), std::greater<double>());
  threshold = tmp[n_refine_cells];

  for (i = 0; i < error.size(); ++i) {
    if (error[i] > threshold) {
      flag[i] = true;
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode refine_mesh(EMContext *ctx, int tidx) {
  PetscErrorCode ierr;
  std::vector<bool> refine_flag;

  PetscFunctionBegin;

  LogEventHelper leh(ctx->RefineMesh);

  refine_flag.resize(ctx->mesh->n_tets());
  std::fill(refine_flag.begin(), refine_flag.end(), false);

  if (tidx < 0) {
    if (ctx->refine_strategy == FixedFraction) {
      ierr = refine_fixed_fraction(ctx->mt_error[XY_POLAR], refine_flag, ctx->refine_fraction); CHKERRQ(ierr);
      ierr = refine_fixed_fraction(ctx->mt_error[YX_POLAR], refine_flag, ctx->refine_fraction); CHKERRQ(ierr);
    } else {
      ierr = refine_fixed_number(ctx->mt_error[XY_POLAR], refine_flag, ctx->refine_fraction); CHKERRQ(ierr);
      ierr = refine_fixed_number(ctx->mt_error[YX_POLAR], refine_flag, ctx->refine_fraction); CHKERRQ(ierr);
    }
  } else {
    if (ctx->refine_strategy == FixedFraction) {
      ierr = refine_fixed_fraction(ctx->csem_error, refine_flag, ctx->refine_fraction); CHKERRQ(ierr);
    } else {
      ierr = refine_fixed_number(ctx->csem_error, refine_flag, ctx->refine_fraction); CHKERRQ(ierr);
    }
  }

  ctx->mesh->refine_tetgen(refine_flag);

  PetscFunctionReturn(0);
}

PetscErrorCode distribute_tasks(EMContext *ctx, std::vector<TupleII> &tasks) {
  std::set<TupleII> ftmask;
  std::set<TupleII>::iterator sit;
  PetscInt i, n_tasks, t_beg, t_end;
  std::vector<TupleII> cs_tasks, mt_tasks;

  PetscFunctionBegin;

  for (i = 0; i < (PetscInt)ctx->obs.size(); ++i) {
    ftmask.insert(std::make_tuple(ctx->fidx[i], ctx->tidx[i]));
  }

  for (sit = ftmask.begin(); sit != ftmask.end(); ++sit) {
    if (std::get<1>(*sit) < 0) {
      mt_tasks.push_back(*sit);
    } else {
      cs_tasks.push_back(*sit);
    }
  }

  n_tasks = (PetscInt)cs_tasks.size();

  t_beg = 0;
  for (i = 0; i < ctx->group_id; ++i) {
    t_beg += n_tasks / ctx->n_groups + ((n_tasks % ctx->n_groups) > i ? 1 : 0);
  }
  t_end = t_beg + n_tasks / ctx->n_groups + ((n_tasks % ctx->n_groups) > ctx->group_id ? 1 : 0);

  for (i = t_beg; i < t_end; ++i) {
    tasks.push_back(cs_tasks[i]);
  }

  n_tasks = (PetscInt)mt_tasks.size();

  t_beg = 0;
  for (i = 0; i < ctx->group_id; ++i) {
    t_beg += n_tasks / ctx->n_groups + ((n_tasks % ctx->n_groups) > i ? 1 : 0);
  }
  t_end = t_beg + n_tasks / ctx->n_groups + ((n_tasks % ctx->n_groups) > ctx->group_id ? 1 : 0);

  for (i = t_beg; i < t_end; ++i) {
    tasks.push_back(mt_tasks[i]);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode forward_mt(EMContext *ctx, int fidx) {
  int cycle;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  LogStageHelper freq_lsh(string_format("Freq-%d", (int)fidx));

  ierr = PetscViewerASCIIPrintf(ctx->LS_log, "Freq %g Hz:\n", ctx->freqs[fidx]); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(ctx->LS_log); CHKERRQ(ierr);

  ctx->mesh->copy(*ctx->original_mesh);

  for (cycle = 0; cycle < ctx->max_adaptive_refinements + 1; ++cycle) {
    LogStageHelper freq_refine_lsh(string_format("Freq-%d-Refine-%d", fidx, cycle));

    ierr = PetscViewerASCIIPrintf(ctx->LS_log, "Cycle %d, cells %d, dofs %d:\n", cycle, ctx->mesh->n_tets(), ctx->mesh->n_edges()); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(ctx->LS_log); CHKERRQ(ierr);

    ierr = create_linear_system(ctx, -3); CHKERRQ(ierr);

    ierr = assemble_matrix(ctx, fidx); CHKERRQ(ierr);
    ierr = create_pc(ctx); CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(ctx->LS_log, "Solving for XY mode:\n"); CHKERRQ(ierr);
    ierr = assemble_rhs_mt(ctx, fidx, XY_POLAR); CHKERRQ(ierr);
    ierr = solve_linear_system(ctx, ctx->s, ctx->mt_e[XY_POLAR], ctx->K_max_it, ctx->e_rtol); CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(ctx->LS_log, "Solving for YX mode:\n"); CHKERRQ(ierr);
    ierr = assemble_rhs_mt(ctx, fidx, YX_POLAR); CHKERRQ(ierr);
    ierr = solve_linear_system(ctx, ctx->s, ctx->mt_e[YX_POLAR], ctx->K_max_it, ctx->e_rtol); CHKERRQ(ierr);

    ierr = assemble_rhs_dual(ctx); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(ctx->LS_log, "Solving for dual mode:\n"); CHKERRQ(ierr);
    ierr = solve_linear_system(ctx, ctx->s, ctx->dual_e, ctx->K_max_it, ctx->dual_rtol); CHKERRQ(ierr);

    ierr = estimate_error(ctx, fidx, -3); CHKERRQ(ierr);
    ierr = save_mesh(ctx, (string_format("%s-%02d-%02d.vtk", ctx->oprefix, fidx, cycle)).c_str(), -3); CHKERRQ(ierr);

    ierr = destroy_pc(ctx); CHKERRQ(ierr);

    ierr = PetscViewerASCIIPopTab(ctx->LS_log); CHKERRQ(ierr);

    if (cycle == ctx->max_adaptive_refinements || ctx->mesh->n_edges() >= ctx->max_dofs) {
      ierr = calculate_response_mt(ctx, fidx); CHKERRQ(ierr);
      ierr = destroy_linear_system(ctx); CHKERRQ(ierr);
      break;
    } else {
      ierr = refine_mesh(ctx, -3); CHKERRQ(ierr);
    }

    ierr = destroy_linear_system(ctx); CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopTab(ctx->LS_log); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode forward_csem(EMContext *ctx, int fidx, int tidx) {
  int cycle;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  LogStageHelper freq_lsh(string_format("Freq-%d-TX-%d", fidx, tidx));

  ierr = PetscViewerASCIIPrintf(ctx->LS_log, "Freq %g Hz, TX %d:\n", ctx->freqs[fidx], tidx); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(ctx->LS_log); CHKERRQ(ierr);

  ctx->mesh->copy(*ctx->original_mesh);

  for (cycle = 0; cycle < ctx->max_adaptive_refinements + 1; ++cycle) {
    LogStageHelper freq_refine_lsh(string_format("Freq-%d-TX-%d-Refine-%d", fidx, tidx, cycle));

    ierr = PetscViewerASCIIPrintf(ctx->LS_log, "Cycle %d, cells %d, dofs %d:\n", cycle, ctx->mesh->n_tets(), ctx->mesh->n_edges()); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(ctx->LS_log); CHKERRQ(ierr);

    ierr = create_linear_system(ctx, tidx); CHKERRQ(ierr);

    ierr = assemble_matrix(ctx, fidx); CHKERRQ(ierr);
    ierr = create_pc(ctx); CHKERRQ(ierr);

    ierr = assemble_rhs_csem(ctx, fidx, tidx); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(ctx->LS_log, "Solving for E mode:\n"); CHKERRQ(ierr);
    ierr = solve_linear_system(ctx, ctx->s, ctx->csem_e, ctx->K_max_it, ctx->e_rtol); CHKERRQ(ierr);

    ierr = assemble_rhs_dual(ctx); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(ctx->LS_log, "Solving for dual mode:\n"); CHKERRQ(ierr);
    ierr = solve_linear_system(ctx, ctx->s, ctx->dual_e, ctx->K_max_it, ctx->dual_rtol); CHKERRQ(ierr);

    ierr = estimate_error(ctx, fidx, tidx); CHKERRQ(ierr);
    ierr = save_mesh(ctx, (string_format("%s-%02d-%03d-%02d.vtk", ctx->oprefix, fidx, tidx, cycle)).c_str(), tidx); CHKERRQ(ierr);

    ierr = destroy_pc(ctx); CHKERRQ(ierr);

    ierr = PetscViewerASCIIPopTab(ctx->LS_log); CHKERRQ(ierr);

    if (cycle == ctx->max_adaptive_refinements || ctx->mesh->n_edges() >= ctx->max_dofs) {
      ierr = calculate_response_csem(ctx, fidx, tidx); CHKERRQ(ierr);
      ierr = destroy_linear_system(ctx); CHKERRQ(ierr);
      break;
    } else {
      ierr = refine_mesh(ctx, tidx); CHKERRQ(ierr);
    }

    ierr = destroy_linear_system(ctx); CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopTab(ctx->LS_log); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode em_forward(EMContext *ctx) {
  int i;
  PetscErrorCode ierr;
  std::vector<TupleII> tasks;

  PetscFunctionBegin;

  LogStageHelper lsh("Forward");

  ierr = refine_receiving_area(ctx); CHKERRQ(ierr);

  ierr = update_background_model(ctx); CHKERRQ(ierr);

  ierr = distribute_tasks(ctx, tasks); CHKERRQ(ierr);

  for (i = 0; i < (int)tasks.size(); ++i) {
    if (std::get<1>(tasks[i]) < 0) {
      ierr = forward_mt(ctx, std::get<0>(tasks[i])); CHKERRQ(ierr);
    } else {
      ierr = forward_csem(ctx, std::get<0>(tasks[i]), std::get<1>(tasks[i])); CHKERRQ(ierr);
    }
  }

  ierr = save_rsp(ctx, (std::string(ctx->oprefix) + ".rsp").c_str()); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
