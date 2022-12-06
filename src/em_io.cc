#include "em_ctx.h"
#include "em_utils.h"
#include "em_mesh.h"

#include <petsc.h>

#include <fstream>

static PetscErrorCode read_rho(EMContext *ctx, std::stringstream &ss) {
  int i, j, n_rho;

  PetscFunctionBegin;

  ss >> n_rho >> ctx->aniso_form;

  switch (ctx->aniso_form) {
    case Isotropic:
      ctx->rho.resize(n_rho, 1);

      ctx->rho_name.resize(1);
      ctx->rho_name[0] = "rho";
      break;
    case Vertical:
      ctx->rho.resize(n_rho, 2);

      ctx->rho_name.resize(2);
      ctx->rho_name[0] = "rho_h";
      ctx->rho_name[1] = "rho_v";
      break;
    case Triaxial:
      ctx->rho.resize(n_rho, 3);

      ctx->rho_name.resize(3);
      ctx->rho_name[0] = "rho_x";
      ctx->rho_name[1] = "rho_y";
      ctx->rho_name[2] = "rho_z";
      break;
    case Arbitrary:
      ctx->rho.resize(n_rho, 6);
      ctx->rho_name.resize(6);
      ctx->rho_name[0] = "rho_x";
      ctx->rho_name[1] = "rho_y";
      ctx->rho_name[2] = "rho_z";
      ctx->rho_name[3] = "eta_x";
      ctx->rho_name[4] = "eta_y";
      ctx->rho_name[5] = "eta_z";
      break;
    default:
      SETERRQ(ctx->world_comm, EM_ERR_USER, "%s", string_format("Unsupported anisotropy form %d.", ctx->aniso_form).c_str());
      break;
  }

  ctx->lb.resize(n_rho);
  ctx->ub.resize(n_rho);

  for (i = 0; i < n_rho; ++i) {
    switch (ctx->aniso_form) {
    case Isotropic:
      for (j = 0; j < 1; ++j) {
        ss >> ctx->rho(i, j);
      }
      break;
    case Vertical:
      for (j = 0; j < 2; ++j) {
        ss >> ctx->rho(i, j);
      }
      break;
    case Triaxial:
      for (j = 0; j < 3; ++j) {
        ss >> ctx->rho(i, j);
      }
      break;
    case Arbitrary:
      for (j = 0; j < 6; ++j) {
        ss >> ctx->rho(i, j);
      }
      break;
    default:
      SETERRQ(ctx->world_comm, EM_ERR_USER, "%s", string_format("Unsupported anisotropy form %d.", ctx->aniso_form).c_str());
      break;
    }
    ss >> ctx->ub[i] >> ctx->ub[i];
  }

  PetscFunctionReturn(0);
}

PetscErrorCode read_mdl(EMContext *ctx) {
  tetgenio in;
  tetgenio::facet *f;
  tetgenio::polygon *p;

  std::string l;
  std::stringstream ss;

  int i, j;

  PetscFunctionBegin;

  std::ifstream ifs(ctx->iprefix + std::string(".mdl"));
  if (!ifs.good()) {
    SETERRQ(ctx->world_comm, EM_ERR_USER, "%s", string_format("Unable to open file %s.mdl.", ctx->iprefix).c_str());
  }

  while (std::getline(ifs, l)) {
    ss << parse_string(l);
  }

  ss >> in.numberofpoints;
  in.pointlist = new double[in.numberofpoints * 3];

  for (i = 0; i < in.numberofpoints; ++i) {
    for (j = 0; j < 3; ++j) {
      ss >> in.pointlist[i * 3 + j];
    }
  }

  ss >> in.numberoffacets;
  in.facetlist = new tetgenio::facet[in.numberoffacets];
  in.facetmarkerlist = new int[in.numberoffacets];

  for (i = 0; i < in.numberoffacets; ++i) {
    f = in.facetlist + i;
    f->numberofpolygons = 1;
    f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
    f->numberofholes = 0;
    f->holelist = NULL;

    p = f->polygonlist;

    ss >> p->numberofvertices;
    p->vertexlist = new int[p->numberofvertices];
    for (j = 0; j < p->numberofvertices; ++j) {
      ss >> p->vertexlist[j];
    }
    ss >> in.facetmarkerlist[i];
  }

  ss >> in.numberofregions;
  in.regionlist = new double[in.numberofregions * 5];
  for (i = 0; i < in.numberofregions; ++i) {
    for (j = 0; j < 5; ++j) {
      ss >> in.regionlist[i * 5 + j];
    }
    in.regionlist[i * 5 + 4] = std::pow(in.regionlist[i * 5 + 4], 3);
  }

  ctx->coarse_mesh->create(&in);

  read_rho(ctx, ss);

  PetscFunctionReturn(0);
}

PetscErrorCode read_mesh(EMContext *ctx) {
  tetgenio in;

  std::string l;
  std::stringstream ss;

  int i, j;

  PetscFunctionBegin;

  std::ifstream ifs_mesh(ctx->iprefix + std::string(".mesh"));
  if (!ifs_mesh.good()) {
    SETERRQ(ctx->world_comm, EM_ERR_USER, "%s", string_format("Unable to open file %s.mesh.", ctx->iprefix).c_str());
  }

  while (std::getline(ifs_mesh, l)) {
    ss << parse_string(l);
  }

  ss >> in.numberofpoints;
  in.pointlist = new double[in.numberofpoints * 3];

  for (i = 0; i < in.numberofpoints; ++i) {
    for (j = 0; j < 3; ++j) {
      ss >> in.pointlist[i * 3 + j];
    }
  }

  ss >> in.numberoftetrahedra;
  in.tetrahedronlist = new int[in.numberoftetrahedra * 4];
  in.numberoftetrahedronattributes = 1;
  in.tetrahedronattributelist = new double[in.numberoftetrahedra];

  for (i = 0; i < in.numberoftetrahedra; ++i) {
    for (j = 0; j < 4; ++j) {
      ss >> in.tetrahedronlist[i * 4 + j];
    }
    ss >> in.tetrahedronattributelist[i];
  }

  ctx->coarse_mesh->create(&in);

  read_rho(ctx, ss);

  PetscFunctionReturn(0);
}

PetscErrorCode read_emd(EMContext *ctx) {
  std::string l;
  std::stringstream ss;
  double re, im, err_re, err_im;
  size_t n_freqs, n_rxes, n_txes, n_obses;

  PetscFunctionBegin;

  std::ifstream ifs(ctx->iprefix + std::string(".emd"));
  if (!ifs.good()) {
    SETERRQ(ctx->world_comm, EM_ERR_USER, "%s", string_format("Unable to open file %s.emd.", ctx->iprefix).c_str());
  }

  while (std::getline(ifs, l)) {
    ss << parse_string(l);
  }

  ss >> n_freqs;
  ctx->freqs.resize(n_freqs);
  for (size_t i = 0; i < n_freqs; ++i) {
    ss >> ctx->freqs[i];
  }

  ss >> n_txes;
  ctx->tx.resize(n_txes * TX_SIZE);
  for (size_t i = 0; i < n_txes * TX_SIZE; ++i) {
    ss >> ctx->tx[i];
  }

  ss >> n_rxes;
  ctx->rx.resize(n_rxes);
  for (size_t i = 0; i < n_rxes; ++i) {
    ss >> ctx->rx[i][0] >> ctx->rx[i][1] >> ctx->rx[i][2];
  }

  ss >> n_obses;
  ctx->otype.resize(n_obses);
  ctx->fidx.resize(n_obses);
  ctx->tidx.resize(n_obses);
  ctx->ridx.resize(n_obses);
  ctx->obs.resize(n_obses);
  ctx->rsp.resize(n_obses);
  ctx->obserr.resize(n_obses);

  for (size_t i = 0; i < n_obses; ++i) {
    ss >> ctx->otype[i] >> ctx->fidx[i] >> ctx->tidx[i] >> ctx->ridx[i];
    ss >> re >> im >> err_re >> err_im;
    switch (ctx->otype[i]) {
      case R_XY_AP:
      case R_YX_AP:
      case F_EX_AP:
      case F_EY_AP:
      case F_EZ_AP:
      case F_HX_AP:
      case F_HY_AP:
      case F_HZ_AP:
        err_re = err_re / 100.0;
        err_im = err_im * DTOR;
        ctx->obs[i] = Complex(std::log(re), im * DTOR);
        ctx->obserr[i] = Complex(err_re * err_re, err_im * err_im);
        break;
      case F_EX_RI:
      case F_EY_RI:
      case F_EZ_RI:
      case F_HX_RI:
      case F_HY_RI:
      case F_HZ_RI:
      case Z_XX_RI:
      case Z_XY_RI:
      case Z_YX_RI:
      case Z_YY_RI:
      case T_ZX_RI:
      case T_ZY_RI:
        ctx->obs[i] = Complex(re, im);
        ctx->obserr[i] = Complex(err_re * err_re, err_im * err_im);
        break;
      default:
        ctx->obs[i] = Complex(re, im);
        ctx->obserr[i] = Complex(err_re * err_re, err_im * err_im);
        break;
    }
  }
  ctx->rsp.setZero();

  PetscFunctionReturn(0);
}

PetscErrorCode save_mesh(EMContext *ctx, const char *fname, int tidx) {
  int i, j;
  TetAccessor tet;
  std::map<std::string, Eigen::VectorXd> data;

  PetscFunctionBegin;

  for (i = 0; i < ctx->rho.cols(); ++i) {
    data[ctx->rho_name[i]].resize(ctx->mesh->n_tets());
    for (j = 0; j < ctx->mesh->n_tets(); ++j) {
      tet = TetAccessor(ctx->mesh.get(), j);
      data[ctx->rho_name[i]][j] = ctx->rho(tet.attribute(), i);
    }
  }

  if (tidx < 0) {
    data["error_xy"] = ctx->mt_error[XY_POLAR];
    data["error_yx"] = ctx->mt_error[YX_POLAR];
  } else {
    data["error"] = ctx->csem_error;
  }

  ctx->mesh->save_vtk(fname, data);

  PetscFunctionReturn(0);
}

PetscErrorCode save_rsp(EMContext *ctx, const char *fn) {
  int i, j;
  FILE *fp;
  Complex obs, rsp;

  PetscFunctionBegin;

  PetscCall(MPI_Allreduce(MPI_IN_PLACE, &ctx->rsp[0], ctx->rsp.size() * 2, MPI_DOUBLE, MPI_SUM, ctx->world_comm));

  PetscCall(PetscFOpen(ctx->world_comm, fn, "w", &fp));

  PetscFPrintf(ctx->world_comm, fp, "# frequencies\n");
  PetscFPrintf(ctx->world_comm, fp, "%d\n", (int)ctx->freqs.size());
  for (i = 0; i < (int)ctx->freqs.size(); ++i) {
    PetscFPrintf(ctx->world_comm, fp, "%.4E\n", ctx->freqs[i]);
  }

  PetscFPrintf(ctx->world_comm, fp, "# transmiters\n");
  PetscFPrintf(ctx->world_comm, fp, "%d\n", (int)ctx->tx.size() / TX_SIZE);
  for (i = 0; i < (int)ctx->tx.size() / TX_SIZE; ++i) {
    for (j = 0; j < TX_SIZE; ++j) {
      PetscFPrintf(ctx->world_comm, fp, "% .4E ", ctx->tx[i * TX_SIZE + j]);
    }
    PetscFPrintf(ctx->world_comm, fp, "\n");
  }

  PetscFPrintf(ctx->world_comm, fp, "# recievers\n");
  PetscFPrintf(ctx->world_comm, fp, "%d\n", (int)ctx->rx.size());
  for (i = 0; i < (int)ctx->rx.size(); ++i) {
    PetscFPrintf(ctx->world_comm, fp, "% .4E % .4E % .4E\n", ctx->rx[i][0], ctx->rx[i][1], ctx->rx[i][2]);
  }

  PetscFPrintf(ctx->world_comm, fp, "# observations\n");
  PetscFPrintf(ctx->world_comm, fp, "%d\n", (int)ctx->rsp.size());
  PetscFPrintf(ctx->world_comm, fp, "#%6s %6s %5s %7s %13s %13s %13s %13s\n", "otype", "fidx", "tidx",
               "ridx", "amp", "phase", "amp`", "phase`");
  for (i = 0; i < ctx->rsp.size(); ++i) {
    switch (ctx->otype[i]) {
    case R_XY_AP:
    case R_YX_AP:
    case F_EX_AP:
    case F_EY_AP:
    case F_EZ_AP:
    case F_HX_AP:
    case F_HY_AP:
    case F_HZ_AP:
      obs = Complex(std::exp(std::real(ctx->obs[i])), std::imag(ctx->obs[i]) * RTOD);
      rsp = Complex(std::exp(std::real(ctx->rsp[i])), std::imag(ctx->rsp[i]) * RTOD);
      PetscFPrintf(ctx->world_comm, fp, "% 7d % 6d % 5d % 7d % 13.6E % 13.3f % 13.6E % 13.3f\n",
                   ctx->otype[i], ctx->fidx[i], ctx->tidx[i], ctx->ridx[i], std::real(obs),
                   std::imag(obs), std::real(rsp), std::imag(rsp));
      break;
    case F_EX_RI:
    case F_EY_RI:
    case F_EZ_RI:
    case F_HX_RI:
    case F_HY_RI:
    case F_HZ_RI:
    case Z_XX_RI:
    case Z_XY_RI:
    case Z_YX_RI:
    case Z_YY_RI:
    case T_ZX_RI:
    case T_ZY_RI:
      obs = ctx->obs[i];
      rsp = ctx->rsp[i];
      PetscFPrintf(ctx->world_comm, fp, "% 7d % 6d % 5d % 7d % 13.6E % 13.6E % 13.6E % 13.6E\n",
                   ctx->otype[i], ctx->fidx[i], ctx->tidx[i], ctx->ridx[i], std::real(obs),
                   std::imag(obs), std::real(rsp), std::imag(rsp));
      break;
    default:
      obs = ctx->obs[i];
      rsp = ctx->rsp[i];
      PetscFPrintf(ctx->world_comm, fp, "% 7d % 6d % 5d % 7d % 13.6E % 13.6E % 13.6E % 13.6E\n",
                   ctx->otype[i], ctx->fidx[i], ctx->tidx[i], ctx->ridx[i], std::real(obs),
                   std::imag(obs), std::real(rsp), std::imag(rsp));
      break;
    }
  }

  PetscCall(PetscFClose(ctx->world_comm, fp));

  PetscFunctionReturn(0);
}
