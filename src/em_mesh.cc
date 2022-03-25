#include "em_mesh.h"

#include <metis.h>

#include <petsc.h>

#include <queue>
#include <tuple>

void Mesh::create_from_tetgen(tetgenio *in) {
  int i, t;

  assert(in != NULL);

  if (tio_ == in) {
    return;
  }

  if (tio_) {
    delete tio_;
    tio_ = NULL;
  }
  tio_ = new tetgenio;

  if (in->tetrahedronlist) {
    tetrahedralize((char *)"zDprfnneQ", in, tio_, NULL);
  } else {
    tetrahedralize((char *)"zDpq1.2AafnneQ", in, tio_, NULL);
  }

  build_vertex_info();
  build_edge_info();
  build_neighbor_info();
  build_boundary_info();
  partition();
  init_kd_tree();
}

void Mesh::copy(const Mesh &m) {
  if (this == &m) {
    return;
  }

  create_from_tetgen(m.tio_);
}

void Mesh::refine_tetgen(const std::vector<bool> &flag) {
  int t, nt, i;
  tetgenio out;

  assert(((int)flag.size()) == n_tets());
  assert(tio_->numberoftetrahedra == n_tets());

  nt = 0;
  for (t = 0; t < n_tets(); ++t) {
    if (flag[t]) {
      nt += 1;
    }
  }

  if (nt == 0) {
    return;
  }

  if (tio_->tetrahedronvolumelist) {
    delete[] tio_->tetrahedronvolumelist;
    tio_->tetrahedronvolumelist = NULL;
  }
  tio_->tetrahedronvolumelist = new double[tio_->numberoftetrahedra];
  std::fill(tio_->tetrahedronvolumelist, tio_->tetrahedronvolumelist + tio_->numberoftetrahedra,
            -1.0);

  for (t = 0; t < n_tets(); ++t) {
    if (flag[t]) {
      tio_->tetrahedronvolumelist[t] = TetAccessor(this, t).volume() / 2.0;
    }
  }

  tetrahedralize((char *)"zDprq1.2AafnneQ", tio_, &out, NULL);

  create_from_tetgen(&out);
}

void Mesh::save_vtk(const char *fn, const std::map<std::string, Eigen::VectorXd> &data) {
  FILE *fp;
  int n_cells, n_pts, i, j;
  std::map<std::string, Eigen::VectorXd>::const_iterator it;

  PetscFOpen(comm_, fn, "w", &fp);

  n_cells = n_tets();
  n_pts = n_vertices();

  PetscFPrintf(comm_, fp, "# vtk DataFile Version 2.0\n");
  PetscFPrintf(comm_, fp, "Unstructured Grid\n");
  PetscFPrintf(comm_, fp, "ASCII\n");
  PetscFPrintf(comm_, fp, "DATASET UNSTRUCTURED_GRID\n");
  PetscFPrintf(comm_, fp, "POINTS %d double\n", n_pts);

  for (i = 0; i < n_pts; ++i) {
    PetscFPrintf(comm_, fp, "%.17g %.17g %.17g\n", vertex(i)[0], vertex(i)[1],
                 vertex(i)[2]);
  }
  PetscFPrintf(comm_, fp, "\n");

  PetscFPrintf(comm_, fp, "CELLS %d %d\n", n_cells, n_cells * (4 + 1));
  for (i = 0; i < n_cells; ++i) {
    PetscFPrintf(comm_, fp, "%d", VERTICES_PER_TET);
    for (j = 0; j < VERTICES_PER_TET; ++j) {
      PetscFPrintf(comm_, fp, " %4d", tet2vertex(i, j));
    }
    PetscFPrintf(comm_, fp, "\n");
  }
  PetscFPrintf(comm_, fp, "\n");

  PetscFPrintf(comm_, fp, "CELL_TYPES %d\n", n_cells);
  for (i = 0; i < n_cells; i++) {
    PetscFPrintf(comm_, fp, "%d\n", 10);
  }
  PetscFPrintf(comm_, fp, "\n");

  PetscFPrintf(comm_, fp, "CELL_DATA %d\n", n_cells);

  PetscFPrintf(comm_, fp, "SCALARS subdomain double 1\n");
  PetscFPrintf(comm_, fp, "LOOKUP_TABLE default\n");
  for (i = 0; i < n_cells; ++i) {
    PetscFPrintf(comm_, fp, "%.15g\n", (double)subdomain_[i]);
  }
  PetscFPrintf(comm_, fp, "\n");

  PetscFPrintf(comm_, fp, "SCALARS tet_attr double 1\n");
  PetscFPrintf(comm_, fp, "LOOKUP_TABLE default\n");
  for (i = 0; i < n_cells; ++i) {
    PetscFPrintf(comm_, fp, "%.15g\n", (double)attribute(i));
  }
  PetscFPrintf(comm_, fp, "\n");

  for (it = data.begin(); it != data.end(); ++it) {
    PetscFPrintf(comm_, fp, "SCALARS %s double 1\n", it->first.c_str());
    PetscFPrintf(comm_, fp, "LOOKUP_TABLE default\n");
    for (i = 0; i < n_cells; ++i) {
      PetscFPrintf(comm_, fp, "%.15g\n", (double)it->second[i]);
    }
    PetscFPrintf(comm_, fp, "\n");
  }

  PetscFClose(comm_, fp);
}

void Mesh::partition() {
  std::vector<idx_t> part, xadj, adjncy;
  idx_t i, j, nnz, nvtxs, ncon, np, options[METIS_NOPTIONS], dummy, ierr;

  nnz = 0;
  for (i = 0; i < n_tets(); ++i) {
    for (j = 0; j < TRIANGLES_PER_TET; ++j) {
      if (neighbor(i, j) >= 0) {
        ++nnz;
      }
    }
  }
  xadj.resize(n_tets() + 1);
  adjncy.resize(nnz);

  nnz = 0;
  for (i = 0; i < n_tets(); ++i) {
    xadj[i] = nnz;
    for (j = 0; j < TRIANGLES_PER_TET; ++j) {
      if (neighbor(i, j) >= 0) {
        adjncy[nnz++] = neighbor(i, j);
      }
    }
  }
  xadj[i] = nnz;

  ncon = 1;
  np = comm_size_;
  nvtxs = n_tets();
  part.resize(nvtxs);

  METIS_SetDefaultOptions(options);

  if (np == 1) {
    ierr = METIS_OK;
    std::fill(part.begin(), part.end(), 0);
  } else if (np > 1 && np <= 8) {
    ierr = METIS_PartGraphRecursive(&nvtxs, &ncon, &xadj[0], &adjncy[0], NULL, NULL, NULL, &np,
                                    NULL, NULL, options, &dummy, &part[0]);
  } else {
    ierr = METIS_PartGraphKway(&nvtxs, &ncon, &xadj[0], &adjncy[0], NULL, NULL, NULL, &np, NULL,
                               NULL, options, &dummy, &part[0]);
  }

  (void)ierr;
  assert(ierr == METIS_OK);

  subdomain_.resize(n_tets());
  std::copy(part.begin(), part.end(), subdomain_.begin());

  extract_ghost_cells();
  compute_new_vertex_indices_();
  compute_new_edge_indices_();
}

void Mesh::get_vertex_owners(std::vector<int> &owners) {
  int t, v;
  bool coin_flip;

  owners.resize(n_vertices());
  std::fill(owners.begin(), owners.end(), -1);

  coin_flip = true;
  for (t = 0; t < n_tets(); ++t) {
    for (v = 0; v < VERTICES_PER_TET; ++v) {
      if (owners[new_vertex_indices_[tet2vertex(t, v)]] < 0) {
        owners[new_vertex_indices_[tet2vertex(t, v)]] = subdomain_[t];
      } else {
        if (coin_flip) {
          owners[new_vertex_indices_[tet2vertex(t, v)]] = subdomain_[t];
        }
        coin_flip = !coin_flip;
      }
    }
  }
}

void Mesh::get_edge_owners(std::vector<int> &owners) {
  int t, e;
  bool coin_flip;

  owners.resize(n_edges());
  std::fill(owners.begin(), owners.end(), -1);

  coin_flip = true;
  for (t = 0; t < n_tets(); ++t) {
    for (e = 0; e < EDGES_PER_TET; ++e) {
      if (owners[new_edge_indices_[tet2edge(t, e)]] < 0) {
        owners[new_edge_indices_[tet2edge(t, e)]] = subdomain_[t];
      } else {
        if (coin_flip) {
          owners[new_edge_indices_[tet2edge(t, e)]] = subdomain_[t];
        }
        coin_flip = !coin_flip;
      }
    }
  }
}

void Mesh::compute_new_vertex_indices_() {
  std::vector<int> owners;
  int v, subdomain, next_free_index;

  get_vertex_owners(owners);

  next_free_index = 0;

  for (subdomain = 0; subdomain < comm_size_; ++subdomain) {
    for (v = 0; v < n_vertices(); ++v) {
      if (owners[v] == subdomain) {
        new_vertex_indices_[v] = next_free_index++;
      }
    }
  }
}

void Mesh::compute_new_edge_indices_() {
  std::vector<int> owners;
  int e, subdomain, next_free_index;

  get_edge_owners(owners);

  next_free_index = 0;

  for (subdomain = 0; subdomain < comm_size_; ++subdomain) {
    for (e = 0; e < n_edges(); ++e) {
      if (owners[e] == subdomain) {
        new_edge_indices_[e] = next_free_index++;
      }
    }
  }
}

void Mesh::extract_ghost_cells() {
  int t, v;
  std::vector<bool> vertices_on_subdomain;

  vertices_on_subdomain.resize(n_vertices());
  std::fill(vertices_on_subdomain.begin(), vertices_on_subdomain.end(), false);
  for (t = 0; t < n_tets(); ++t) {
    if (subdomain_[t] == comm_rank_) {
      for (v = 0; v < VERTICES_PER_TET; ++v) {
        vertices_on_subdomain[tet2vertex(t, v)] = true;
      }
    }
  }

  ghost_flag_.resize(n_tets());
  std::fill(ghost_flag_.begin(), ghost_flag_.end(), false);
  for (t = 0; t < n_tets(); ++t) {
    if (subdomain_[t] != comm_rank_) {
      for (v = 0; v < VERTICES_PER_TET; ++v) {
        if (vertices_on_subdomain[tet2vertex(t, v)]) {
          ghost_flag_[t] = true;
          break;
        }
      }
    }
  }
}

void Mesh::build_vertex_info() {
  int v;

  new_vertex_indices_.resize(n_vertices());
  for (v = 0; v < n_vertices(); ++v) {
    new_vertex_indices_[v] = v;
  }
}

void Mesh::build_edge_info() {
  int i, j;

  for (i = 0; i < tio_->numberofedges; ++i) {
    if (tio_->edgelist[i * 2 + 0] > tio_->edgelist[i * 2 + 1]) {
      std::swap(tio_->edgelist[i * 2 + 0], tio_->edgelist[i * 2 + 1]);
    }
  }

  new_edge_indices_.resize(n_edges());
  for (i = 0; i < n_edges(); ++i) {
    new_edge_indices_[i] = i;
  }
}

void Mesh::build_neighbor_info() {
  int t, f, n;

  tet_non_.resize(n_tets() * TRIANGLES_PER_TET);
  std::fill(tet_non_.begin(), tet_non_.end(), -1);

  for (t = 0; t < n_tets(); ++t) {
    for (f = 0; f < TRIANGLES_PER_TET; ++f) {
      n = neighbor(t, f);
      if (n < 0) {
        continue;
      }
      tet_non_[t * TRIANGLES_PER_TET + f] =
          std::find(tio_->neighborlist + n * TRIANGLES_PER_TET,
                    tio_->neighborlist + (n + 1) * TRIANGLES_PER_TET, t) -
          (tio_->neighborlist + n * TRIANGLES_PER_TET);
    }
  }
}

void Mesh::build_boundary_info() {
  int f, i;

  vertex_bdr_marker_.resize(n_vertices());
  std::fill(vertex_bdr_marker_.begin(), vertex_bdr_marker_.end(), false);

  edge_bdr_marker_.resize(n_edges());
  std::fill(edge_bdr_marker_.begin(), edge_bdr_marker_.end(), false);

  for (f = 0; f < tio_->numberoftrifaces; ++f) {
    if (tio_->face2tetlist[f * 2 + 0] < 0 || tio_->face2tetlist[f * 2 + 1] < 0) {
      for (i = 0; i < VERTICES_PER_TRIANGLE; ++i) {
        vertex_bdr_marker_[tio_->trifacelist[f * VERTICES_PER_TRIANGLE + i]] = true;
      }
      for (i = 0; i < EDGES_PER_TRIANGLE; ++i) {
        edge_bdr_marker_[tio_->face2edgelist[f * EDGES_PER_TRIANGLE + i]] = true;
      }
    }
  }
}

void Mesh::init_kd_tree() {
  int t, v;

  kd_index_.reset(new KDTree(3, *this, 100));
  kd_index_->buildIndex();

  vertex_to_tet_.resize(n_vertices());
  for (t = 0; t < n_tets(); ++t) {
    for (v = 0; v < VERTICES_PER_TET; ++v) {
      vertex_to_tet_[new_vertex_indices_[tet2vertex(t, v)]] = t;
    }
  }
}

std::tuple<Point, int> Mesh::find_closest_vertex(const Point &p) {
  size_t idx;
  double dist;

  kd_index_->knnSearch(&p[0], 1, &idx, &dist);

  return std::make_tuple(vertex(idx), new_vertex_indices_[idx]);
}

int Mesh::find_cell_around_point(const Point &p) {
  int i, tidx, t;
  TetAccessor tet;
  std::queue<int> q;
  std::vector<bool> touched_tets;

  touched_tets.resize(n_tets());
  std::fill(touched_tets.begin(), touched_tets.end(), false);

  q.push(vertex_to_tet_[std::get<1>(find_closest_vertex(p))]);

  t = -1;
  while (!q.empty()) {
    tidx = q.front();
    tet = TetAccessor(this, tidx);
    q.pop();

    if (touched_tets[tidx]) {
      continue;
    }

    touched_tets[tidx] = true;
    if (tet.contains_point(p)) {
      t = tidx;
      break;
    }

    for (i = 0; i < 4; ++i) {
      if (tet.neighbor(i) >= 0) {
        if (!touched_tets[tet.neighbor(i)]) {
          q.push(tet.neighbor(i));
        }
      }
    }
  }

  assert(t >= 0);

  return t;
}
