#include "em_mesh.h"

#include <metis.h>

#include <petsc.h>

#include <queue>
#include <tuple>

void Mesh::create(const std::vector<Point> &vertices, const Eigen::MatrixXi &tets,
                  const std::vector<int> &tet_attributes) {
  this->vertices_ = vertices;
  this->tetrahedra_ = tets;
  this->tet_attributes_ = tet_attributes;

  build_edge_info();
  build_neighbor_info();
  build_boundary_info();

  init_kd_tree();

  partition();
}

void Mesh::create_from_tetgen(tetgenio *in) {
  int i, t;
  Eigen::MatrixXi tetrahedra;
  std::vector<Point> vertices;
  std::vector<int> tet_attributes;

  if (tio_) {
    delete tio_;
    tio_ = NULL;
  }
  tio_ = new tetgenio;

  if (in->tetrahedronlist) {
    tetrahedralize((char *)"zDprq1.2AafQ", in, tio_, NULL);
  } else {
    tetrahedralize((char *)"zDpq1.2AafQ", in, tio_, NULL);
  }

  vertices.resize(tio_->numberofpoints);
  for (i = 0; i < tio_->numberofpoints; ++i) {
    vertices[i][0] = tio_->pointlist[i * 3 + 0];
    vertices[i][1] = tio_->pointlist[i * 3 + 1];
    vertices[i][2] = tio_->pointlist[i * 3 + 2];
  }

  tetrahedra.resize(tio_->numberoftetrahedra, 4);
  tet_attributes.resize(tio_->numberoftetrahedra);
  for (t = 0; t < tio_->numberoftetrahedra; ++t) {
    tetrahedra(t, 0) = tio_->tetrahedronlist[t * 4 + 0];
    tetrahedra(t, 1) = tio_->tetrahedronlist[t * 4 + 1];
    tetrahedra(t, 2) = tio_->tetrahedronlist[t * 4 + 2];
    tetrahedra(t, 3) = tio_->tetrahedronlist[t * 4 + 3];
    tet_attributes[t] = tio_->tetrahedronattributelist[t];
  }

  create(vertices, tetrahedra, tet_attributes);
}

void Mesh::copy(const Mesh &m) {
  create_from_tetgen(m.tio_);
}

void Mesh::refine_tetgen(const std::vector<bool> &flag) {
  int t, nt, i;
  tetgenio *out;
  Eigen::MatrixXi tetrahedra;
  std::vector<Point> vertices;
  std::vector<int> tet_attributes;

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

  out = new tetgenio;
  tetrahedralize((char *)"zDprq1.2AafQ", tio_, out, NULL);

  delete tio_;
  tio_ = out;

  vertices.resize(tio_->numberofpoints);
  for (i = 0; i < tio_->numberofpoints; ++i) {
    vertices[i][0] = tio_->pointlist[i * 3 + 0];
    vertices[i][1] = tio_->pointlist[i * 3 + 1];
    vertices[i][2] = tio_->pointlist[i * 3 + 2];
  }

  tetrahedra.resize(tio_->numberoftetrahedra, VERTICES_PER_TET);
  tet_attributes.resize(tio_->numberoftetrahedra);
  for (t = 0; t < tio_->numberoftetrahedra; ++t) {
    tetrahedra(t, 0) = tio_->tetrahedronlist[t * 4 + 0];
    tetrahedra(t, 1) = tio_->tetrahedronlist[t * 4 + 1];
    tetrahedra(t, 2) = tio_->tetrahedronlist[t * 4 + 2];
    tetrahedra(t, 3) = tio_->tetrahedronlist[t * 4 + 3];
    tet_attributes[t] = tio_->tetrahedronattributelist[t];
  }

  create(vertices, tetrahedra, tet_attributes);
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
    PetscFPrintf(comm_, fp, "%.17g %.17g %.17g\n", vertices_[i][0], vertices_[i][1],
                 vertices_[i][2]);
  }
  PetscFPrintf(comm_, fp, "\n");

  PetscFPrintf(comm_, fp, "CELLS %d %d\n", n_cells, n_cells * (4 + 1));
  for (i = 0; i < n_cells; ++i) {
    PetscFPrintf(comm_, fp, "%d", VERTICES_PER_TET);
    for (j = 0; j < VERTICES_PER_TET; ++j) {
      PetscFPrintf(comm_, fp, " %4d", tetrahedra_(i, j));
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
    PetscFPrintf(comm_, fp, "%.15g\n", (double)tet_attributes_[i]);
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
      if (tet_neighbors_(i, j) >= 0) {
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
      if (tet_neighbors_(i, j) >= 0) {
        adjncy[nnz++] = tet_neighbors_(i, j);
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
  compute_new_edge_indices();
}

void Mesh::get_edge_owners(std::vector<int> &owners) {
  int t, e;
  bool coin_flip;

  owners.resize(n_edges());
  std::fill(owners.begin(), owners.end(), -1);

  coin_flip = true;
  for (t = 0; t < n_tets(); ++t) {
    for (e = 0; e < EDGES_PER_TET; ++e) {
      if (owners[edge_number_[tet_edges_(t, e)]] < 0) {
        owners[edge_number_[tet_edges_(t, e)]] = subdomain_[t];
      } else {
        if (coin_flip == true) {
          owners[edge_number_[tet_edges_(t, e)]] = subdomain_[t];
        }
        coin_flip = !coin_flip;
      }
    }
  }
}

void Mesh::compute_new_edge_indices() {
  std::vector<int> owners;
  int e, subdomain, next_free_index;

  get_edge_owners(owners);

  next_free_index = 0;

  for (subdomain = 0; subdomain < comm_size_; ++subdomain) {
    for (e = 0; e < n_edges(); ++e) {
      if (owners[e] == subdomain) {
        edge_number_[e] = next_free_index++;
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
        vertices_on_subdomain[tetrahedra_(t, v)] = true;
      }
    }
  }

  ghost_flag_.resize(n_tets());
  std::fill(ghost_flag_.begin(), ghost_flag_.end(), false);
  for (t = 0; t < n_tets(); ++t) {
    if (subdomain_[t] != comm_rank_) {
      for (v = 0; v < VERTICES_PER_TET; ++v) {
        if (vertices_on_subdomain[tetrahedra_(t, v)]) {
          ghost_flag_[t] = true;
          break;
        }
      }
    }
  }
}

void Mesh::build_edge_info() {
  struct Edge {
    Edge() { this->es_ = this->ee_ = -1; }

    Edge(int es, int ee) {
      this->es_ = std::min(es, ee);
      this->ee_ = std::max(es, ee);
    }

    bool operator==(const Edge &e) const { return (e.es_ == this->es_ && e.ee_ == this->ee_); }

    bool operator<(const Edge &e) const { return (es_ < e.es_ || (es_ == e.es_ && ee_ < e.ee_)); }

    int es_, ee_;
  };

  typedef std::tuple<Edge, int, int> TetEdge;

  int i, t, e, newe, ne;
  std::vector<TetEdge> edges;
  int lec[EDGES_PER_TET][VERTICES_PER_EDGE] = { { 0, 1 }, { 0, 2 }, { 0, 3 },
                                                { 1, 2 }, { 1, 3 }, { 2, 3 } };

  edges.resize(n_tets() * EDGES_PER_TET);

  for (t = 0; t < n_tets(); ++t) {
    for (i = 0; i < EDGES_PER_TET; ++i) {
      edges[t * EDGES_PER_TET + i] =
          std::make_tuple(Edge(tetrahedra_(t, lec[i][0]), tetrahedra_(t, lec[i][1])), t, i);
    }
  }

  std::sort(edges.begin(), edges.end());
  tet_edges_.resize(n_tets(), EDGES_PER_TET);

  e = 0;
  newe = 0;
  ne = 0;
  while (newe < n_tets() * EDGES_PER_TET) {
    while (newe < n_tets() * EDGES_PER_TET) {
      if (std::get<0>(edges[e]) == std::get<0>(edges[newe])) {
        ++newe;
      } else {
        break;
      }
    }

    e = newe;
    ++ne;
  }

  edges_.resize(ne, VERTICES_PER_EDGE);

  e = 0;
  newe = 0;
  ne = 0;
  while (e < n_tets() * EDGES_PER_TET) {
    while (newe < n_tets() * EDGES_PER_TET) {
      if (std::get<0>(edges[e]) == std::get<0>(edges[newe])) {
        ++newe;
      } else {
        break;
      }
    }

    edges_(ne, 0) = std::get<0>(edges[e]).es_;
    edges_(ne, 1) = std::get<0>(edges[e]).ee_;

    for (; e < newe; ++e) {
      tet_edges_(std::get<1>(edges[e]), std::get<2>(edges[e])) = ne;
    }

    ++ne;
  }

  edge_number_.resize(n_edges());
  for (e = 0; e < n_edges(); ++e) {
    edge_number_[e] = e;
  }
}

void Mesh::build_neighbor_info() {
  struct Face {
    Face() { nodes_[0] = nodes_[1] = nodes_[2] = -1; };

    Face(int *nodes) {
      std::copy(nodes, nodes + VERTICES_PER_TRIANGLE, nodes_);
      std::sort(nodes_, nodes_ + VERTICES_PER_TRIANGLE);
    }

    bool operator<(const Face &f) const {
      return (nodes_[0] < f.nodes_[0]) || (nodes_[0] == f.nodes_[0] && nodes_[1] < f.nodes_[1]) ||
             (nodes_[0] == f.nodes_[0] && nodes_[1] == f.nodes_[1] && nodes_[2] < f.nodes_[2]);
    }

    bool operator==(const Face &f) {
      return (nodes_[0] == f.nodes_[0] && nodes_[1] == f.nodes_[1] && nodes_[2] == f.nodes_[2]);
    }

    int nodes_[VERTICES_PER_TRIANGLE];
  };

  typedef std::tuple<Face, int, int> TetFace;

  std::vector<TetFace> faces;
  int i, j, t, f, nodes[VERTICES_PER_TRIANGLE];

  faces.resize(n_tets() * TRIANGLES_PER_TET);

  for (t = 0; t < n_tets(); ++t) {
    for (i = 0; i < TRIANGLES_PER_TET; ++i) {
      for (j = 0; j < VERTICES_PER_TRIANGLE; ++j) {
        nodes[j] = tetrahedra_(t, (i + j + 1) % VERTICES_PER_TET);
      }
      faces[t * TRIANGLES_PER_TET + i] = std::make_tuple(Face(nodes), t, i);
    }
  }

  std::sort(faces.begin(), faces.end());

  tet_neighbors_.resize(n_tets(), TRIANGLES_PER_TET);
  tet_neighbors_.setConstant(-1);
  tet_neighbor_of_neighbor_.resize(n_tets(), TRIANGLES_PER_TET);
  tet_neighbor_of_neighbor_.setConstant(-1);

  f = 0;
  while (true) {
    if (TRIANGLES_PER_TET * n_tets() - 1 <= f) {
      break;
    }

    if (std::get<0>(faces[f]) == std::get<0>(faces[f + 1])) {
      tet_neighbors_(std::get<1>(faces[f]), std::get<2>(faces[f])) = std::get<1>(faces[f + 1]);
      tet_neighbor_of_neighbor_(std::get<1>(faces[f]), std::get<2>(faces[f])) =
          std::get<2>(faces[f + 1]);
      tet_neighbors_(std::get<1>(faces[f + 1]), std::get<2>(faces[f + 1])) = std::get<1>(faces[f]);
      tet_neighbor_of_neighbor_(std::get<1>(faces[f + 1]), std::get<2>(faces[f + 1])) =
          std::get<2>(faces[f]);
      f += 2;
    } else {
      f += 1;
    }
  }
}

void Mesh::build_boundary_info() {
  int t, f, i;
  int lec[TRIANGLES_PER_TET]
         [EDGES_PER_TRIANGLE] = { { 3, 4, 5 }, { 1, 2, 5 }, { 0, 2, 4 }, { 0, 1, 3 } };

  vertex_bdr_marker_.resize(n_vertices());
  std::fill(vertex_bdr_marker_.begin(), vertex_bdr_marker_.end(), false);

  edge_bdr_marker_.resize(n_edges());
  std::fill(edge_bdr_marker_.begin(), edge_bdr_marker_.end(), false);

  for (t = 0; t < n_tets(); ++t) {
    for (f = 0; f < TRIANGLES_PER_TET; ++f) {
      if (tet_neighbors_(t, f) < 0) {
        for (i = 0; i < VERTICES_PER_TRIANGLE; ++i) {
          vertex_bdr_marker_[tetrahedra_(t, (f + i + 1) % 4)] = true;
        }
        for (i = 0; i < EDGES_PER_TRIANGLE; ++i) {
          edge_bdr_marker_[tet_edges_(t, lec[f][i])] = true;
        }
      }
    }
  }
}

void Mesh::init_kd_tree() {
  int i, j;

  kd_index_.reset(new KDTree(3, *this, 100));
  kd_index_->buildIndex();

  vertex_to_tet_.resize(n_vertices());
  for (i = 0; i < n_tets(); ++i) {
    for (j = 0; j < VERTICES_PER_TET; ++j) {
      vertex_to_tet_[tetrahedra_(i, j)] = i;
    }
  }
}

int Mesh::find_closest_vertex(const Point &p) {
  size_t idx;
  double dist;

  kd_index_->knnSearch(&p[0], 1, &idx, &dist);

  return idx;
}

int Mesh::find_cell_around_point(const Point &p) {
  int i, tidx, t;
  TetAccessor tet;
  std::queue<int> q;
  std::vector<bool> touched_tets;

  touched_tets.resize(n_tets());
  std::fill(touched_tets.begin(), touched_tets.end(), false);

  q.push(vertex_to_tet_[find_closest_vertex(p)]);

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
