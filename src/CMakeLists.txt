ADD_EXECUTABLE(emfem em_main.cc em_fe.cc em_ctx.cc em_io.cc em_utils.cc em_mesh.cc em_la.cc em_fwd.cc)
TARGET_LINK_LIBRARIES(emfem ${MPI_C_LIBRARIES} ${PETSC_LIBRARY} ${METIS_LIBRARY} ${TETGEN_LIBRARY})
