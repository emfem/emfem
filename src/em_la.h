#ifndef _EM_LA_H_
#define _EM_LA_H_ 1

#include <petsc.h>

struct EMContext;

PetscErrorCode access_vec(Vec, std::vector<PetscInt> &, int, double *);

PetscErrorCode setup_ams(EMContext *);
PetscErrorCode destroy_ams(EMContext *);

PetscErrorCode create_pc(EMContext *);
PetscErrorCode destroy_pc(EMContext *);
PetscErrorCode pc_apply_b(PC, Vec, Vec);

PetscErrorCode matshell_createvecs_a(Mat, Vec *, Vec *);
PetscErrorCode matshell_mult_a(Mat, Vec, Vec);

PetscErrorCode solve_linear_system(EMContext *, const PETScBlockVector &, PETScBlockVector &, PetscInt, PetscReal);

#endif
