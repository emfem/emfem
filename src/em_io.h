#ifndef _EM_IO_H_
#define _EM_IO_H_ 1

#include <petsc.h>

struct EMContext;

PetscErrorCode read_mdl(EMContext *);
PetscErrorCode read_mesh(EMContext *);
PetscErrorCode read_emd(EMContext *);

PetscErrorCode save_mesh(EMContext *, const char *, int);
PetscErrorCode save_rsp(EMContext *, const char *);

#endif
