#ifndef _EM_IO_H_
#define _EM_IO_H_ 1

#include <petsc.h>

struct EMContext;

PetscErrorCode read_mdl(const char *, EMContext *);
PetscErrorCode read_emd(const char *, EMContext *);

PetscErrorCode save_mesh(EMContext *, const char *, int);
PetscErrorCode save_rsp(EMContext *, const char *);

#endif
