#include "em_ctx.h"
#include "em_fwd.h"

int main(int argc, char **argv) {
  EMContext ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

  ierr = process_options(&ctx); CHKERRQ(ierr);

  ierr = create_context(&ctx); CHKERRQ(ierr);

  ierr = em_forward(&ctx); CHKERRQ(ierr);

  ierr = destroy_context(&ctx); CHKERRQ(ierr);

  ierr = PetscFinalize(); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
