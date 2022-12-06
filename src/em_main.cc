#include "em_ctx.h"
#include "em_fwd.h"

int main(int argc, char **argv) {
  EMContext ctx;

  PetscFunctionBegin;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  PetscCall(process_options(&ctx));

  PetscCall(create_context(&ctx));

  PetscCall(em_forward(&ctx));

  PetscCall(destroy_context(&ctx));

  PetscCall(PetscFinalize());

  PetscFunctionReturn(0);
}
