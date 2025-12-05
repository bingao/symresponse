#ifndef SYMRESPONSE_CLEANUP_H
#define SYMRESPONSE_CLEANUP_H

#include "symresponse.h"

#ifdef __cplusplus
extern "C" {
#endif

// Macro helpers: free if non-null, then null the pointer.

#define TINNED_SAFE_FREE_EXPR(p) \
    do { if ((p)) { tinned_expr_free((p)); (p) = NULL; } } while (0)

#define TINNED_SAFE_FREE_ERR(p) \
    do { if ((p)) { tinned_error_free((p)); (p) = NULL; } } while (0)

#define TINNED_SAFE_FREE_STR(p) \
    do { if ((p)) { tinned_string_free((p)); (p) = NULL; } } while (0)

#define TINNED_SAFE_FREE_PERTURBATION(p) \
    do { if ((p)) { tinned_perturbation_free((p)); (p) = NULL; } } while (0)

#define TINNED_SAFE_FREE_PERT_MULTICHAIN(p) \
    do { if ((p)) { tinned_pert_multichain_free((p)); (p) = NULL; } } while (0)

#define TINNED_SAFE_FREE_SUPERCHAINS(p) \
    do { if ((p)) { tinned_expr_superchains_free((p)); (p) = NULL; } } while (0)

#define SYMRESPONSE_SAFE_FREE_LAG(p) \
    do { if ((p)) { symresponse_lagrangian_free((p)); (p) = NULL; } } while (0)

#ifdef __cplusplus
}
#endif

#endif