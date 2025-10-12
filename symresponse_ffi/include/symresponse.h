#ifndef __SYMRESPONSE_H__
#define __SYMRESPONSE_H__

#include "tinned.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \brief
 *  An *opaque* handle that C can only pass around
 */
typedef struct LagrangianHandle LagrangianHandle_t;

/** <No documentation available> */
ExprHandle_t *
symresponse_lagrangian_dao_linear_response_rhs (
    LagrangianHandle_t const * h,
    ExprHandle_t const * density_freq,
    ExprHandle_t const * density_part,
    NumberToleranceHandle_t const * num_tol,
    TinnedErrorHandle_t * * out_err);

#include <stddef.h>
#include <stdint.h>

/** <No documentation available> */
LagrangianHandle_t *
symresponse_lagrangian_dao_new (
    PerturbationHandle_t const * perturbation_a,
    ExprHandle_t const * density_matrix,
    ExprHandle_t const * overlap_matrix,
    slice_ref_ExprHandle_const_ptr_t one_elec_operators,
    ExprHandle_t const * two_elec_operator,
    ExprHandle_t const * xc_energy,
    ExprHandle_t const * xc_potential,
    ExprHandle_t const * h_nuc,
    NumberToleranceHandle_t const * num_tol,
    TinnedErrorHandle_t * * out_err);

/** <No documentation available> */
void
symresponse_lagrangian_free (
    LagrangianHandle_t * lag);

#include <stdbool.h>

#ifdef __cplusplus
} /* extern \"C\" */
#endif

#endif /* __SYMRESPONSE_H__ */
