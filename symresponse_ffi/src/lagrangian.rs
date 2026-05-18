use safer_ffi::prelude::*;
use std::sync::Arc;

use tinned_ffi::{
    ExprBox, ExprHandle, NumberToleranceHandle, PerturbationSlice, TinnedErrorBox,
    perturbation_vec_from_slice, tinned_error_new, try_with_handle,
};

use symresponse::Lagrangian;

/// Opaque handle to a Lagrangian object.
///
/// This type is owned by the library. Users must not access its internals.
/// Use the provided API functions to operate on it.
///
/// Ownership:
/// - Must be freed using `symresponse_lagrangian_free`.
#[derive_ReprC]
#[repr(opaque)]
pub struct LagrangianHandle {
    inner: Arc<dyn Lagrangian>,
}

// Owned box by C after return
pub type LagrangianBox = repr_c::Box<LagrangianHandle>;

impl LagrangianHandle {
    #[inline]
    pub fn new(lag: Arc<dyn Lagrangian>) -> Self {
        Self {
            inner: lag,
        }
    }

    #[inline]
    pub fn as_ref(&self) -> &dyn Lagrangian {
        &*self.inner
    }

    //#[inline]
    //pub fn clone_arc(&self) -> Arc<dyn Lagrangian> {
    //    Arc::clone(&self.inner)
    //}
}

/// Frees a Lagrangian handle.
///
/// This function is NULL-safe. Passing NULL has no effect.
#[ffi_export]
pub fn symresponse_lagrangian_free(lag: Option<LagrangianBox>) {
    drop(lag);
}

/// Returns the Lagrangian expression associated with `h`.
///
/// Parameters:
/// - `h`: Lagrangian handle. Must not be NULL.
/// - `out_err`: Optional output error handle. May be NULL.
///
/// Returns:
/// - A newly allocated expression handle on success.
/// - NULL on failure. If `out_err` is not NULL, it is set to an error object.
///
/// Ownership:
/// - The caller owns the returned expression and must free it with the
///   corresponding Tinned FFI free function.
#[ffi_export]
pub fn symresponse_get_lagrangian(
    h: Option<&LagrangianHandle>,
    out_err: Option<Out<'_, TinnedErrorBox>>,
) -> Option<ExprBox> {
    match try_with_handle(h, "symresponse_get_lagrangian", "LagrangianHandle", |lh| {
        let lag_expr = lh.as_ref().get_lagrangian();
        Ok(ExprBox::new(ExprHandle::new(Arc::clone(lag_expr))))
    }) {
        Ok(expr_box) => Some(expr_box),
        Err(e) => {
            tinned_error_new(out_err, e);
            None
        },
    }
}

/// Computes the response function.
///
/// Parameters:
/// - `exten_perturbations`: Array of extensive perturbations (must not be empty)
/// - `inten_perturbations`: Array of intensive perturbations (may be empty)
/// - `min_wfn_exten_order`: Controls elimination of wave function parameters.
///   See Rust documentation for detailed behavior.
/// - `validate_frequencies`: Whether to validate perturbation frequencies
/// - `num_tol`: Optional numerical tolerance (may be NULL)
///
/// Returns:
/// - Expression handle on success
/// - NULL on failure (see `out_err`)
///
/// Errors:
/// - On failure, `out_err` (if not NULL) will be set.
///
/// See also:
/// Rust API documentation for detailed semantics.
#[ffi_export]
pub fn symresponse_get_response_function(
    h: Option<&LagrangianHandle>,
    exten_slice: Option<&PerturbationSlice>,
    inten_slice: Option<&PerturbationSlice>,
    min_wfn_exten_order: u32,
    validate_frequencies: bool,
    num_tol: Option<&NumberToleranceHandle>,
    out_err: Option<Out<'_, TinnedErrorBox>>,
) -> Option<ExprBox> {
    match try_with_handle(h, "symresponse_response_function", "LagrangianHandle", |lh| {
        let exten_perturbations = match exten_slice {
            Some(slice) => perturbation_vec_from_slice(
                slice,
                "symresponse_response_function:exten_perturbations",
            )?,
            None => Vec::new(),
        };

        let inten_perturbations = match inten_slice {
            Some(slice) => perturbation_vec_from_slice(
                slice,
                "symresponse_response_function:inten_perturbations",
            )?,
            None => Vec::new(),
        };

        let num_tolerance = num_tol.map(|h| h.as_ref().clone());

        lh.as_ref()
            .response_function(
                &exten_perturbations,
                &inten_perturbations,
                min_wfn_exten_order,
                validate_frequencies,
                num_tolerance,
            )
            .map(|arc| ExprBox::new(ExprHandle::new(arc)))
    }) {
        Ok(expr) => Some(expr),
        Err(e) => {
            tinned_error_new(out_err, e);
            None
        },
    }
}

//#[ffi_export]
//pub fn symresponse_get_residue() {
//
//}
