use safer_ffi::prelude::*;
use std::sync::Arc;

use tinned::NumberTolerance;
use tinned_ffi::{
    ExprBox, ExprHandle, NumberToleranceHandle, PerturbationSlice, TinnedErrorBox,
    perturbation_vec_from_slice, tinned_error_new, try_with_handle,
};

use symresponse::Lagrangian;

/// An *opaque* handle that C can only pass around
#[derive_ReprC]
#[repr(opaque)]
pub struct LagrangianHandle {
    inner: Arc<dyn Lagrangian>,
}

/// Owned box by C after return
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

// Free a Lagrangian (NULL-safe).
#[ffi_export]
pub fn symresponse_lagrangian_free(lag: Option<LagrangianBox>) {
    drop(lag);
}

#[ffi_export]
pub fn symresponse_response_function(
    h: Option<&LagrangianHandle>,
    exten_slice: Option<&PerturbationSlice>,
    inten_slice: Option<&PerturbationSlice>,
    min_wfn_exten: u32,
    validate_frequencies: bool,
    num_tol: Option<&NumberToleranceHandle>,
    out_err: Option<Out<'_, TinnedErrorBox>>,
) -> Option<ExprBox> {
    try_with_handle(h, "symresponse_response_function", "LagrangianHandle", |lh| {
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

        let num_tolerance: Option<NumberTolerance> = num_tol.map(|h| h.as_ref().clone());

        let lag_ref: &dyn Lagrangian = lh.as_ref();
        lag_ref
            .response_function(
                &exten_perturbations,
                &inten_perturbations,
                min_wfn_exten,
                validate_frequencies,
                num_tolerance,
            )
            .map(|arc| ExprBox::new(ExprHandle::new(arc)))
    })
    .map_or_else(
        |e| {
            tinned_error_new(out_err, e);
            None
        },
        Some,
    )
}

//#[ffi_export]
//pub fn symresponse_residue() {
//
//}
