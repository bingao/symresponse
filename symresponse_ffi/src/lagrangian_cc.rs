use safer_ffi::prelude::*;
use std::sync::Arc;

use tinned::{Expr, NumberTolerance, generic_error};
use tinned_ffi::{
    ExprBox, ExprHandle, ExprSlice, NumberToleranceHandle, TinnedErrorBox, expr_vec_from_slice,
    tinned_error_new, try_with_handle,
};

use symresponse::LagrangianCc;

use crate::lagrangian::{LagrangianBox, LagrangianHandle};

#[ffi_export]
pub fn symresponse_lagrangian_cc_new(
    unperturbed_hamiltonian: &ExprHandle,
    perturbing_operators: Option<&ExprSlice>,
    cc_amplitude: &ExprHandle,
    cc_excitation_operator: &ExprHandle,
    cc_multiplier: &ExprHandle,
    out_err: Option<Out<'_, TinnedErrorBox>>,
) -> Option<LagrangianBox> {
    let unperturbed_hamiltonian_arc: Arc<dyn Expr> = unperturbed_hamiltonian.clone_arc();

    let perturbing_ops = match perturbing_operators {
        Some(slice) => match expr_vec_from_slice(slice, "symresponse_lagrangian_cc_new") {
            Ok(v) => v,
            Err(e) => {
                tinned_error_new(out_err, e);
                return None;
            },
        },
        None => Vec::new(),
    };

    let cc_amplitude_arc: Arc<dyn Expr> = cc_amplitude.clone_arc();
    let cc_excitation_operator_arc: Arc<dyn Expr> = cc_excitation_operator.clone_arc();
    let cc_multiplier_arc: Arc<dyn Expr> = cc_multiplier.clone_arc();

    match LagrangianCc::new(
        unperturbed_hamiltonian_arc,
        &perturbing_ops,
        cc_amplitude_arc,
        cc_excitation_operator_arc,
        cc_multiplier_arc,
    ) {
        Ok(lag) => Some(LagrangianBox::new(LagrangianHandle::new(Arc::new(lag)))),
        Err(e) => {
            tinned_error_new(out_err, e);
            None
        },
    }
}

#[ffi_export]
pub extern "C" fn symresponse_cc_linear_response_rhs(
    h: Option<&LagrangianHandle>,
    rsp_parameter: &ExprHandle,
    num_tol: Option<&NumberToleranceHandle>,
    out_err: Option<Out<'_, TinnedErrorBox>>,
) -> Option<ExprBox> {
    match try_with_handle(h, "symresponse_cc_linear_response_rhs", "LagrangianHandle", |lh| {
        if let Some(lag) = lh.as_ref().as_any().downcast_ref::<LagrangianCc>() {
            let rsp_parameter_arc: Arc<dyn Expr> = rsp_parameter.clone_arc();
            let num_tolerance: Option<NumberTolerance> = num_tol.map(|h| h.as_ref().clone());
            lag.linear_response_rhs(&rsp_parameter_arc, num_tolerance)
                .map(|arc| ExprBox::new(ExprHandle::new(arc)))
        } else {
            Err(generic_error(
                "Invalid Lagrangian type in symresponse_cc_linear_response_rhs",
                None,
            ))
        }
    }) {
        Ok(expr) => Some(expr),
        Err(e) => {
            tinned_error_new(out_err, e);
            None
        },
    }
}
