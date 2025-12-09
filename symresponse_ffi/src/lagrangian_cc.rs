use safer_ffi::prelude::*;
use std::sync::Arc;

use tinned::{Expr, generic_error};
use tinned_ffi::{
    ExprBox, ExprHandle, ExprSlice, TinnedErrorBox, expr_vec_from_slice, tinned_error_new,
    try_with_handle,
};

use symresponse::LagrangianCc;

use crate::lagrangian::{LagrangianBox, LagrangianHandle};

#[ffi_export]
pub fn symresponse_lagrangian_cc_new(
    unperturbed_hamiltonian: &ExprHandle,
    perturbing_operators: Option<&ExprSlice>,
    cc_amplitudes: &ExprHandle,
    excitation_operators: &ExprHandle,
    multipliers: &ExprHandle,
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

    let cc_amplitudes_arc: Arc<dyn Expr> = cc_amplitudes.clone_arc();
    let excitation_operators_arc: Arc<dyn Expr> = excitation_operators.clone_arc();
    let multipliers_arc: Arc<dyn Expr> = multipliers.clone_arc();

    match LagrangianCc::new(
        unperturbed_hamiltonian_arc,
        &perturbing_ops,
        cc_amplitudes_arc,
        excitation_operators_arc,
        multipliers_arc,
    ) {
        Ok(lag) => Some(LagrangianBox::new(LagrangianHandle::new(Arc::new(lag)))),
        Err(e) => {
            tinned_error_new(out_err, e);
            None
        },
    }
}

#[ffi_export]
pub extern "C" fn symresponse_lagrangian_cc_linear_response_rhs(
    h: Option<&LagrangianHandle>,
    rsp_parameter: &ExprHandle,
    out_err: Option<Out<'_, TinnedErrorBox>>,
) -> Option<ExprBox> {
    try_with_handle(h, "symresponse_lagrangian_cc_linear_response_rhs", "LagrangianHandle", |lh| {
        let lag_ref = lh.as_ref();
        if let Some(lag) = lag_ref.as_any().downcast_ref::<LagrangianCc>() {
            let rsp_parameter_arc: Arc<dyn Expr> = rsp_parameter.clone_arc();
            lag.linear_response_rhs(rsp_parameter_arc).map(|arc| ExprBox::new(ExprHandle::new(arc)))
        } else {
            Err(generic_error(
                "Invalid Lagrangian type in symresponse_lagrangian_cc_linear_response_rhs",
                None,
            ))
        }
    })
    .map_or_else(
        |e| {
            tinned_error_new(out_err, e);
            None
        },
        Some,
    )
}
