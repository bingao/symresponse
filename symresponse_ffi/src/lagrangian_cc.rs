use safer_ffi::prelude::*;
use std::sync::Arc;

use tinned::{Expr, NumberTolerance, generic_error};
use tinned_ffi::{
    ExprBox, ExprHandle, ExprSlice, NumberToleranceHandle, TinnedErrorBox, expr_vec_from_slice,
    tinned_error_new, try_with_handle,
};

use symresponse::LagrangianCc;

use crate::lagrangian::{LagrangianBox, LagrangianHandle};

/// Creates a new coupled-cluster time-averaged quasienergy Lagrangian.
///
/// This function builds the symbolic ingredients needed to compute
/// coupled-cluster response functions and residues through the generic
/// Lagrangian C FFI functions.
///
/// The constructed Lagrangian contains the coupled-cluster amplitudes,
/// Lagrangian multipliers, time-dependent cluster operator, Lambda operator,
/// coupled-cluster quasienergy, multiplier response equation, and full
/// time-averaged quasienergy Lagrangian.
///
/// Parameters:
/// - `unperturbed_hamiltonian`: Unperturbed Hamiltonian expression. Must not
///   be NULL.
/// - `perturbing_operators`: Optional array of perturbing operator
///   expressions. May be NULL, which is equivalent to an empty array. These
///   operators must not contain zeroth-order/unperturbed terms.
/// - `cc_amplitude`: Coupled-cluster amplitude expression. Must not be NULL
///   and must be created using the function `tinned_wfn_parameter_new`.
/// - `cc_excitation_operator`: Coupled-cluster excitation operator expression.
///   Must not be NULL and must be created using the function
///   `tinned_excitation_operator_new`.
/// - `cc_multiplier`: Coupled-cluster Lagrangian multiplier expression. Must
///   not be NULL and must be created using the function
///   `tinned_lag_multiplier_new`.
/// - `out_err`: Optional output error handle. May be NULL.
///
/// Returns:
/// - A newly allocated Lagrangian handle on success.
/// - NULL on failure.
///
/// Ownership:
/// - The caller owns the returned handle and must free it with
///   `symresponse_lagrangian_free`.
///
/// Errors:
/// - If an error occurs, NULL is returned.
/// - If `out_err` is not NULL, it is set to a newly allocated error handle.
/// - Errors occur if any perturbing operator contains an unperturbed term, if
///   any required expression has an unsupported type, or if symbolic
///   construction fails.
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

/// Builds the right-hand side of a coupled-cluster linear response equation.
///
/// The response parameter must be derived from this Lagrangian's
/// coupled-cluster amplitude or Lagrangian multiplier. It may also be a
/// residue parameter whose inner parameter is derived from one of those
/// quantities.
///
/// Parameters:
/// - `h`: Coupled-cluster Lagrangian handle. Must not be NULL and must refer to
///   a Lagrangian created by `symresponse_lagrangian_cc_new`.
/// - `rsp_parameter`: Response parameter expression. Must not be NULL. It must
///   be a `tinned::WfnParameter`, `tinned::LagMultiplier`, or compatible
///   `tinned::ResidueParameter`.
/// - `num_tol`: Optional numerical tolerance. May be NULL.
/// - `out_err`: Optional output error handle. May be NULL.
///
/// Returns:
/// - A newly allocated expression handle on success.
/// - NULL on failure.
///
/// Ownership:
/// - The caller owns the returned expression handle and must free it with the
///   corresponding Tinned FFI expression free function.
///
/// Errors:
/// - If `h` is NULL, has the wrong Lagrangian type, or computation fails,
///   NULL is returned.
/// - If `rsp_parameter` has an unsupported type, is not derived from this
///   Lagrangian's coupled-cluster amplitude or multiplier, or represents an
///   invalid residue response parameter, NULL is returned.
/// - If `out_err` is not NULL, it is set to a newly allocated error handle.
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
