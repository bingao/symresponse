use safer_ffi::prelude::*;
use std::sync::Arc;

use tinned::{Expr, NumberTolerance, Perturbation, generic_error};
use tinned_ffi::{
    ExprBox, ExprHandle, ExprSlice, NumberToleranceHandle, PerturbationHandle, TinnedErrorBox,
    expr_vec_from_slice, tinned_error_new, try_with_handle,
};

use symresponse::{LagrangianDao, SymmetrizeMode};

use crate::lagrangian::{LagrangianBox, LagrangianHandle};

#[derive_ReprC]
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymmetrizeModeReprC {
    Always = 0,
    Never = 1,
    Auto = 2,
}

impl From<SymmetrizeModeReprC> for SymmetrizeMode {
    fn from(v: SymmetrizeModeReprC) -> Self {
        match v {
            SymmetrizeModeReprC::Always => SymmetrizeMode::Always,
            SymmetrizeModeReprC::Never => SymmetrizeMode::Never,
            SymmetrizeModeReprC::Auto => SymmetrizeMode::Auto,
        }
    }
}

#[ffi_export]
pub fn symresponse_lagrangian_dao_new(
    perturbation_a: &PerturbationHandle,
    density_matrix: &ExprHandle,
    overlap_matrix: Option<&ExprHandle>,
    one_elec_operators: Option<&ExprSlice>,
    two_elec_operator: Option<&ExprHandle>,
    xc_energy: Option<&ExprHandle>,
    xc_potential: Option<&ExprHandle>,
    h_nuc: Option<&ExprHandle>,
    symmetrized_mode: Option<&SymmetrizeModeReprC>,
    num_tol: Option<&NumberToleranceHandle>,
    out_err: Option<Out<'_, TinnedErrorBox>>,
) -> Option<LagrangianBox> {
    let pert_a_arc: Arc<Perturbation> = perturbation_a.clone_arc();
    let dens_mat_arc: Arc<dyn Expr> = density_matrix.clone_arc();

    let overlap_arc: Option<Arc<dyn Expr>> = overlap_matrix.map(|h| h.clone_arc());
    let two_elec_arc: Option<Arc<dyn Expr>> = two_elec_operator.map(|h| h.clone_arc());
    let xc_e_arc: Option<Arc<dyn Expr>> = xc_energy.map(|h| h.clone_arc());
    let xc_v_arc: Option<Arc<dyn Expr>> = xc_potential.map(|h| h.clone_arc());
    let h_nuc_arc: Option<Arc<dyn Expr>> = h_nuc.map(|h| h.clone_arc());

    let one_elec_ops: Vec<Arc<dyn Expr>> = match one_elec_operators {
        Some(slice) => match expr_vec_from_slice(slice, "symresponse_lagrangian_dao_new") {
            Ok(v) => v,
            Err(e) => {
                tinned_error_new(out_err, e);
                return None;
            },
        },
        None => Vec::new(),
    };

    let num_tolerance: Option<NumberTolerance> = num_tol.map(|h| h.as_ref().clone());

    match LagrangianDao::new(
        pert_a_arc,
        dens_mat_arc,
        overlap_arc,
        &one_elec_ops,
        two_elec_arc,
        xc_e_arc,
        xc_v_arc,
        h_nuc_arc,
        symmetrized_mode.copied().map(Into::into),
        num_tolerance,
    ) {
        Ok(lag) => Some(LagrangianBox::new(LagrangianHandle::new(Arc::new(lag)))),
        Err(e) => {
            tinned_error_new(out_err, e);
            None
        },
    }
}

#[ffi_export]
pub extern "C" fn symresponse_lagrangian_dao_linear_response_rhs(
    h: Option<&LagrangianHandle>,
    density_freq: &ExprHandle,
    density_part: &ExprHandle,
    num_tol: Option<&NumberToleranceHandle>,
    out_err: Option<Out<'_, TinnedErrorBox>>,
) -> Option<ExprBox> {
    try_with_handle(h, "symresponse_lagrangian_dao_linear_response_rhs", "LagrangianHandle", |lh| {
        let lag_ref = lh.as_ref();
        if let Some(lag) = lag_ref.as_any().downcast_ref::<LagrangianDao>() {
            let dens_freq_arc: Arc<dyn Expr> = density_freq.clone_arc();
            let dens_part_arc: Arc<dyn Expr> = density_part.clone_arc();
            let num_tolerance: Option<NumberTolerance> = num_tol.map(|h| h.as_ref().clone());
            lag.linear_response_rhs(&dens_freq_arc, &dens_part_arc, num_tolerance)
                .map(|arc| ExprBox::new(ExprHandle::new(arc)))
        } else {
            Err(generic_error(
                "Invalid Lagrangian type in symresponse_lagrangian_dao_linear_response_rhs",
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

#[ffi_export]
pub extern "C" fn symresponse_lagrangian_dao_particular_density_solution(
    h: Option<&LagrangianHandle>,
    density_freq: &ExprHandle,
    num_tol: Option<&NumberToleranceHandle>,
    out_err: Option<Out<'_, TinnedErrorBox>>,
) -> Option<ExprBox> {
    try_with_handle(h, "symresponse_lagrangian_dao_particular_density_solution", "LagrangianHandle", |lh| {
        let lag_ref = lh.as_ref();
        if let Some(lag) = lag_ref.as_any().downcast_ref::<LagrangianDao>() {
            let dens_freq_arc: Arc<dyn Expr> = density_freq.clone_arc();
            let num_tolerance: Option<NumberTolerance> = num_tol.map(|h| h.as_ref().clone());
            lag.particular_density_solution(&dens_freq_arc, num_tolerance)
                .map(|arc| ExprBox::new(ExprHandle::new(arc)))
        } else {
            Err(generic_error(
                "Invalid Lagrangian type in symresponse_lagrangian_dao_particular_density_solution",
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
