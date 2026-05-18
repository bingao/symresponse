#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::Arc;

use tinned::{
    ExpAdjointMap, Expr, OneElecMatrix, PertMultichain, Perturbation, Symbol, TinnedError,
};

pub(crate) fn make_perturbing_operator(
    op_name: &str,
    pert_name: &str,
    freq_name: &str,
) -> Result<(Arc<Perturbation>, Arc<dyn Expr>), TinnedError> {
    let freq = Symbol::new(freq_name);
    let pert = Perturbation::new(pert_name, freq);
    let deps = PertMultichain::from_map(BTreeMap::from([(pert.clone(), 1)]));
    let op = OneElecMatrix::builder(op_name).is_perturbing(true).dependencies(deps).build()?;

    Ok((pert, op))
}

// Builds the couple-cluster similarity-transformed operator
pub(crate) fn make_cc_st_operator(
    cluster_operator: Arc<dyn Expr>,
    operator: Arc<dyn Expr>,
    max_commutator_order: u32,
) -> Result<Arc<dyn Expr>, TinnedError> {
    ExpAdjointMap::builder(cluster_operator, operator, Some(true))
        .left_action(false)
        .max_commutator_order(max_commutator_order)
        .build()
}
