use std::collections::BTreeMap;
use std::sync::Arc;

use tinned::{Expr, OneElecMatrix, PertMultichain, Perturbation, Symbol, TinnedError};

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
