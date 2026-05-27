use std::sync::Arc;

use tinned::{ExpAdjointMap, Expr, TinnedError};

// Builds the couple-cluster similarity-transformed operator
pub(crate) fn make_cc_st_operator(
    cluster_operator: Arc<dyn Expr>,
    operator: Arc<dyn Expr>,
    max_commutator_order: u32,
) -> Result<Arc<dyn Expr>, TinnedError> {
    ExpAdjointMap::builder(cluster_operator, operator, Some(true), None)
        .left_action(false)
        .max_commutator_order(max_commutator_order)
        .build()
}
