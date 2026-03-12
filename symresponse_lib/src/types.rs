use std::sync::Arc;

use tinned::{Expr, Perturbation};

// One- and two-electron density matrices, containing elimination rule.
// Used for LagrangianOrbCc

// Detailed information of a response function, used by methods
// `find_optimal_response_function()`, `response_function_with_weight()`,
// `find_optimal_elimination_order()`
pub struct ResponseDetail {
    pub expression: Arc<dyn Expr>,
    pub min_wfn_exten: u32,
    pub exten_perturbations: Vec<Arc<Perturbation>>,
    pub inten_perturbations: Vec<Arc<Perturbation>>,
}
