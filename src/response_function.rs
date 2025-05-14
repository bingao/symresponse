use std::sync::Arc;

use tinned::{Expr, Perturbation};

// Return result of the function `find_optimal_response_function()`
pub struct ResponseFunction {
    expression: Arc<dyn Expr>,
    min_wfn_exten: u32,
    exten_perturbations: Vec<Arc<Perturbation>>,
    inten_perturbations: Vec<Arc<Perturbation>>,
}

impl ResponseFunction {
    #[inline]
    pub fn new(
        expression: Arc<dyn Expr>,
        min_wfn_exten: u32,
        exten_perturbations: Vec<Arc<Perturbation>>,
        inten_perturbations: Vec<Arc<Perturbation>>,
    ) -> Self {
        Self {
            expression,
            min_wfn_exten,
            exten_perturbations,
            inten_perturbations,
        }
    }

    #[inline]
    pub fn expression(&self) -> &Arc<dyn Expr> {
        &self.expression
    }

    #[inline]
    pub fn min_wfn_exten(&self) -> u32 {
        self.min_wfn_exten
    }

    #[inline]
    pub fn exten_perturbations(&self) -> &[Arc<Perturbation>] {
        &self.exten_perturbations
    }

    #[inline]
    pub fn inten_perturbations(&self) -> &[Arc<Perturbation>] {
        &self.inten_perturbations
    }
}
