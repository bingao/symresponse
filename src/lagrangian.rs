use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use typetag;

use tinned::{
    Expr, NumberTolerance, Perturbation, TinnedError, differentiate_expr, generic_error,
    is_zero_expr,
};

use crate::lagrangian_internal::sealed::LagrangianInternal;

// Return result of the function `find_optimal_response_function()`
pub struct ResponseFunction {
    expression: Arc<dyn Expr>,
    min_wfn_exten: u32,
    exten_perturbations: Vec<Arc<Perturbation>>,
    inten_perturbations: Vec<Arc<Perturbation>>,
}

// Base Lagrangian trait
#[typetag::serde]
pub trait Lagrangian: std::fmt::Debug + Send + Sync + LagrangianInternal {
    // Return response function according to given extensive and intensive
    // perturbations, and minimum order of differentiated wave function
    // parameters to be eliminated, with respect to extensive perturbations.
    //
    // `exten_perturbations` and `inten_perturbations` contain, respectively,
    // extensive and intensive perturbations. The former must contain at least
    // one extensive perturbation, while the latter can be empty.
    //
    // `min_wfn_exten` as 0, means it will be automatically determined as the
    // next integer of the floor function of the half number of extensive
    // perturbations. For `min_wfn_exten` greater than the number of extensive
    // perturbations, it means no elimination of wave function parameters so
    // that more Lagrangian multipliers can be eliminated.
    fn response_function(
        &self,
        exten_perturbations: &[Arc<Perturbation>],
        inten_perturbations: &[Arc<Perturbation>],
        min_wfn_exten: u32,
        num_tol: Option<NumberTolerance>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        // Validate that: (i) at least one extensive perturbation, (ii) sum of
        // all perturbations' frequencies is zero, and (iii) extensive and
        // intensive perturbations are disjoint
        if exten_perturbations.is_empty() {
            return Err(generic_error(
                "Lagrangian requires at least one extensive perturbation",
                None,
            ));
        }
        if self.is_non_zero_sum_freqs(exten_perturbations, inten_perturbations, num_tol.clone())? {
            return Err(generic_error(
                "Lagrangian gets perturbations with non-zero sum frequencies",
                None,
            ));
        }
        if self.has_common_perturbation(exten_perturbations, inten_perturbations) {
            return Err(generic_error(
                "Lagrangian gets same extensive and intensive perturbation(s)",
                None,
            ));
        }

        // Differentiate the quasi-energy (derivative) Lagrangian
        let mut result =
            differentiate_expr(self.get_lagrangian(), &exten_perturbations).map_err(|e| {
                generic_error(
                    "Differentiation with respect to extensive perturbations failed",
                    Some(Box::new(e)),
                )
            })?;
        result = differentiate_expr(&result, &inten_perturbations).map_err(|e| {
            generic_error(
                "Differentiation with respect to intensive perturbations failed",
                Some(Box::new(e)),
            )
        })?;
        // Usually the differentiated quasi-energy Lagrangian cannot be zero
        if is_zero_expr(&result, num_tol.clone()) {
            return Ok(result);
        }

        // Minimum order for the elimination of wave function parameters is the
        // next integer of the floor function of the half number of
        // perturbations, according to Table IV, J. Chem. Phys. 129, 214103 (2008)
        let mut min_wfn_order: u32 = 1 + (exten_perturbations.len() / 2) as u32;
        if min_wfn_exten > 0 {
            if min_wfn_exten < min_wfn_order {
                return Err(generic_error(
                    &format!("Invalid minimum order {}", min_wfn_exten),
                    None,
                ));
            } else {
                min_wfn_order = min_wfn_exten;
            }
        }

        // Eliminate wave function parameter
        if min_wfn_exten <= exten_perturbations.len() as u32 {
            result = self
                .eliminate_wfn_parameter(&result, exten_perturbations, min_wfn_order)
                .map_err(|e| {
                    generic_error(
                        "Elimination of wave function parameter failed",
                        Some(Box::new(e)),
                    )
                })?;
        }

        // Minimum order for the elimination of Lagrangian multipliers,
        // see Table V, J. Chem. Phys. 129, 214103 (2008)
        let min_multiplier_order: u32 = if min_wfn_exten <= exten_perturbations.len() as u32 {
            exten_perturbations.len() as u32 - min_wfn_order + 1
        } else {
            0
        };

        // Eliminate Lagrangian multipliers
        result = self
            .eliminate_lag_multipliers(&result, exten_perturbations, min_multiplier_order)
            .map_err(|e| {
                generic_error(
                    "Elimination of Lagrangian multipliers failed",
                    Some(Box::new(e)),
                )
            })?;

        // Usually `result` cannot be zero after elimination
        if is_zero_expr(&result, num_tol) {
            return Ok(result);
        }

        // Evaluation at zero perturbation strength
        self.at_zero_strength(&result)
    }

    // Return optimal response function(s) by performing different elimination
    // rules. Optimal response function(s) has a minimal weight as determined
    // by a user-defined weighting function. The weighting function takes
    // (un)perturbed wave function parameters and Lagrangian multipliers as
    // input.
    //
    // `avail_perturbations` contains perturbations that can either be
    // extensive or intensive. When it is empty, optimal response function(s)
    // will be searched with fixed extensive and intensive perturbations.
    // Otherwise, all possible extensive and intensive perturbations, and
    // `min_wfn_exten` will be considered.
    //
    // `exten_perturbations` and `inten_perturbations` contain, respectively,
    // extensive and intensive perturbations. The former must contain at least
    // one extensive perturbation when `avail_perturbations` is empty. The
    // latter can be empty in any case.
    //
    // `excluded_operators` contains operators that should be excluded from
    // response functions. For example, a perturbed operator can or should be
    // removed if users are not able to evaluate it afterwards.
    //    fn find_optimal_response_function<F>(
    //        &self,
    //        avail_perturbations: &[Arc<Perturbation>],
    //        exten_perturbations: &[Arc<Perturbation>],
    //        inten_perturbations: &[Arc<Perturbation>],
    //        excluded_operators: Vec<Arc<dyn Expr>>,
    //        weight_fn: F,
    //    ) -> Result<Option<(f64, Vec<ResponseFunction>)>, TinnedError>
    //    where
    //        F: Fn(
    //            &BTreeMap<u32, BTreeSet<Arc<dyn Expr>>>,
    //            &BTreeMap<u32, BTreeSet<Arc<dyn Expr>>>,
    //        ) -> f64,
    //    {
    //        // Track best weight and best candidates
    //        let mut best_weight: Option<f64> = None;
    //        let mut best_responses: Vec<ResponseFunction> = Vec::new();
    //
    //        // You might loop over min_wfn_exten values or perturbation combinations
    //        for min_wfn_exten in 0..=some_max_value {
    //            // 1. Try to build the response function (may fail)
    //            let response_expr = self.response_function(
    //                exten_perturbations,
    //                inten_perturbations,
    //                min_wfn_exten,
    //                &excluded_operators,
    //            )?;
    //
    //            // (Optional) Skip if response_expr is "zero" or invalid
    //            if is_zero_expr(&response_expr) {
    //                continue;
    //            }
    //
    //            // 2. Compute weight
    //            let weight = weight_fn(&response_expr);
    //
    //            // 3. Compare with best so far
    //            match best_weight {
    //                None => {
    //                    best_weight = Some(weight);
    //                    best_responses.clear();
    //                    best_responses.push(ResponseFunction {
    //                        expression: response_expr,
    //                        min_wfn_exten,
    //                        exten_perturbations: exten_perturbations.clone(),
    //                        inten_perturbations: inten_perturbations.clone(),
    //                    });
    //                }
    //                Some(best) if weight < best => {
    //                    best_weight = Some(weight);
    //                    best_responses.clear();
    //                    best_responses.push(ResponseFunction {
    //                        expression: response_expr,
    //                        min_wfn_exten,
    //                        exten_perturbations: exten_perturbations.clone(),
    //                        inten_perturbations: inten_perturbations.clone(),
    //                    });
    //                }
    //                Some(best) if (weight - best).abs() <= 1e-12 => {
    //                    // Tie: add additional optimal response
    //                    best_responses.push(ResponseFunction {
    //                        expression: response_expr,
    //                        min_wfn_exten,
    //                        exten_perturbations: exten_perturbations.clone(),
    //                        inten_perturbations: inten_perturbations.clone(),
    //                    });
    //                }
    //                _ => {
    //                    // Weight worse: do nothing
    //                }
    //            }
    //        }
    //
    //        // Final result
    //        match best_weight {
    //            Some(weight) => Ok(Some((weight, best_responses))),
    //            None => Ok(None),
    //        }
    //    }

    // Get the time-averaged quasi-energy (derivative) Lagrangian
    fn get_lagrangian(&self) -> &Arc<dyn Expr>;
}
