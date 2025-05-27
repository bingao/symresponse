use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use rayon::prelude::*;

use tinned::{
    differentiate_expr, generic_error, is_zero_expr, Expr, NumberTolerance, Perturbation,
    TinnedError,
};

use crate::lagrangian_internal::sealed::LagrangianInternal;
use crate::types::ResponseDetail;

// Base Lagrangian trait
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
        if is_zero_expr(&result, num_tol.clone()) {
            return Ok(result);
        }

        // Evaluation at zero perturbation strength
        self.at_zero_strength(&result, num_tol)
    }

    // Returns residue according to given extensive and intensive
    // perturbations, and minimum order of differentiated wave function
    // parameters to be eliminated, with respect to extensive perturbations.
    //
    // `residue_info` contains excited state as the key `Arc<dyn Expr>`, and
    // the value informs in which direction perturbations approach the
    // excitation energy.
    #[inline]
    fn residue(
        &self,
        exten_perturbations: &[Arc<Perturbation>],
        inten_perturbations: &[Arc<Perturbation>],
        min_wfn_exten: u32,
        residue_info: &HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)>,
        num_tol: Option<NumberTolerance>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        if !self.validate_residue_info(exten_perturbations, inten_perturbations, residue_info) {
            return Err(generic_error(
                "Invalid residue information for given (extensive and intensive) perturbations.",
                None,
            ));
        }

        // Computes response function.
        let mut result = self.response_function(
            exten_perturbations,
            inten_perturbations,
            min_wfn_exten,
            num_tol,
        )?;

        // Sets up `residue_set` and `residue_map` for retaining and replacing
        // differentiated parameters and their higher-order ones.
        let mut parameters: Vec<Arc<dyn Expr>> = self.get_wfn_parameter();
        parameters.extend(self.get_lag_multiplier());
        let (residue_set, residue_map) =
            self.build_residue_parameters(&parameters, residue_info)?;

        // Retains differentiated parameters specified by `residue_info`, as
        // well as their higher-order ones while removes other
        // (un)differentiated parameters.
        result = result.retain(&residue_set, false)?;

        // Replaces differentiated parameters by their residue parameters.
        result.replace(&residue_map, false)
    }

    // Returns response function with its weight.
    //
    // The weight is computed by a user-defined weighting function, which takes
    // (un)perturbed wave function parameters and Lagrangian multipliers as
    // input.
    //
    // `excluded_operators` contains operators that should be excluded from the
    // response function. For example, a perturbed operator can or should be
    // removed if users are not able to evaluate it afterwards.
    fn response_function_with_weight<F>(
        &self,
        exten_perturbations: &[Arc<Perturbation>],
        inten_perturbations: &[Arc<Perturbation>],
        min_wfn_exten: u32,
        num_tol: Option<NumberTolerance>,
        excluded_operators: &HashSet<Arc<dyn Expr>>,
        weight_fn: &F,
    ) -> Result<Option<(i64, ResponseDetail)>, TinnedError>
    where
        F: Sync
            + Send
            + Fn(
                &BTreeMap<u32, HashSet<Arc<dyn Expr>>>,
                &BTreeMap<u32, HashSet<Arc<dyn Expr>>>,
            ) -> i64,
    {
        let expr = self.response_function(
            exten_perturbations,
            inten_perturbations,
            min_wfn_exten,
            num_tol,
        )?;

        if expr.exist_any(excluded_operators) {
            return Ok(None);
        }

        let mut wfn_map = BTreeMap::<u32, HashSet<Arc<dyn Expr>>>::new();
        let mut lag_map = BTreeMap::<u32, HashSet<Arc<dyn Expr>>>::new();

        for param in self.get_wfn_parameter() {
            for (order, found) in expr.find_superchains(&param) {
                wfn_map.entry(order).or_default().extend(found);
            }
        }

        for param in self.get_lag_multiplier() {
            for (order, found) in expr.find_superchains(&param) {
                lag_map.entry(order).or_default().extend(found);
            }
        }

        let weight = weight_fn(&wfn_map, &lag_map);
        let rf = ResponseDetail {
            expression: expr,
            min_wfn_exten,
            exten_perturbations: exten_perturbations.to_vec(),
            inten_perturbations: inten_perturbations.to_vec(),
        };

        Ok(Some((weight, rf)))
    }

    // Return optimal response function(s) by performing different elimination
    // rules. Optimal response function(s) has a minimal weight as determined
    // by a user-defined weighting function. The weighting function takes
    // (un)perturbed wave function parameters and Lagrangian multipliers as
    // input.
    //
    // Otherwise, all possible extensive and intensive perturbations, and
    // `min_wfn_exten` will be considered.
    //
    // `exten_perturbations` and `inten_perturbations` contain, respectively,
    // extensive and intensive perturbations. The former must contain at least
    // one extensive perturbation.
    //
    // `excluded_operators` contains operators that should be excluded from
    // response functions. For example, a perturbed operator can or should be
    // removed if users are not able to evaluate it afterwards.
    //
    // Optimal response function(s) will be searched by varying the order of
    // differentiated wave function parameters to be eliminated with respect to
    // extensive perturbations
    fn find_optimal_elimination_order<F>(
        &self,
        exten_perturbations: &[Arc<Perturbation>],
        inten_perturbations: &[Arc<Perturbation>],
        num_tol: Option<NumberTolerance>,
        excluded_operators: &HashSet<Arc<dyn Expr>>,
        weight_fn: &F,
        parallel: bool,
    ) -> Result<Option<(i64, Vec<ResponseDetail>)>, TinnedError>
    where
        F: Sync
            + Send
            + Fn(
                &BTreeMap<u32, HashSet<Arc<dyn Expr>>>,
                &BTreeMap<u32, HashSet<Arc<dyn Expr>>>,
            ) -> i64,
    {
        let min_wfn_order: u32 = 1 + (exten_perturbations.len() / 2) as u32;
        let max_wfn_order: u32 = 1 + exten_perturbations.len() as u32;
        let range_orders = min_wfn_order..=max_wfn_order;

        // Iterates orders of differentiated wave function parameters with
        // respect to extensive perturbations to be eliminated
        let results: Vec<(i64, ResponseDetail)> = if parallel {
            range_orders
                .into_par_iter()
                .map(|order| {
                    self.response_function_with_weight(
                        exten_perturbations,
                        inten_perturbations,
                        order,
                        num_tol.clone(),
                        excluded_operators,
                        weight_fn,
                    )
                })
                .collect::<Result<Vec<_>, TinnedError>>()?
                .into_iter()
                .flatten() // drop None entries (due to excluded operators)
                .collect()
        } else {
            range_orders
                .into_iter()
                .map(|order| {
                    self.response_function_with_weight(
                        exten_perturbations,
                        inten_perturbations,
                        order,
                        num_tol.clone(),
                        excluded_operators,
                        weight_fn,
                    )
                })
                .collect::<Result<Vec<_>, TinnedError>>()?
                .into_iter()
                .flatten()
                .collect()
        };

        if results.is_empty() {
            return Ok(None);
        }

        // Find all response functions with minimal weight
        let min_weight = results.iter().map(|(w, _)| *w).min().ok_or_else(|| {
            generic_error(
                "Unexpected: non-empty results but failed to compute minimum weight",
                None,
            )
        })?;

        let optimal = results
            .into_iter()
            .filter_map(|(w, r)| if w == min_weight { Some(r) } else { None })
            .collect();

        Ok(Some((min_weight, optimal)))
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
    fn find_optimal_response_function<F>(
        &self,
        avail_perturbations: &[Arc<Perturbation>],
        exten_perturbations: &[Arc<Perturbation>],
        inten_perturbations: &[Arc<Perturbation>],
        num_tol: Option<NumberTolerance>,
        excluded_operators: &HashSet<Arc<dyn Expr>>,
        weight_fn: &F,
    ) -> Result<Option<(i64, Vec<ResponseDetail>)>, TinnedError>
    where
        F: Sync
            + Send
            + Fn(
                &BTreeMap<u32, HashSet<Arc<dyn Expr>>>,
                &BTreeMap<u32, HashSet<Arc<dyn Expr>>>,
            ) -> i64,
    {
        if avail_perturbations.is_empty() {
            return self.find_optimal_elimination_order(
                exten_perturbations,
                inten_perturbations,
                num_tol.clone(),
                excluded_operators,
                weight_fn,
                true,
            );
        }

        // Convert available perturbations into a map of perturbation and
        // number of occurrences (multiplicity)
        let mut avail_map = BTreeMap::new();
        for p in avail_perturbations {
            *avail_map.entry(p.clone()).or_insert(0) += 1;
        }

        let num_perturbations = avail_map.len();
        let keys: Vec<_> = avail_map.keys().cloned().collect();

        // Make sure there is at least one extensive perturbation, so the first
        // subchain 0...0 is discarded for empty extensive perturbations
        let range_perturbations = if exten_perturbations.is_empty() {
            1..(1 << num_perturbations)
        } else {
            0..(1 << num_perturbations)
        };

        // We generate all subsets of the unique perturbations by following the
        // direct approach in Chapter 1, "Next Subset of an n-Set
        // (NEXSUB/LEXSUB)", Combinatorial Algorithms For Computers and
        // Calculators (2nd Edition), Albert Nijenhuis and Herbert S. Wilf. New
        // extensive perturbations are from the subset while new intensive
        // perturbations are from the complement of the subset.
        let results: Vec<(i64, Vec<ResponseDetail>)> = range_perturbations
            .into_par_iter()
            .map(|mask| {
                let mut new_exten = exten_perturbations.to_vec();
                let mut new_inten = inten_perturbations.to_vec();

                for (i, pert) in keys.iter().enumerate() {
                    let count = *avail_map.get(pert).ok_or_else(|| {
                        generic_error("Unexpected: failed to get the key in avail_map", None)
                    })?;
                    let target = if (mask & (1 << i)) != 0 {
                        &mut new_exten
                    } else {
                        &mut new_inten
                    };
                    for _ in 0..count {
                        target.push(pert.clone());
                    }
                }

                self.find_optimal_elimination_order(
                    &new_exten,
                    &new_inten,
                    num_tol.clone(),
                    excluded_operators,
                    weight_fn,
                    false,
                )
            })
            .collect::<Result<Vec<_>, TinnedError>>()?
            .into_iter()
            .flatten()
            .collect();

        if results.is_empty() {
            Ok(None)
        } else {
            let min_weight = results.iter().map(|(w, _)| *w).min().ok_or_else(|| {
                generic_error(
                    "Unexpected: non-empty results but failed to compute minimum weight",
                    None,
                )
            })?;
            let optimal: Vec<_> = results
                .into_iter()
                .filter(|(w, _)| *w == min_weight)
                .flat_map(|(_, r)| r)
                .collect();

            Ok(Some((min_weight, optimal)))
        }
    }

    // Returns the time-averaged quasi-energy (derivative) Lagrangian
    fn get_lagrangian(&self) -> &Arc<dyn Expr>;

    // Returns unperturbed wave function parameters
    fn get_wfn_parameter(&self) -> Vec<Arc<dyn Expr>>;

    // Returns unperturbed Lagrangian multipliers
    fn get_lag_multiplier(&self) -> Vec<Arc<dyn Expr>>;
}
