use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use rayon::prelude::*;

use tinned::{Expr, NumberTolerance, Perturbation, TinnedError, generic_error};

use crate::lagrangian_internal::sealed::LagrangianInternal;
use crate::types::ResponseDetail;

// Base Lagrangian trait
pub trait Lagrangian: std::fmt::Debug + Send + Sync + LagrangianInternal {
    // Exposes the trait object as `dyn Any` to enable runtime downcasting of
    // trait objects to concrete types.
    fn as_any(&self) -> &dyn std::any::Any;

    // Return response function according to given extensive and intensive
    // perturbations, and minimum order of differentiated wave function
    // parameters to be eliminated, with respect to extensive perturbations.
    //
    // `exten_perturbations` and `inten_perturbations` contain, respectively,
    // extensive and intensive perturbations. The former must contain at least
    // one extensive perturbation, while the latter can be empty.
    //
    // `min_wfn_exten_order` as 0, means it will be automatically determined as the
    // next integer of the floor function of the half number of extensive
    // perturbations. For `min_wfn_exten_order` greater than the number of extensive
    // perturbations, it means no elimination of wave function parameters so
    // that more Lagrangian multipliers can be eliminated.
    fn response_function(
        &self,
        exten_perturbations: &[Arc<Perturbation>],
        inten_perturbations: &[Arc<Perturbation>],
        min_wfn_exten_order: u32,
        validate_frequencies: bool,
        num_tol: Option<NumberTolerance>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        let diff_lagrangian = self.differentiate_lagrangian(
            self.get_lagrangian(),
            exten_perturbations,
            inten_perturbations,
            validate_frequencies,
            num_tol.clone(),
        )?;

        self.do_elimination(&diff_lagrangian, exten_perturbations, min_wfn_exten_order, num_tol)
    }

    // Returns residue according to given extensive and intensive
    // perturbations, and minimum order of differentiated wave function
    // parameters to be eliminated, with respect to extensive perturbations.
    //
    // `residue_relations` contains excited state as the key `Arc<dyn Expr>`,
    // and the value informs in which direction perturbations approach the
    // excitation energy.
    #[inline]
    fn residue(
        &self,
        exten_perturbations: &[Arc<Perturbation>],
        inten_perturbations: &[Arc<Perturbation>],
        min_wfn_exten_order: u32,
        residue_relations: &HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)>,
        validate_frequencies: bool,
        num_tol: Option<NumberTolerance>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        let residue_setup = self.prepare_residue_analysis(
            exten_perturbations,
            inten_perturbations,
            &self.get_wfn_parameters(),
            &self.get_lagrangian_multipliers(),
            residue_relations,
        )?;

        //FIXME: For the test dao_3p_tme, n+1 rule gives 0 response function, how to explain? Is it valid?
        // Computes response function.
        let result = self.response_function(
            exten_perturbations,
            inten_perturbations,
            min_wfn_exten_order,
            validate_frequencies,
            num_tol,
        )?;

        self.do_residue_analysis(&result, &residue_setup)
    }

    // Returns optimal response function(s) by performing different elimination
    // rules. Optimal response function(s) has a minimal weight as determined
    // by a user-defined weighting function. The weighting function takes
    // (un)perturbed wave function parameters and Lagrangian multipliers as
    // input.
    //
    // Otherwise, all possible extensive and intensive perturbations, and
    // `min_wfn_exten_order` will be considered.
    //
    // `exten_perturbations` and `inten_perturbations` contain, respectively,
    // extensive and intensive perturbations. The former must contain at least
    // one extensive perturbation.
    //
    // `excluded_operators` contains operators that should be excluded from
    // response functions. For example, a perturbed operator can or should be
    // removed if users are not able to evaluate it afterwards.
    //
    // When a non-empty `residue_relations` is given, optimal residues will be found.
    //
    // Optimal response function(s) will be searched by varying the order of
    // differentiated wave function parameters to be eliminated with respect to
    // extensive perturbations
    fn find_optimal_elimination_order(
        &self,
        exten_perturbations: &[Arc<Perturbation>],
        inten_perturbations: &[Arc<Perturbation>],
        num_tol: Option<NumberTolerance>,
        excluded_operators: &HashSet<Arc<dyn Expr>>,
        weight_fn: &(
             dyn Fn(
            &BTreeMap<u32, HashSet<Arc<dyn Expr>>>,
            &BTreeMap<u32, HashSet<Arc<dyn Expr>>>,
        ) -> i64
                 + Send
                 + Sync
         ),
        residue_relations: Option<&HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)>>,
        validate_frequencies: bool,
        parallel: bool,
    ) -> Result<Option<(i64, Vec<ResponseDetail>)>, TinnedError> {
        let residue_setup = if let Some(res_relations) =
            residue_relations.filter(|relations| !relations.is_empty())
        {
            Some(self.prepare_residue_analysis(
                exten_perturbations,
                inten_perturbations,
                &self.get_wfn_parameters(),
                &self.get_lagrangian_multipliers(),
                res_relations,
            )?)
        } else {
            None
        };

        self.find_optimal_elimination_order_impl(
            self.get_lagrangian(),
            exten_perturbations,
            inten_perturbations,
            &self.get_wfn_parameters(),
            &self.get_lagrangian_multipliers(),
            num_tol.clone(),
            excluded_operators,
            weight_fn,
            residue_setup,
            validate_frequencies,
            parallel,
        )
    }

    // Returns optimal response function(s) by performing different elimination
    // rules. Optimal response function(s) has a minimal weight as determined
    // by a user-defined weighting function. The weighting function takes
    // (un)perturbed wave function parameters and Lagrangian multipliers as
    // input.
    //
    // `avail_perturbations` contains perturbations that can either be
    // extensive or intensive. When it is empty, optimal response function(s)
    // will be searched with fixed extensive and intensive perturbations.
    // Otherwise, all possible extensive and intensive perturbations, and
    // `min_wfn_exten_order` will be considered.
    //
    // `exten_perturbations` and `inten_perturbations` contain, respectively,
    // extensive and intensive perturbations. The former must contain at least
    // one extensive perturbation when `avail_perturbations` is empty. The
    // latter can be empty in any case.
    //
    // `excluded_operators` contains operators that should be excluded from
    // response functions. For example, a perturbed operator can or should be
    // removed if users are not able to evaluate it afterwards.
    //
    // When a non-empty `residue_relations` is given, optimal residues will be found.
    fn find_optimal_response_function(
        &self,
        avail_perturbations: &[Arc<Perturbation>],
        exten_perturbations: &[Arc<Perturbation>],
        inten_perturbations: &[Arc<Perturbation>],
        num_tol: Option<NumberTolerance>,
        excluded_operators: &HashSet<Arc<dyn Expr>>,
        weight_fn: &(
             dyn Fn(
            &BTreeMap<u32, HashSet<Arc<dyn Expr>>>,
            &BTreeMap<u32, HashSet<Arc<dyn Expr>>>,
        ) -> i64
                 + Send
                 + Sync
         ),
        residue_relations: Option<&HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)>>,
        validate_frequencies: bool,
    ) -> Result<Option<(i64, Vec<ResponseDetail>)>, TinnedError> {
        if avail_perturbations.is_empty() {
            return self.find_optimal_elimination_order(
                exten_perturbations,
                inten_perturbations,
                num_tol.clone(),
                excluded_operators,
                weight_fn,
                residue_relations,
                validate_frequencies,
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

        if num_perturbations >= usize::BITS as usize {
            return Err(generic_error(
                format!(
                    "Too many unique perturbations {} in avail_perturbations",
                    num_perturbations
                ),
                None,
            ));
        }

        let keys: Vec<_> = avail_map.keys().cloned().collect();

        // Make sure there is at least one extensive perturbation, so the first
        // subchain 0...0 is discarded for empty extensive perturbations
        let range_perturbations = if exten_perturbations.is_empty() {
            1..(1usize << num_perturbations)
        } else {
            0..(1usize << num_perturbations)
        };

        // We can reuse `residue_setup`
        let residue_setup = if let Some(res_relations) =
            residue_relations.filter(|relations| !relations.is_empty())
        {
            Some(self.prepare_residue_analysis(
                exten_perturbations,
                inten_perturbations,
                &self.get_wfn_parameters(),
                &self.get_lagrangian_multipliers(),
                res_relations,
            )?)
        } else {
            None
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
                    let target = if (mask & (1usize << i)) != 0 {
                        &mut new_exten
                    } else {
                        &mut new_inten
                    };
                    for _ in 0..count {
                        target.push(pert.clone());
                    }
                }

                self.find_optimal_elimination_order_impl(
                    self.get_lagrangian(),
                    &new_exten,
                    &new_inten,
                    &self.get_wfn_parameters(),
                    &self.get_lagrangian_multipliers(),
                    num_tol.clone(),
                    excluded_operators,
                    weight_fn,
                    residue_setup.clone(),
                    validate_frequencies,
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
    fn get_wfn_parameters(&self) -> Vec<Arc<dyn Expr>>;

    // Returns unperturbed Lagrangian multipliers
    fn get_lagrangian_multipliers(&self) -> Vec<Arc<dyn Expr>>;
}
