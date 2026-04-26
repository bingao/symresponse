use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use rayon::prelude::*;

use tinned::{Expr, NumberTolerance, Perturbation, TinnedError, generic_error};

use crate::lagrangian_internal::sealed::LagrangianInternal;
use crate::types::ResponseDetail;

/// Base trait for all Lagrangian implementations.
///
/// A `Lagrangian` represents a time-averaged quasi-energy functional and
/// provides methods to compute response functions, residues, and optimal
/// elimination strategies within response theory.
///
/// Implementations must provide access to:
/// - The symbolic Lagrangian expression
/// - Wave function parameters
/// - Lagrangian multipliers
///
/// This trait also provides default implementations for:
/// - Response function computation
/// - Residue computation
/// - Optimization of elimination strategies
pub trait Lagrangian: std::fmt::Debug + Send + Sync + LagrangianInternal {
    /// Returns a reference to `self` as `dyn Any` for downcasting.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Computes the response function for the given perturbations.
    ///
    /// # Arguments
    ///
    /// * `exten_perturbations` - Extensive perturbations (must contain at least one element)
    /// * `inten_perturbations` - Intensive perturbations (may be empty)
    /// * `min_wfn_exten_order` - Minimum order of differentiated wave function
    ///   parameters to eliminate with respect to extensive perturbations. For
    ///   `n_exten` extensive perturbations,
    ///     - `0`: automatically determined as `floor(n_exten / 2) + 1`
    ///     - `> n_exten`: disables elimination of wave function parameters
    /// * `validate_frequencies` - Whether to validate perturbation frequencies
    /// * `num_tol` - Optional numerical tolerance used to determine whether a
    ///   `tinned::Number` should be treated as zero.
    ///
    /// # Returns
    ///
    /// A symbolic expression representing the response function.
    ///
    /// # Errors
    ///
    /// Returns an error if differentiation or elimination fails.
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

    /// Computes the residue associated with the response function.
    ///
    /// # Arguments
    ///
    /// * `residue_relations` - Mapping from excited states to:
    ///     - direction of approach (bool)
    ///     - associated perturbations
    ///
    /// Other arguments are identical to [`Lagrangian::response_function`].
    ///
    /// # Returns
    ///
    /// A symbolic expression representing the residue.
    ///
    /// # Errors
    ///
    /// Returns an error if preparation or evaluation fails.
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

    /// Finds optimal response functions or residues by varying elimination rules.
    ///
    /// This method searches for optimal results by varying the minimum order of
    /// differentiated wave function parameters that are eliminated with respect
    /// to **extensive perturbations**.
    ///
    /// In contrast to [`Lagrangian::find_optimal_response_function`], this
    /// method assumes that the sets of extensive and intensive perturbations
    /// are fixed, and only explores different elimination strategies.
    ///
    /// The optimality is determined by a user-provided weighting function.
    ///
    /// # Arguments
    ///
    /// * `excluded_operators` - Operators that must not appear in the final result
    /// * `weight_fn` - Function that assigns a weight based on:
    ///     - (un)differentiated wave function parameters
    ///     - (un)differentiated Lagrangian multipliers
    /// * `residue_relations` - If provided, computes optimal residues instead of
    ///   response functions
    /// * `parallel` - Whether to evaluate candidates in parallel
    ///
    /// Other arguments are identical to [`Lagrangian::response_function`].
    ///
    /// # Returns
    ///
    /// Returns:
    ///
    /// - `None` if no valid solution is found
    /// - `Some((weight, results))` otherwise
    ///
    /// where:
    ///
    /// - `weight` is the minimal value obtained from `weight_fn`
    /// - `results` is a collection of optimal solutions represented as
    ///   [`ResponseDetail`]
    ///
    /// Each [`ResponseDetail`] contains:
    /// - the resulting response function or residue
    /// - the corresponding elimination rules used to obtain it
    ///
    /// # Errors
    ///
    /// Returns an error if evaluation or elimination fails.
    //FIXME: add unit test for this method
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

    /// Finds optimal response functions or residues by exploring perturbation partitions.
    ///
    /// This method extends [`Lagrangian::find_optimal_elimination_order`] by
    /// additionally exploring different ways to partition perturbations into
    /// **extensive** and **intensive** sets.
    ///
    /// - If `avail_perturbations` is empty, this method reduces to
    ///   [`Lagrangian::find_optimal_elimination_order`].
    /// - Otherwise, all valid partitions of perturbations are considered.
    ///
    /// The optimality is determined by the same weighting function.
    ///
    /// # Returns
    ///
    /// Returns:
    ///
    /// - `None` if no valid solution is found
    /// - `Some((weight, results))` otherwise
    ///
    /// where:
    ///
    /// - `weight` is the minimal value obtained from `weight_fn`
    /// - `results` is a collection of optimal solutions represented as
    ///   [`ResponseDetail`]
    ///
    /// See [`ResponseDetail`] for details on the structure of each result.
    //FIXME: add unit test for this method
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

    /// Returns the symbolic expression of time-averaged quasi-energy (derivative) Lagrangian.
    fn get_lagrangian(&self) -> &Arc<dyn Expr>;

    /// Returns unperturbed wave function parameters.
    fn get_wfn_parameters(&self) -> Vec<Arc<dyn Expr>>;

    /// Returns unperturbed Lagrangian multipliers.
    fn get_lagrangian_multipliers(&self) -> Vec<Arc<dyn Expr>>;
}
