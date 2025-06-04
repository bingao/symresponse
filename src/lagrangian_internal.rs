pub(crate) mod sealed {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use tinned::{
        Add, Expr, NumberTolerance, PertMultichain, Perturbation, ResidueParameter, TinnedError,
        differentiate_expr, generic_error, is_zero_expr, sum_pert_frequencies,
    };

    pub trait LagrangianInternal {
        // Checks if the sum of perturbations' frequencies is non zero
        #[inline]
        fn is_non_zero_sum_freqs(
            &self,
            exten_perturbations: &[Arc<Perturbation>],
            inten_perturbations: &[Arc<Perturbation>],
            num_tol: Option<NumberTolerance>,
        ) -> Result<bool, TinnedError> {
            let freq_sum_ext = sum_pert_frequencies(&exten_perturbations).map_err(|e| {
                generic_error(
                    "Sum of extensive perturbations' frequencies failed",
                    Some(Box::new(e)),
                )
            })?;
            let freq_sum_int = sum_pert_frequencies(&inten_perturbations).map_err(|e| {
                generic_error(
                    "Sum of intensive perturbations' frequencies failed",
                    Some(Box::new(e)),
                )
            })?;
            let total_freq = Add::new(vec![freq_sum_ext, freq_sum_int]).map_err(|e| {
                generic_error(
                    "Sum of all perturbations' frequencies failed",
                    Some(Box::new(e)),
                )
            })?;

            Ok(!is_zero_expr(&total_freq, num_tol))
        }

        // Checks if extensive and intensive perturbations have any common one(s)
        #[inline]
        fn has_common_perturbation(
            &self,
            exten_perturbations: &[Arc<Perturbation>],
            inten_perturbations: &[Arc<Perturbation>],
        ) -> bool {
            let exten_set: HashSet<_> = exten_perturbations.iter().collect();
            inten_perturbations.iter().any(|p| exten_set.contains(p))
        }

        // For each element of `residue_info`, its perturbations
        // (`Vec<Arc<Perturbation>>`) must be a proper subchain of the union of
        // extensive and intensive perturbations.
        //
        // Should perturbations (`Vec<Arc<Perturbation>>`) of `residue_info` be
        // unique? I am not sure, so I skip such a check.
        //
        // Should all perturbations from values of `residue_info` must be a proper
        // subchain of the union of extensive and intensive perturbations? I do not
        // find a proof for now, so I skip this check as well.
        #[inline]
        fn validate_residue_info(
            &self,
            exten_perturbations: &[Arc<Perturbation>],
            inten_perturbations: &[Arc<Perturbation>],
            residue_info: &HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)>,
        ) -> bool {
            let mut all_perturbations: Vec<Arc<Perturbation>> = exten_perturbations.to_vec();
            all_perturbations.extend(inten_perturbations.to_vec());
            let all_pert_chain = PertMultichain::from_slice(&all_perturbations);

            for (_excited_state, (_positive_frequency, perturbations)) in residue_info {
                if !all_pert_chain.is_subchain_vec(perturbations) {
                    return false;
                }
            }

            true
        }

        // For each parameter, compute its differentiated one with respect to
        // perturbations given in `residue_info`. A set `residue_set`
        // containing all these differentiated parameters returns, which can be
        // used to retain differentiated parameters and their higher-order ones
        // in an expression.
        //
        // For each differentiated parameter, the corresponding residue
        // parameter is also computed. A map `residue_map` returns, which
        // contains each differentiated parameter and its residue parameter as
        // key and value. The map can be used to replace differentiated
        // parameters by their residue ones.
        fn build_residue_parameters(
            &self,
            parameters: &[Arc<dyn Expr>],
            residue_info: &HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)>,
        ) -> Result<
            (
                HashSet<Arc<dyn Expr>>,
                HashMap<Arc<dyn Expr>, Arc<dyn Expr>>,
            ),
            TinnedError,
        > {
            let mut residue_set: HashSet<Arc<dyn Expr>> =
                HashSet::with_capacity(parameters.len() * residue_info.len());
            let mut residue_map: HashMap<Arc<dyn Expr>, Arc<dyn Expr>> =
                HashMap::with_capacity(residue_set.capacity());

            for param in parameters {
                for (excited_state, (positive_frequency, perturbations)) in residue_info {
                    let diff_param = differentiate_expr(param, perturbations)?;
                    let residue_param = ResidueParameter::builder(
                        perturbations.clone(),
                        excited_state.clone(),
                        diff_param.clone(),
                    )
                    .positive_frequency(*positive_frequency)
                    .build()?;
                    residue_set.insert(diff_param.clone());
                    residue_map.insert(diff_param, residue_param);
                }
            }

            Ok((residue_set, residue_map))
        }

        // Eliminate differentiated wave function parameter with respect to
        // extensive perturbations `exten_perturbations` from the derivative of
        // quasi-energy Lagrangian. Orders of differentiated wave function
        // parameter to be eliminated are from the minimum one `min_wfn_order` to
        // the maximum one as the size of `exten_perturbations`.
        fn eliminate_wfn_parameter(
            &self,
            lagrangian: &Arc<dyn Expr>,
            exten_perturbations: &[Arc<Perturbation>],
            min_wfn_order: u32,
        ) -> Result<Arc<dyn Expr>, TinnedError>;

        // Eliminate differentiated Lagrangian multipliers with respect to
        // extensive perturbations `exten_perturbations` from the derivative of
        // quasi-energy Lagrangian. Orders of differentiated Lagrangian multipliers
        // to be eliminated are from the minimum one `min_multiplier_order` to the
        // maximum one as the size of `exten_perturbations`.
        fn eliminate_lag_multipliers(
            &self,
            lagrangian: &Arc<dyn Expr>,
            exten_perturbations: &[Arc<Perturbation>],
            min_multiplier_order: u32,
        ) -> Result<Arc<dyn Expr>, TinnedError>;

        // Evaluation at zero perturbation strength
        #[inline]
        fn at_zero_strength(
            &self,
            lagrangian: &Arc<dyn Expr>,
            num_tol: Option<NumberTolerance>,
        ) -> Result<Arc<dyn Expr>, TinnedError> {
            // Remove unperturbed time-differentiated quantities (and unperturbed T
            // matrix for the AO density matrix-based response theory) as well as
            // their perturbed ones but with zero sum of perturbation frequencies.
            // Replace those with non-zero sum of frequencies by corresponding
            // derivatives in the frequency domain multiplied by the sum of
            // frequencies.
            lagrangian.clean_temporum(num_tol)
        }
    }
}
