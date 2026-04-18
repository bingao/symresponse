pub(crate) mod sealed {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use tinned::{
        Add, Expr, NumberTolerance, PertMultichain, Perturbation, ResidueParameter, TinnedError,
        differentiate_expr, expression_error, generic_error, is_zero_expr, sum_pert_frequencies,
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
                generic_error("Sum of all perturbations' frequencies failed", Some(Box::new(e)))
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

        // Perform differentiation with respect to extensive and intensive perturbations
        #[inline]
        fn do_differentiation(
            &self,
            expr: &Arc<dyn Expr>,
            exten_perturbations: &[Arc<Perturbation>],
            inten_perturbations: &[Arc<Perturbation>],
        ) -> Result<Arc<dyn Expr>, TinnedError> {
            let result = differentiate_expr(expr, &exten_perturbations).map_err(|e| {
                expression_error(
                    "Differentiation with respect to extensive perturbations failed",
                    expr,
                    Some(Box::new(e)),
                )
            })?;
            differentiate_expr(&result, &inten_perturbations).map_err(|e| {
                expression_error(
                    "Differentiation with respect to intensive perturbations failed",
                    expr,
                    Some(Box::new(e)),
                )
            })
        }

        // Post operations after differentiating the Lagrangian
        #[inline]
        fn post_differentiation(
            &self,
            lagrangian: &Arc<dyn Expr>,
            _exten_perturbations: &[Arc<Perturbation>],
            _inten_perturbations: &[Arc<Perturbation>],
        ) -> Result<Arc<dyn Expr>, TinnedError> {
            Ok(lagrangian.clone())
        }

        // For each element of `residue_relations`, its perturbations
        // (`Vec<Arc<Perturbation>>`) must be a proper subchain of the union of
        // extensive and intensive perturbations.
        //
        // I am not sure if perturbations (`Vec<Arc<Perturbation>>`) of
        // `residue_relations` should be unique, so I skip such a check.
        //
        // I am not sure if all perturbations from values of
        // `residue_relations` must be a proper subchain of the union of
        // extensive and intensive perturbations. I do not find a proof for
        // now, so I skip this check as well.
        #[inline]
        fn validate_residue_relations(
            &self,
            exten_perturbations: &[Arc<Perturbation>],
            inten_perturbations: &[Arc<Perturbation>],
            residue_relations: &HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)>,
        ) -> bool {
            let mut all_perturbations: Vec<Arc<Perturbation>> = exten_perturbations.to_vec();
            all_perturbations.extend(inten_perturbations.to_vec());
            let all_pert_chain = PertMultichain::from_slice(&all_perturbations);

            for (_excited_state, (_positive_frequency, perturbations)) in residue_relations {
                if !all_pert_chain.is_subchain_vec(perturbations) {
                    return false;
                }
            }

            true
        }

        // For each parameter, compute its differentiated one with respect to
        // perturbations given in `residue_relations`. A set `diff_params`
        // returns with all these differentiated parameters, which can be used
        // to retain differentiated parameters and their higher-order ones in
        // an expression.
        //
        // For each differentiated parameter, the corresponding residue
        // parameter is also computed. A map `residue_map` returns, which
        // contains each differentiated parameter and its residue parameter as
        // key and value. The map can be used to replace differentiated
        // parameters by their residue ones.
        fn build_residue_parameters(
            &self,
            parameters: &[Arc<dyn Expr>],
            residue_relations: &HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)>,
        ) -> Result<(HashSet<Arc<dyn Expr>>, HashMap<Arc<dyn Expr>, Arc<dyn Expr>>), TinnedError>
        {
            let mut diff_params: HashSet<Arc<dyn Expr>> =
                HashSet::with_capacity(parameters.len() * residue_relations.len());
            let mut residue_map: HashMap<Arc<dyn Expr>, Arc<dyn Expr>> =
                HashMap::with_capacity(diff_params.capacity());

            for param in parameters {
                for (excited_state, (positive_frequency, perturbations)) in residue_relations {
                    let param_deriv = differentiate_expr(param, perturbations)?;
                    let residue_param = ResidueParameter::builder(
                        perturbations.clone(),
                        excited_state.clone(),
                        param_deriv.clone(),
                    )
                    .positive_frequency(*positive_frequency)
                    .build()?;

                    diff_params.insert(param_deriv.clone());
                    residue_map.insert(param_deriv, residue_param);
                }
            }

            Ok((diff_params, residue_map))
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

        // Finalizes the right-hand side of (linear) response equation
        fn finalize_response_rhs(
            &self,
            general_rhs: &Arc<dyn Expr>,
            diff_parameter: &Arc<dyn Expr>,
            residue_info: Option<(&ResidueParameter, Arc<dyn Expr>)>,
            num_tol: Option<NumberTolerance>,
        ) -> Result<Arc<dyn Expr>, TinnedError> {
            let params_to_remove = HashSet::from([diff_parameter.clone()]);

            let result =
                general_rhs.substitute_zero_perturbations(num_tol)?.remove(&params_to_remove)?;

            let Some((residue_parameter, unperturbed_parameter)) = residue_info else {
                return Ok(result);
            };

            let residue_relations: HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)> =
                HashMap::from([(
                    residue_parameter.excited_state().clone(),
                    (
                        residue_parameter.positive_frequency(),
                        residue_parameter.perturbations().to_vec(),
                    ),
                )]);

            let (diff_params, residue_map) =
                self.build_residue_parameters(&[unperturbed_parameter], &residue_relations)?;

            result.retain(&diff_params, true)?.replace(&residue_map, true)
        }
    }
}
