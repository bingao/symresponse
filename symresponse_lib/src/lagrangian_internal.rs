pub(crate) mod sealed {
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::sync::Arc;

    use rayon::prelude::*;

    use tinned::{
        Add, Expr, NumberTolerance, PertMultichain, PertSequence, Perturbation, ResidueParameter,
        TinnedError, differentiate_expr, expression_error, generic_error, is_zero_expr,
        sum_pert_frequencies,
    };

    use crate::types::{EliminationScheme, LinearRhsInput, ResidueSetup, ResponseDetail};

    pub trait LagrangianInternal {
        // For AO density-matrix based response theory, we need the perturbation a
        #[inline]
        fn get_extra_perturbations(&self) -> Vec<Arc<Perturbation>> {
            Vec::new()
        }

        // Checks if the sum of perturbations' frequencies is non zero
        #[inline]
        fn is_non_zero_sum_freqs(
            &self,
            exten_perturbations: &[Arc<Perturbation>],
            inten_perturbations: &[Arc<Perturbation>],
            num_tol: Option<NumberTolerance>,
        ) -> Result<(), TinnedError> {
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

            let extra_perturbations = self.get_extra_perturbations();

            let mut terms = Vec::with_capacity(3);
            terms.push(freq_sum_ext);
            terms.push(freq_sum_int);

            if !extra_perturbations.is_empty() {
                let freq_sum_extra = sum_pert_frequencies(&extra_perturbations).map_err(|e| {
                    generic_error(
                        "Sum of extra perturbations' frequencies failed",
                        Some(Box::new(e)),
                    )
                })?;
                terms.push(freq_sum_extra);
            }

            let total_freq = Add::new(terms).map_err(|e| {
                generic_error("Sum of all perturbations' frequencies failed", Some(Box::new(e)))
            })?;

            if is_zero_expr(&total_freq, num_tol) {
                Ok(())
            } else {
                Err(generic_error(
                    format!(
                        "Lagrangian gets perturbations with non-zero sum frequencies {}",
                        total_freq
                    ),
                    None,
                ))
            }
        }

        // Checks if extensive and intensive perturbations have any common one(s)
        #[inline]
        fn has_common_perturbation(
            &self,
            exten_perturbations: &[Arc<Perturbation>],
            inten_perturbations: &[Arc<Perturbation>],
        ) -> Result<(), TinnedError> {
            let exten_set: HashSet<_> = exten_perturbations.iter().collect();
            if let Some(common) = inten_perturbations.iter().find(|p| exten_set.contains(p)) {
                return Err(generic_error(
                    format!("Lagrangian gets same extensive and intensive perturbation {}", common),
                    None,
                ));
            }

            Ok(())
        }

        // Validates given extensive and intensive perturbations
        fn validate_perturbations(
            &self,
            exten_perturbations: &[Arc<Perturbation>],
            inten_perturbations: &[Arc<Perturbation>],
            validate_frequencies: bool,
            num_tol: Option<NumberTolerance>,
        ) -> Result<(), TinnedError> {
            // Validate that: (i) at least one extensive perturbation, (ii) sum of
            // all perturbations' frequencies is zero, and (iii) extensive and
            // intensive perturbations are disjoint
            if exten_perturbations.is_empty() {
                return Err(generic_error("At least one extensive perturbation is required", None));
            }

            if validate_frequencies {
                self.is_non_zero_sum_freqs(exten_perturbations, inten_perturbations, num_tol)?;
            }

            self.has_common_perturbation(exten_perturbations, inten_perturbations)
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

        // Return differentiated Lagrangian with respect to given extensive and
        // intensive perturbations.
        //
        // `exten_perturbations` and `inten_perturbations` contain, respectively,
        // extensive and intensive perturbations. The former must contain at least
        // one extensive perturbation, while the latter can be empty.
        fn differentiate_lagrangian(
            &self,
            lagrangian: &Arc<dyn Expr>,
            exten_perturbations: &[Arc<Perturbation>],
            inten_perturbations: &[Arc<Perturbation>],
            validate_frequencies: bool,
            num_tol: Option<NumberTolerance>,
        ) -> Result<Arc<dyn Expr>, TinnedError> {
            // Validates perturbations
            self.validate_perturbations(
                exten_perturbations,
                inten_perturbations,
                validate_frequencies,
                num_tol,
            )?;

            // Differentiates the quasi-energy (derivative) Lagrangian
            self.do_differentiation(lagrangian, exten_perturbations, inten_perturbations)
        }

        // Eliminates differentiated wave function parameter with respect to
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

        // Eliminates differentiated Lagrangian multipliers with respect to
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

        // Returns response function from the given differentiated `lagrangian`
        // by eliminating differentiated wave function parameters and/or
        // Lagrangian multipliers, and evaluating at zero perturbation strength.
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
        fn do_elimination(
            &self,
            lagrangian: &Arc<dyn Expr>,
            exten_perturbations: &[Arc<Perturbation>],
            min_wfn_exten_order: u32,
            num_tol: Option<NumberTolerance>,
        ) -> Result<Arc<dyn Expr>, TinnedError> {
            let elimination_scheme =
                EliminationScheme::new(exten_perturbations.len() as u32, min_wfn_exten_order)?;

            // Eliminate wave function parameter
            let mut result = if elimination_scheme.eliminate_wfn() {
                self.eliminate_wfn_parameter(
                    lagrangian,
                    exten_perturbations,
                    elimination_scheme.min_wfn_order(),
                )
                .map_err(|e| {
                    generic_error(
                        "Elimination of wave function parameter failed",
                        Some(Box::new(e)),
                    )
                })?
            } else {
                lagrangian.clone()
            };

            // Eliminate Lagrangian multipliers
            result = self
                .eliminate_lag_multipliers(
                    &result,
                    exten_perturbations,
                    elimination_scheme.min_multiplier_order(),
                )
                .map_err(|e| {
                    generic_error("Elimination of Lagrangian multipliers failed", Some(Box::new(e)))
                })?;

            // Usually `result` cannot be zero after elimination
            if is_zero_expr(&result, num_tol.clone()) {
                Ok(result)
            } else {
                // Evaluation at zero perturbation strength
                result.substitute_zero_perturbations(num_tol)
            }
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
        ) -> Result<(), TinnedError> {
            if residue_relations.is_empty() {
                return Err(generic_error("Empty residue relations encountered", None));
            }

            let extra_perturbations = self.get_extra_perturbations();
            let mut all_perturbations = Vec::with_capacity(
                exten_perturbations.len() + inten_perturbations.len() + extra_perturbations.len(),
            );

            all_perturbations.extend_from_slice(exten_perturbations);
            all_perturbations.extend_from_slice(inten_perturbations);
            all_perturbations.extend(extra_perturbations);

            let all_pert_chain = PertMultichain::from_slice(&all_perturbations);

            for (_excited_state, (_positive_frequency, perturbations)) in residue_relations {
                if !all_pert_chain.is_subchain_vec(perturbations) {
                    return Err(generic_error(
                        format!(
                            "Residue relations have perturbation(s) {} not existing in given extensive and intensive perturbations",
                            perturbations as &dyn PertSequence
                        ),
                        None,
                    ));
                }
            }

            Ok(())
        }

        // For the given first order residue information (excited state,
        // frequency direction of approach and perturbations), and an
        // unperturbed `parameter`, returns differentiated parameter and its
        // residue with respect to the `perturbations`.
        //
        // The differentiated parameter can be used to retain the
        // differentiated parameter itself and its higher-order ones in an
        // expression.
        //
        // The residue parameter can be used as a replacement of the
        // differentiated parameter.
        fn build_first_order_residue_parameter(
            &self,
            parameter: &Arc<dyn Expr>,
            excited_state: &Arc<dyn Expr>,
            positive_frequency: bool,
            perturbations: &[Arc<Perturbation>],
        ) -> Result<(Arc<dyn Expr>, Arc<dyn Expr>), TinnedError> {
            let diff_parameter = differentiate_expr(parameter, &perturbations)?;
            let residue_parameter = ResidueParameter::builder(
                perturbations.to_vec(),
                excited_state.clone(),
                diff_parameter.clone(),
            )
            .positive_frequency(positive_frequency)
            .build()?;

            Ok((diff_parameter, residue_parameter))
        }

        // Returns differentiated response parameters with respect to
        // perturbations involved in residue computation specified by
        // `residue_relations`, and a map between each differentiated response
        // parameter and its residue.
        //
        // The set of these differentiated response parameters can be used to
        // retain differentiated parameters and their higher-order ones in an
        // expression.
        //
        // The map between differentiated response parameters and their
        // residues can be used to replace differentiated parameters by their
        // residues.
        fn build_residue_parameters(
            &self,
            parameters: &[Arc<dyn Expr>],
            residue_relations: &HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)>,
        ) -> Result<ResidueSetup, TinnedError> {
            let mut all_diff_parameters: Vec<HashSet<Arc<dyn Expr>>> =
                Vec::with_capacity(residue_relations.len());
            let mut residue_map: HashMap<Arc<dyn Expr>, Arc<dyn Expr>> =
                HashMap::with_capacity(parameters.len() * residue_relations.len());

            // Applies the retain operation to the expression for each residue relation in turn
            for (excited_state, (positive_frequency, perturbations)) in residue_relations {
                let mut diff_parameters: HashSet<Arc<dyn Expr>> =
                    HashSet::with_capacity(parameters.len());

                // Retains an expression if it contains any differentiated parameter,
                // not necessarily all differentiated parameters.
                for parameter in parameters {
                    let (diff_parameter, residue_parameter) = self
                        .build_first_order_residue_parameter(
                            parameter,
                            excited_state,
                            *positive_frequency,
                            perturbations,
                        )?;

                    if residue_map.contains_key(&diff_parameter) {
                        return Err(expression_error(
                            format!(
                                "Duplicate differentiated parameter detected {}",
                                diff_parameter
                            ),
                            parameter,
                            None,
                        ));
                    }

                    diff_parameters.insert(diff_parameter.clone());
                    residue_map.insert(diff_parameter, residue_parameter);
                }

                all_diff_parameters.push(diff_parameters);
            }

            Ok(ResidueSetup::new(all_diff_parameters, residue_map))
        }

        // Validates given extensive and intensive perturbations, and residue
        // relations. Returns differentiated wave function parameters and
        // Lagrangian multipliers (named as response parameter) with respect to
        // perturbations involved in residue computation, and a map between
        // each differentiated response parameter and its residue.
        #[inline]
        fn prepare_residue_analysis(
            &self,
            exten_perturbations: &[Arc<Perturbation>],
            inten_perturbations: &[Arc<Perturbation>],
            wfn_parameters: &[Arc<dyn Expr>],
            lagrangian_multipliers: &[Arc<dyn Expr>],
            residue_relations: &HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)>,
        ) -> Result<ResidueSetup, TinnedError> {
            self.validate_residue_relations(
                exten_perturbations,
                inten_perturbations,
                residue_relations,
            )?;

            // Sets up `diff_parameters` and `residue_map` for retaining and replacing
            // differentiated parameters and their higher-order ones.
            let mut parameters =
                Vec::with_capacity(wfn_parameters.len() + lagrangian_multipliers.len());

            parameters.extend_from_slice(wfn_parameters);
            parameters.extend_from_slice(lagrangian_multipliers);

            self.build_residue_parameters(&parameters, residue_relations)
        }

        // Returns the residue of a response function `rsp_function` according
        // to a given residue setup.
        //
        // `residue_setup` contains differentiated response parameter with
        // respect to perturbations involved in residue computation, and a map
        // between each differentiated response parameter and its residue.
        #[inline]
        fn do_residue_analysis(
            &self,
            rsp_function: &Arc<dyn Expr>,
            residue_setup: &ResidueSetup,
        ) -> Result<Arc<dyn Expr>, TinnedError> {
            // Retains differentiated parameters specified by `residue_relations`,
            // as well as their higher-order ones while removes other
            // (un)differentiated parameters.
            let result = rsp_function.retain_all(residue_setup.diff_parameters(), true)?;

            //FIXME: `result` may become zero after `retain()`, we could consider the "complementary" residue parameters

            // Replaces differentiated parameters by their residue parameters.
            result.replace_all(residue_setup.residue_map(), true)
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
        fn score_response_function(
            &self,
            rsp_function: Arc<dyn Expr>,
            wfn_parameters: &[Arc<dyn Expr>],
            lagrangian_multipliers: &[Arc<dyn Expr>],
            exten_perturbations: &[Arc<Perturbation>],
            inten_perturbations: &[Arc<Perturbation>],
            min_wfn_exten_order: u32,
            excluded_operators: &HashSet<Arc<dyn Expr>>,
            weight_fn: &(
                 dyn Fn(
                &BTreeMap<u32, HashSet<Arc<dyn Expr>>>,
                &BTreeMap<u32, HashSet<Arc<dyn Expr>>>,
            ) -> i64
                     + Send
                     + Sync
             ),
        ) -> Result<Option<(i64, ResponseDetail)>, TinnedError> {
            if rsp_function.match_any(excluded_operators, false) {
                return Ok(None);
            }

            let mut wfn_map = BTreeMap::<u32, HashSet<Arc<dyn Expr>>>::new();
            let mut lag_map = BTreeMap::<u32, HashSet<Arc<dyn Expr>>>::new();

            for param in wfn_parameters {
                for (order, found) in rsp_function.find_all(&param) {
                    wfn_map.entry(order).or_default().extend(found);
                }
            }

            for param in lagrangian_multipliers {
                for (order, found) in rsp_function.find_all(&param) {
                    lag_map.entry(order).or_default().extend(found);
                }
            }

            let weight = weight_fn(&wfn_map, &lag_map);
            let result = ResponseDetail {
                expression: rsp_function,
                min_wfn_exten_order,
                exten_perturbations: exten_perturbations.to_vec(),
                inten_perturbations: inten_perturbations.to_vec(),
            };

            Ok(Some((weight, result)))
        }

        // This is the implementation for the method
        // `find_optimal_elimination_order()` to find optimal response
        // function(s) by performing different elimination rules.
        //
        // When a non-empty `residue_relations` is given, optimal residues will
        // be found.
        //
        // This method is also called by `find_optimal_response_function()`
        // inside the iteration of extensive and intensive perturbations. It is
        // valid in our current policy because the methods
        // `prepare_residue_analysis()` and `validate_residue_relations()` are
        // only called once and residue setup can be used for different
        // extensive and intensive perturbations
        fn find_optimal_elimination_order_impl(
            &self,
            lagrangian: &Arc<dyn Expr>,
            exten_perturbations: &[Arc<Perturbation>],
            inten_perturbations: &[Arc<Perturbation>],
            wfn_parameters: &[Arc<dyn Expr>],
            lagrangian_multipliers: &[Arc<dyn Expr>],
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
            residue_setup: Option<ResidueSetup>,
            validate_frequencies: bool,
            parallel: bool,
        ) -> Result<Option<(i64, Vec<ResponseDetail>)>, TinnedError>
        where
            Self: Sync,
        {
            // Differentiates Lagrangian that can be reused later
            let diff_lagrangian = self.differentiate_lagrangian(
                lagrangian,
                exten_perturbations,
                inten_perturbations,
                validate_frequencies,
                num_tol.clone(),
            )?;

            let min_wfn_order: u32 = 1 + (exten_perturbations.len() / 2) as u32;
            let max_wfn_order: u32 = 1 + exten_perturbations.len() as u32;
            let range_orders = min_wfn_order..=max_wfn_order;

            let score_elimination_order =
                |order: u32| -> Result<Option<(i64, ResponseDetail)>, TinnedError> {
                    let mut result = self.do_elimination(
                        &diff_lagrangian,
                        exten_perturbations,
                        order,
                        num_tol.clone(),
                    )?;

                    if let Some(res_setup) = &residue_setup {
                        result = self.do_residue_analysis(&result, res_setup)?;
                    }

                    self.score_response_function(
                        result,
                        wfn_parameters,
                        lagrangian_multipliers,
                        exten_perturbations,
                        inten_perturbations,
                        order,
                        excluded_operators,
                        weight_fn,
                    )
                };

            // Iterates orders of differentiated wave function parameters with
            // respect to extensive perturbations to be eliminated
            let results: Vec<(i64, ResponseDetail)> = if parallel {
                range_orders
                    .into_par_iter()
                    .map(score_elimination_order)
                    .collect::<Result<Vec<_>, TinnedError>>()?
                    .into_iter()
                    .flatten() // drop None entries (due to excluded operators)
                    .collect()
            } else {
                range_orders
                    .into_iter()
                    .map(score_elimination_order)
                    .collect::<Result<Vec<_>, TinnedError>>()?
                    .into_iter()
                    .flatten()
                    .collect()
            };

            if results.is_empty() {
                return Ok(None);
            }

            // Finds all response functions with minimal weight
            let min_weight = results.iter().map(|(w, _)| *w).min().ok_or_else(|| {
                generic_error(
                    "Unexpected: non-empty results but failed to compute minimum weight",
                    None,
                )
            })?;

            let optimal = results
                .into_iter()
                .filter_map(|(w, r)| {
                    if w == min_weight {
                        Some(r)
                    } else {
                        None
                    }
                })
                .collect();

            Ok(Some((min_weight, optimal)))
        }

        // Builds the right-hand side of (linear) response equation
        fn build_linear_rhs(
            &self,
            rhs_input: LinearRhsInput<'_>,
            num_tol: Option<NumberTolerance>,
        ) -> Result<Arc<dyn Expr>, TinnedError> {
            let result = differentiate_expr(rhs_input.equation, rhs_input.derivative)?
                .substitute_zero_perturbations(num_tol)?
                .remove_one(rhs_input.diff_parameter)?;

            let Some((residue_parameter, unperturbed_parameter)) = rhs_input.residue_info else {
                return Ok(result);
            };

            let (diff_parameter, residue_parameter) = self.build_first_order_residue_parameter(
                &unperturbed_parameter,
                residue_parameter.excited_state(),
                residue_parameter.positive_frequency(),
                residue_parameter.perturbations(),
            )?;

            result.retain_one(&diff_parameter, true)?.replace_one(
                &diff_parameter,
                residue_parameter,
                true,
            )
        }
    }
}
