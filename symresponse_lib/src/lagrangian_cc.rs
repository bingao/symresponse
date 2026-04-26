use std::sync::Arc;

use tinned::{
    AdjointMap, AdjointMode, DotProduct, ExcitationOperator, ExpAdjointMap, Expr, LagMultiplier,
    MatrixAdd, MatrixMul, Number, NumberTolerance, Perturbation, ResidueParameter, TimeEvolution,
    TinnedError, WfnParameter, downcast_from_arc, expression_error, is_expr_type,
};

use crate::lagrangian::Lagrangian;
use crate::lagrangian_internal::sealed::LagrangianInternal;
use crate::types::LinearRhsInput;

/// Coupled-cluster time-averaged quasienergy Lagrangian without orbital relaxation.
///
/// `LagrangianCc` builds and stores the symbolic ingredients needed to compute
/// coupled-cluster response functions and residues through the [`Lagrangian`]
/// interface.
///
/// The stored expression contains:
///
/// - coupled-cluster amplitudes as wave function parameters,
/// - coupled-cluster Lagrangian multipliers,
/// - the time-dependent cluster operator,
/// - the Lambda, or de-excitation, operator,
/// - the coupled-cluster quasienergy,
/// - the equation used to build right-hand sides for multiplier responses, and
/// - the full time-averaged quasienergy Lagrangian.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LagrangianCc {
    // Coupled-cluster amplitudes
    cc_amplitude: Arc<dyn Expr>,
    // Lagrangian multipliers
    cc_multiplier: Arc<dyn Expr>,
    // Time-dependent cluster operator
    cluster_operator: Arc<dyn Expr>,
    // Lambda operator or de-excitation operator
    cc_lambda_operator: Arc<dyn Expr>,
    // Similarity-transformed Hamiltonian, or coupled-cluster quasienergy. To
    // compute equation (28), J. Phys. Chem. A 2025, 129, 3709-3721.
    cc_quasi_energy: Arc<dyn Expr>,
    // To compute the right-hand side of the response equation of Lagrangian
    // multipliers, ses equation (29), J. Phys. Chem. A 2025, 129, 3709-3721.
    cc_multiplier_equation: Arc<dyn Expr>,
    lagrangian_expr: Arc<dyn Expr>,
}

impl LagrangianCc {
    /// Returns the maximum commutator order used for the coupled-cluster
    /// similarity transformation.
    #[inline]
    pub fn max_commutator_order() -> u32 {
        4
    }

    /// Builds a coupled-cluster time-averaged quasienergy Lagrangian by
    /// following equation (20), J. Phys. Chem. A 2025, 129, 3709-3721.
    ///
    /// This constructor builds the cluster operator, the Lambda operator, the
    /// coupled-cluster quasienergy, the multiplier response equation, and the
    /// full Lagrangian expression.
    ///
    /// Perturbing operators are required not to contain an unperturbed term.
    /// The coupled-cluster amplitude must be a `tinned::WfnParameter`, the
    /// excitation operator must be a `tinned::ExcitationOperator`, and the
    /// multiplier must be a `tinned::LagMultiplier`.
    ///
    /// # Arguments
    ///
    /// * `unperturbed_hamiltonian` - Unperturbed Hamiltonian.
    /// * `perturbing_operators` - Perturbing operators without zeroth-order
    ///   terms.
    /// * `cc_amplitude` - Coupled-cluster amplitudes.
    /// * `cc_excitation_operator` - Coupled-cluster excitation operator.
    /// * `cc_multiplier` - Coupled-cluster Lagrangian multipliers.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - any perturbing operator contains an unperturbed term,
    /// - `cc_amplitude` is not a `tinned::WfnParameter`,
    /// - `cc_excitation_operator` is not a `tinned::ExcitationOperator`,
    /// - `cc_multiplier` is not a `tinned::LagMultiplier`, or
    /// - construction of any symbolic expression fails.
    pub fn new(
        unperturbed_hamiltonian: Arc<dyn Expr>,
        perturbing_operators: &[Arc<dyn Expr>],
        cc_amplitude: Arc<dyn Expr>,
        cc_excitation_operator: Arc<dyn Expr>,
        cc_multiplier: Arc<dyn Expr>,
    ) -> Result<Self, TinnedError> {
        // Build terms for similarity-transformed Hamiltonian, or coupled-cluster quasienergy
        fn build_quasi_energy_term(
            cluster_operator: &Arc<dyn Expr>,
            electron_operator: &Arc<dyn Expr>,
        ) -> Result<Arc<dyn Expr>, TinnedError> {
            ExpAdjointMap::builder(cluster_operator.clone(), electron_operator.clone(), Some(true))
                .left_action(false)
                .max_commutator_order(LagrangianCc::max_commutator_order())
                .build()
        }

        // Build terms for the right-hand side of the response equation of Lagrangian multipliers
        fn build_multiplier_equation_term(
            cc_excitation_operator: &Arc<dyn Expr>,
            cluster_operator: &Arc<dyn Expr>,
            electron_operator: &Arc<dyn Expr>,
        ) -> Result<Arc<dyn Expr>, TinnedError> {
            build_quasi_energy_term(
                cluster_operator,
                &AdjointMap::new(
                    vec![cc_excitation_operator.clone()],
                    electron_operator.clone(),
                    Some(false),
                    Some(AdjointMode::Commutative),
                )?,
            )
        }

        // We require perturbing operators do not have zeroth-order/unperturbed term
        for op in perturbing_operators {
            if op.has_unperturbed_term() {
                return Err(expression_error(
                    "Perturbing operator should not have zeroth-order term",
                    op,
                    None,
                ));
            }
        }
        // Check types of coupled-cluster amplitudes and Lagrangian multipliers
        if !is_expr_type::<WfnParameter>(&cc_amplitude) {
            return Err(expression_error(
                "Invalid type of coupled-cluster amplitudes",
                &cc_amplitude,
                None,
            ));
        }
        if !is_expr_type::<ExcitationOperator>(&cc_excitation_operator) {
            return Err(expression_error(
                "Invalid type of coupled-cluster excitation operator",
                &cc_excitation_operator,
                None,
            ));
        }
        if !is_expr_type::<LagMultiplier>(&cc_multiplier) {
            return Err(expression_error(
                "Invalid type of Lagrangian multipliers",
                &cc_multiplier,
                None,
            ));
        }

        // Theoretically. the following dot products allow for swaping
        // excitation opertors and CC amplitudes/multipliers. But that does not
        // give us any benefit for symbolic differentiation and computation. We
        // simply set it as `false` here.
        let cluster_operator = DotProduct::new(
            cc_excitation_operator.clone(),
            false,
            cc_amplitude.clone(),
            false,
            Some(false),
        )?;
        let cc_lambda_operator = DotProduct::new(
            cc_excitation_operator.clone(),
            true,
            cc_multiplier.clone(),
            false,
            Some(false),
        )?;

        // Unperturbed Hamiltonian and perturbation operators, see Equations
        // (2) and (5), J. Phys. Chem. A 2025, 129, 3709-3721.
        let num_elec_operators = perturbing_operators.len() + 1;

        // Terms to construct Equation (28), J. Phys. Chem. A 2025, 129, 3709-3721.
        let mut quasi_energy_terms = Vec::with_capacity(num_elec_operators);
        // Terms to construct Equation (29), J. Phys. Chem. A 2025, 129, 3709-3721.
        let mut multiplier_eq_terms = Vec::with_capacity(2 * num_elec_operators);
        // Terms to construct coupled-cluster quasienergy Lagrangian, Equation
        // (20), J. Phys. Chem. A 2025, 129, 3709-3721.
        let mut lagrangian_terms = Vec::with_capacity(2 * num_elec_operators + 1);

        let mut quasi_energy_term =
            build_quasi_energy_term(&cluster_operator, &unperturbed_hamiltonian)?;
        quasi_energy_terms.push(quasi_energy_term.clone());
        lagrangian_terms.push(quasi_energy_term.clone());
        lagrangian_terms.push(MatrixMul::new(vec![cc_lambda_operator.clone(), quasi_energy_term])?);

        let mut multiplier_eq_term = build_multiplier_equation_term(
            &cc_excitation_operator,
            &cluster_operator,
            &unperturbed_hamiltonian,
        )?;
        multiplier_eq_terms.push(multiplier_eq_term.clone());
        multiplier_eq_terms
            .push(MatrixMul::new(vec![cc_lambda_operator.clone(), multiplier_eq_term])?);

        for oper in perturbing_operators {
            quasi_energy_term = build_quasi_energy_term(&cluster_operator, oper)?;
            quasi_energy_terms.push(quasi_energy_term.clone());
            lagrangian_terms.push(quasi_energy_term.clone());
            lagrangian_terms
                .push(MatrixMul::new(vec![cc_lambda_operator.clone(), quasi_energy_term])?);

            multiplier_eq_term =
                build_multiplier_equation_term(&cc_excitation_operator, &cluster_operator, oper)?;
            multiplier_eq_terms.push(multiplier_eq_term.clone());
            multiplier_eq_terms
                .push(MatrixMul::new(vec![cc_lambda_operator.clone(), multiplier_eq_term])?);
        }

        let cc_quasi_energy = MatrixAdd::new(quasi_energy_terms)?;
        let cc_multiplier_equation = MatrixAdd::new(multiplier_eq_terms)?;

        // Perform -i*d/dt (backward) on coupled-cluster amplitudes
        let dt_cc_amplitude =
            TimeEvolution::builder(cc_amplitude.clone()).is_forward(false).build()?;

        // Here, we should have an inner product (`DotProduct`) between
        // Lagrangian multipliers and the time-differentiated coupled-cluster
        // amplitudes instead of `MatrixMul`. But it will be problematic for
        // the sum of `lagrangian_terms` unless we make both `ExpAdjointMap` and
        // `AdjointMap` be scalar (a bit weird too) or wrapped in another
        // scalar `Expr` like `ExpectationValue` (unnecessary layer for users).
        lagrangian_terms.push(MatrixMul::new(vec![cc_multiplier.clone(), dt_cc_amplitude])?);
        let lagrangian_expr = MatrixAdd::new(lagrangian_terms)?;

        Ok(Self {
            cc_amplitude,
            cc_multiplier,
            cluster_operator,
            cc_lambda_operator,
            cc_quasi_energy,
            cc_multiplier_equation,
            lagrangian_expr,
        })
    }

    /// Builds the right-hand side of a linear response equation for a given
    /// response parameter.
    ///
    /// The response parameter is expected to be derived from this Lagrangian's
    /// coupled-cluster amplitudes or Lagrangian multipliers, and must be one of:
    ///
    /// - a `tinned::WfnParameter` representing differentiated coupled-cluster
    ///   amplitudes,
    /// - a differentiated `tinned::LagMultiplier` representing Lagrangian
    ///   multipliers, or
    /// - a `tinned::ResidueParameter` whose inner parameter is either a
    ///   `tinned::WfnParameter` or a `tinned::LagMultiplier`.
    ///
    /// For differentiated coupled-cluster amplitudes and Lagrangian
    /// multipliers, equations (28) and (29) in J. Phys. Chem. A 2025,
    /// 129, 3709–3721 are used to construct the right-hand side.
    ///
    /// When `rsp_parameter` is a `tinned::ResidueParameter`, this method
    /// constructs the right-hand side of higher-order residues. It rejects
    /// residue amplitudes or residue multipliers when
    /// `tinned::ResidueParameter::parameter`'s derivative is equivalent to
    /// `tinned::ResidueParameter::perturbations`; such cases should instead be
    /// obtained from the left and right eigenvectors of the nonsymmetric
    /// Jacobian.
    ///
    /// For higher-order residues,
    /// `tinned::ResidueParameter::parameter`'s derivative is a superchain of
    /// `tinned::ResidueParameter::perturbations`. All terms not containing
    /// `tinned::ResidueParameter::parameter` or its higher-order derivatives
    /// are removed, and the retained (un)differentiated
    /// `tinned::ResidueParameter::parameter` instances are replaced with the
    /// corresponding residue `tinned::ResidueParameter` instances.
    ///
    /// # Arguments
    ///
    /// * `rsp_parameter` - The response parameter for which the right-hand side
    ///   is constructed.
    /// * `num_tol` - Optional numerical tolerance used to determine whether a
    ///   `tinned::Number` should be treated as zero.
    ///
    /// # Errors
    ///
    /// Returns an error if `rsp_parameter` has an unsupported type, if a
    /// residue parameter wraps an unsupported inner parameter, if
    /// `rsp_parameter` is not derived from this Lagrangian's coupled-cluster
    /// amplitudes or Lagrangian multipliers, if the method is called for a
    /// residue amplitude or residue multiplier, or if symbolic construction
    /// fails.
    pub fn linear_response_rhs(
        &self,
        rsp_parameter: &Arc<dyn Expr>,
        num_tol: Option<NumberTolerance>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        let rhs_input = if let Some(cc_amplitude) = downcast_from_arc::<WfnParameter>(rsp_parameter)
        {
            if !cc_amplitude.match_one_self(&self.cc_amplitude, true) {
                return Err(expression_error(
                    format!(
                        "Response parameter is not derived from this Lagrangian's coupled-cluster amplitude {}",
                        &self.cc_amplitude
                    ),
                    rsp_parameter,
                    None,
                ));
            }

            LinearRhsInput {
                equation: &self.cc_quasi_energy,
                derivative: cc_amplitude.derivative(),
                diff_parameter: rsp_parameter,
                residue_info: None,
            }
        } else if let Some(multiplier) = downcast_from_arc::<LagMultiplier>(rsp_parameter) {
            if !multiplier.match_one_self(&self.cc_multiplier, true) {
                return Err(expression_error(
                    format!(
                        "Response parameter is not derived from this Lagrangian's multiplier {}",
                        &self.cc_multiplier
                    ),
                    rsp_parameter,
                    None,
                ));
            }

            LinearRhsInput {
                equation: &self.cc_multiplier_equation,
                derivative: multiplier.derivative(),
                diff_parameter: rsp_parameter,
                residue_info: None,
            }
        } else if let Some(res_param) = downcast_from_arc::<ResidueParameter>(rsp_parameter) {
            let diff_parameter = res_param.parameter();

            if let Some(cc_amplitude) = downcast_from_arc::<WfnParameter>(diff_parameter) {
                if !cc_amplitude.match_one_self(&self.cc_amplitude, true) {
                    return Err(expression_error(
                        format!(
                            "Residue response parameter is not derived from this Lagrangian's coupled-cluster amplitude {}",
                            &self.cc_amplitude
                        ),
                        rsp_parameter,
                        None,
                    ));
                }

                // `ResidueParameter` ensures that `res_param.perturbations()`
                // is a subchain of `cc_amplitude.derivative()`, so we check if
                // the former is also a superchain of the latter.
                if cc_amplitude.derivative().is_superchain_vec(res_param.perturbations()) {
                    return Err(expression_error(
                        "linear_response_rhs() should not be called for a residue CC amplitude",
                        rsp_parameter,
                        None,
                    ));
                }

                LinearRhsInput {
                    equation: &self.cc_quasi_energy,
                    derivative: cc_amplitude.derivative(),
                    diff_parameter,
                    residue_info: Some((res_param, self.cc_amplitude.clone())),
                }
            } else if let Some(multiplier) = downcast_from_arc::<LagMultiplier>(diff_parameter) {
                if !multiplier.match_one_self(&self.cc_multiplier, true) {
                    return Err(expression_error(
                        format!(
                            "Residue response parameter is not derived from this Lagrangian's multiplier {}",
                            &self.cc_multiplier
                        ),
                        rsp_parameter,
                        None,
                    ));
                }

                if multiplier.derivative().is_superchain_vec(res_param.perturbations()) {
                    return Err(expression_error(
                        "linear_response_rhs() should not be called for a residue Lagrangian multiplier",
                        rsp_parameter,
                        None,
                    ));
                }

                LinearRhsInput {
                    equation: &self.cc_multiplier_equation,
                    derivative: multiplier.derivative(),
                    diff_parameter,
                    residue_info: Some((res_param, self.cc_multiplier.clone())),
                }
            } else {
                return Err(expression_error(
                    "Invalid parameter type of residue parameter",
                    rsp_parameter,
                    None,
                ));
            }
        } else {
            return Err(expression_error(
                "Invalid type of response parameter",
                rsp_parameter,
                None,
            ));
        };

        let rhs = self.build_linear_rhs(rhs_input, num_tol)?;

        MatrixMul::new(vec![Number::minus_one(), rhs])
    }

    /// Returns the time-dependent cluster operator.
    #[inline]
    pub fn cluster_operator(&self) -> &Arc<dyn Expr> {
        &self.cluster_operator
    }

    /// Returns the Lambda, or de-excitation, operator.
    #[inline]
    pub fn cc_lambda_operator(&self) -> &Arc<dyn Expr> {
        &self.cc_lambda_operator
    }
}

impl LagrangianInternal for LagrangianCc {
    #[inline]
    fn eliminate_wfn_parameter(
        &self,
        lagrangian: &Arc<dyn Expr>,
        exten_perturbations: &[Arc<Perturbation>],
        min_wfn_order: u32,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        lagrangian.eliminate(self.cc_amplitude.clone(), exten_perturbations, min_wfn_order)
    }

    #[inline]
    fn eliminate_lag_multipliers(
        &self,
        lagrangian: &Arc<dyn Expr>,
        exten_perturbations: &[Arc<Perturbation>],
        min_multiplier_order: u32,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        lagrangian.eliminate(self.cc_multiplier.clone(), exten_perturbations, min_multiplier_order)
    }
}

impl Lagrangian for LagrangianCc {
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline]
    fn get_lagrangian(&self) -> &Arc<dyn Expr> {
        &self.lagrangian_expr
    }

    #[inline]
    fn get_wfn_parameters(&self) -> Vec<Arc<dyn Expr>> {
        vec![self.cc_amplitude.clone()]
    }

    #[inline]
    fn get_lagrangian_multipliers(&self) -> Vec<Arc<dyn Expr>> {
        vec![self.cc_multiplier.clone()]
    }
}
