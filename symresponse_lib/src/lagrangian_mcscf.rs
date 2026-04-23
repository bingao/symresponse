use std::sync::Arc;

use tinned::{
    AdjointMap, AdjointMode, DotProduct, ExcitationOperator, ExpAdjointMap, Expr, MatrixAdd,
    NumberTolerance, Perturbation, ResidueParameter, TinnedError, WfnParameter, downcast_from_arc,
    expression_error, is_expr_type,
};

use crate::lagrangian::Lagrangian;
use crate::lagrangian_internal::sealed::LagrangianInternal;
use crate::types::LinearRhsInput;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LagrangianMcscf {
    // Orbital- and state-rotation parameters
    rotation_parameters: Arc<dyn Expr>,
    // To compute the right-hand side of the response equation of orbital- and
    // state-rotation parameters, ses Equation (xx), ...
    rhs_parameters: Arc<dyn Expr>,
    lagrangian_expr: Arc<dyn Expr>,
}

impl LagrangianMcscf {
    // Builds the time-dependent quasi-energy for MCSCF
    pub fn new(
        unperturbed_hamiltonian: Arc<dyn Expr>,
        perturbing_operators: &[Arc<dyn Expr>],
        rotation_operators: Arc<dyn Expr>,
        rotation_parameters: Arc<dyn Expr>,
    ) -> Result<Self, TinnedError> {
        // Build terms for the right-hand side of the response equation of
        // orbital- and state-rotation parameters
        fn build_rhs_term(
            rotation_operators: &Arc<dyn Expr>,
            term: Arc<dyn Expr>,
        ) -> Result<Arc<dyn Expr>, TinnedError> {
            AdjointMap::new(
                vec![rotation_operators.clone()],
                term,
                Some(true),
                Some(AdjointMode::Symmetrized),
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

        // Check types of orbital- and state-rotation parameters
        if !is_expr_type::<WfnParameter>(&rotation_parameters) {
            return Err(expression_error(
                "Invalid type of rotation parameters",
                &rotation_parameters,
                None,
            ));
        }
        if rotation_parameters.has_unperturbed_term() {
            return Err(expression_error(
                "Non-perturbing rotation parameters",
                &rotation_parameters,
                None,
            ));
        }
        if !is_expr_type::<ExcitationOperator>(&rotation_operators) {
            return Err(expression_error(
                "Invalid type of rotation generators",
                &rotation_operators,
                None,
            ));
        }

        // Theoretically, the Hermitian inner product between orbital- and
        // state-rotation operators and parameters allows for swapping. But
        // that does not give us any benefit for symbolic differentiation and
        // computation. We simply set it as `false` here.
        let lambda_operator = DotProduct::new(
            rotation_operators.clone(),
            true,
            rotation_parameters.clone(),
            false,
            Some(false),
        )?;

        let len_lag_terms = perturbing_operators.len() + 2;
        let mut lag_terms = Vec::with_capacity(len_lag_terms);
        let mut rhs_terms = Vec::with_capacity(len_lag_terms);

        let mut term =
            ExpAdjointMap::builder(lambda_operator.clone(), unperturbed_hamiltonian, Some(false))
                .left_action(false)
                .build()?;
        lag_terms.push(term.clone());
        // Note that `rhs_terms` should be multiplied by `Number::imaginary_unit()`
        rhs_terms.push(build_rhs_term(&rotation_operators, term)?);

        for oper in perturbing_operators {
            term = ExpAdjointMap::builder(lambda_operator.clone(), oper.clone(), Some(false))
                .left_action(false)
                .build()?;
            lag_terms.push(term.clone());
            rhs_terms.push(build_rhs_term(&rotation_operators, term)?);
        }

        term = ExpAdjointMap::builder_time_evolution(lambda_operator, false, Some(false))
            .left_action(false)
            .build()?;
        lag_terms.push(term.clone());
        rhs_terms.push(build_rhs_term(&rotation_operators, term)?);

        let lagrangian_expr = MatrixAdd::new(lag_terms)?;
        let rhs_parameters = MatrixAdd::new(rhs_terms)?;

        Ok(Self {
            rotation_parameters,
            rhs_parameters,
            lagrangian_expr,
        })
    }

    // Returns right-hand side (RHS) of the (linear) response equation.
    // `rsp_parameter`, which can be the type of `WfnParameter` (orbital- and
    // state-rotation parameters), or `ResidueParameter` (for residues).
    //
    // (1) For types `WfnParameter`, we simply follow, for example, Equations
    //     (54) and (56), in manuscript.
    //
    // (2) For type `ResidueParameter`, we need to check its field
    //     `parameter`, which must be the type WfnParameter`.
    //
    // 2a) If `parameter`'s derivative is equivalent to `perturbations` of
    //     `ResidueParameter`, we have residue rotation parameters, which
    //     should be solved separately, and users should not call this method.
    //
    // 2b) If `parameter`'s derivative is a superchain of `perturbations`, we
    //     have higher-order residue rotation parameters.  We need to remove
    //     all terms not containing `parameter` or its higher-order
    //     differentiated ones, and replace retained (un)differentiated
    //     `parameter`'s with corresponding residue `parameter`'s.
    //
    // Note that `rsp_parameter` should be a differentiated
    // `self.rotation_parameters`, otherwise the result will be incorrect.
    pub fn linear_response_rhs(
        &self,
        rsp_parameter: &Arc<dyn Expr>,
        num_tol: Option<NumberTolerance>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        let rhs_input = if let Some(rot_param) = downcast_from_arc::<WfnParameter>(rsp_parameter) {
            LinearRhsInput {
                equation: &self.rhs_parameters,
                derivative: rot_param.derivative(),
                diff_parameter: rsp_parameter,
                residue_info: None,
            }
        } else if let Some(res_param) = downcast_from_arc::<ResidueParameter>(rsp_parameter) {
            let diff_parameter = res_param.parameter();

            if let Some(rot_param) = downcast_from_arc::<WfnParameter>(diff_parameter) {
                // `ResidueParameter` ensures that `res_param.perturbations()`
                // is a subchain of `rot_param.derivative()`, so we check if
                // the former is also a superchain of the latter.
                if rot_param.derivative().is_superchain_vec(res_param.perturbations()) {
                    return Err(expression_error(
                        "linear_response_rhs() should not be called for residue rotation parameters",
                        rsp_parameter,
                        None,
                    ));
                }

                LinearRhsInput {
                    equation: &self.rhs_parameters,
                    derivative: rot_param.derivative(),
                    diff_parameter,
                    residue_info: Some((res_param, self.rotation_parameters.clone())),
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

        self.build_linear_rhs(rhs_input, num_tol)
    }
}

impl LagrangianInternal for LagrangianMcscf {
    #[inline]
    fn eliminate_wfn_parameter(
        &self,
        lagrangian: &Arc<dyn Expr>,
        exten_perturbations: &[Arc<Perturbation>],
        min_wfn_order: u32,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        lagrangian.eliminate(self.rotation_parameters.clone(), exten_perturbations, min_wfn_order)
    }

    #[inline]
    fn eliminate_lag_multipliers(
        &self,
        lagrangian: &Arc<dyn Expr>,
        _exten_perturbations: &[Arc<Perturbation>],
        _min_multiplier_order: u32,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        Ok(lagrangian.clone())
    }
}

impl Lagrangian for LagrangianMcscf {
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
        vec![self.rotation_parameters.clone()]
    }

    #[inline]
    fn get_lagrangian_multipliers(&self) -> Vec<Arc<dyn Expr>> {
        Vec::new()
    }
}
