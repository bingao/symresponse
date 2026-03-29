use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tinned::{
    AdjointMap, AdjointMode, DotProduct, ExpAdjointMap, Expr, MatrixAdd, NumberTolerance,
    Perturbation, ResidueParameter, TinnedError, WfnParameter, differentiate_expr,
    downcast_from_arc, expression_error, is_expr_type,
};

use crate::lagrangian::Lagrangian;
use crate::lagrangian_internal::sealed::LagrangianInternal;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LagrangianMcscf {
    // Perturbing operators
    perturbing_operators: HashSet<Arc<dyn Expr>>,
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
        // Check the type of orbital- and state-rotation parameters
        if !is_expr_type::<WfnParameter>(&rotation_parameters) {
            return Err(expression_error(
                "Invalid type of orbital- and state-rotation parameters",
                &rotation_parameters,
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

        let mut term = ExpAdjointMap::builder(
            lambda_operator.clone(),
            unperturbed_hamiltonian.clone(),
            Some(false),
        )
        .left_action(false)
        .build()?;
        lag_terms.push(term.clone());
        // Note that `rhs_terms` should be multiplied by `Number::imaginary_unit()`
        rhs_terms.push(AdjointMap::new(
            vec![rotation_operators.clone()],
            term,
            Some(true),
            Some(AdjointMode::Symmetric),
        )?);

        for oper in perturbing_operators {
            term = ExpAdjointMap::builder(lambda_operator.clone(), oper.clone(), Some(false))
                .left_action(false)
                .build()?;
            lag_terms.push(term.clone());
            rhs_terms.push(AdjointMap::new(
                vec![rotation_operators.clone()],
                term,
                Some(true),
                Some(AdjointMode::Symmetric),
            )?);
        }

        term = ExpAdjointMap::builder_temporum(lambda_operator, false, Some(false))
            .left_action(false)
            .build()?;
        lag_terms.push(term.clone());
        rhs_terms.push(AdjointMap::new(
            vec![rotation_operators.clone()],
            term,
            Some(true),
            Some(AdjointMode::Symmetric),
        )?);

        let lagrangian_expr = MatrixAdd::new(lag_terms)?;
        let rhs_parameters = MatrixAdd::new(rhs_terms)?;

        // Users may accidentally provide duplicated perturbation operators,
        // but it does not matter for the field `perturbing_operators`
        // because we use the field only for removing undifferentiated
        // perturbation operators.
        Ok(Self {
            perturbing_operators: perturbing_operators.iter().cloned().collect(),
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
    #[inline]
    pub fn linear_response_rhs(
        &self,
        rsp_parameter: Arc<dyn Expr>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        // Undifferentiated perturbation operators and the perturbed
        // `rsp_parameter` itself will be removed from RHS
        let mut rhs_set = self.perturbing_operators.clone();
        rhs_set.insert(rsp_parameter.clone());

        if let Some(rot_param) = downcast_from_arc::<WfnParameter>(&rsp_parameter) {
            let result = differentiate_expr(&self.rhs_parameters, rot_param.derivative())?;

            result.remove(&rhs_set)
        } else if let Some(res_param) = downcast_from_arc::<ResidueParameter>(&rsp_parameter) {
            if let Some(rot_param) = downcast_from_arc::<WfnParameter>(res_param.parameter()) {
                // `ResidueParameter` ensures that `res_param.perturbations()`
                // is a subchain of `rot_param.derivative()`, so we check if
                // the former is also a superchain of the latter.
                if rot_param.derivative().is_superchain_vec(res_param.perturbations()) {
                    return Err(expression_error(
                        "linear_response_rhs() should not be called for residue rotation parameters",
                        &rsp_parameter,
                        None,
                    ));
                }

                let rhs_deriv = differentiate_expr(&self.rhs_parameters, rot_param.derivative())?;

                let residue_info: HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)> =
                    std::iter::once((
                        res_param.excited_state().clone(),
                        (res_param.positive_frequency(), res_param.perturbations().to_vec()),
                    ))
                    .collect();

                let (residue_set, residue_map) = self.build_residue_parameters(
                    &vec![self.rotation_parameters.clone()],
                    &residue_info,
                )?;

                rhs_deriv.retain(&residue_set, true)?.replace(&residue_map, true)
            } else {
                Err(expression_error(
                    "Invalid parameter type of residue parameter",
                    &rsp_parameter,
                    None,
                ))
            }
        } else {
            Err(expression_error("Invalid type of response parameter", &rsp_parameter, None))
        }
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
        lagrangian.eliminate(&self.rotation_parameters, exten_perturbations, min_wfn_order)
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

    #[inline]
    fn at_zero_strength(
        &self,
        lagrangian: &Arc<dyn Expr>,
        num_tol: Option<NumberTolerance>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        // Remove undifferentiated perturbation operators and unperturbed
        // time-differentiated quantities
        lagrangian.remove(&self.perturbing_operators)?.apply_zero_rules(num_tol)
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
    fn get_wfn_parameter(&self) -> Vec<Arc<dyn Expr>> {
        vec![self.rotation_parameters.clone()]
    }

    #[inline]
    fn get_lag_multiplier(&self) -> Vec<Arc<dyn Expr>> {
        Vec::new()
    }
}
