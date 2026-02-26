use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tinned::{
    AdjointMap, DotProduct, ExpAdjointMap, Expr, LagMultiplier, MatrixAdd, MatrixMul, Perturbation,
    ResidueParameter, TemporumOperator, TinnedError, WfnParameter, differentiate_expr,
    downcast_from_arc, expression_error, is_expr_type,
};

use crate::lagrangian::Lagrangian;
use crate::lagrangian_internal::sealed::LagrangianInternal;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LagrangianCc {
    perturbing_operators: HashSet<Arc<dyn Expr>>,
    // Coupled-cluster amplitudes
    cc_amplitudes: Arc<dyn Expr>,
    // Lagrangian multipliers
    multipliers: Arc<dyn Expr>,
    // Similarity-transformed Hamiltonian, or exponential map. To compute
    // Equation (28), J. Phys. Chem. A 2025, 129, 3709-3721.
    cc_hamiltonian: Arc<dyn Expr>,
    // To compute the right-hand side of the response equation of Lagrangian
    // multipliers, ses Equation (29), J. Phys. Chem. A 2025, 129, 3709-3721.
    rhs_multiplier: Arc<dyn Expr>,
    lagrangian_expr: Arc<dyn Expr>,
}

impl LagrangianCc {
    // Builds time-averaged quasi-energy Lagrangian for coupled-cluster models
    // without orbital relaxation.
    pub fn new(
        unperturbed_hamiltonian: Arc<dyn Expr>,
        perturbing_operators: &[Arc<dyn Expr>],
        cc_amplitudes: Arc<dyn Expr>,
        excitation_operators: Arc<dyn Expr>,
        multipliers: Arc<dyn Expr>,
    ) -> Result<Self, TinnedError> {
        // Check types of coupled-cluster amplitudes and Lagrangian multipliers
        if !is_expr_type::<WfnParameter>(&cc_amplitudes) {
            return Err(expression_error(
                "Invalid type of coupled-cluster amplitudes",
                &cc_amplitudes,
                None,
            ));
        }
        if !is_expr_type::<LagMultiplier>(&multipliers) {
            return Err(expression_error(
                "Invalid type of Lagrangian multipliers",
                &multipliers,
                None,
            ));
        }

        // Theoretically. the following dot products allow for swaping
        // excitation opertors and CC amplitudes/multipliers. But that does not
        // give us any benefit for symbolic differentiation and computation. We
        // simply set it as `false` here.
        let cluster_operator = DotProduct::new(
            excitation_operators.clone(),
            false,
            cc_amplitudes.clone(),
            false,
            Some(false),
        )?;
        let lambda_operator = DotProduct::new(
            excitation_operators.clone(),
            true,
            multipliers.clone(),
            false,
            Some(false),
        )?;

        // Unperturbed Hamiltonian and perturbation operators, see Equations
        // (2) and (5), J. Phys. Chem. A 2025, 129, 3709-3721.
        let len_hamiltonian_terms = perturbing_operators.len() + 1;

        // Terms to construct Equation (28), J. Phys. Chem. A 2025, 129, 3709-3721.
        let mut cc_hamiltonian_terms = Vec::with_capacity(len_hamiltonian_terms);
        // Terms to construct Equation (29), J. Phys. Chem. A 2025, 129, 3709-3721.
        let mut multiplier_terms = Vec::with_capacity(2 * len_hamiltonian_terms);
        // Terms to construct coupled-cluster quasi-energy Lagrangian, Equation
        // (20), J. Phys. Chem. A 2025, 129, 3709-3721.
        let mut lag_terms = Vec::with_capacity(multiplier_terms.capacity() + 1);

        let mut hamiltonian_term =
            ExpAdjointMap::builder(cluster_operator.clone(), unperturbed_hamiltonian.clone())
                .left_action(false)
                .max_fold(4)
                .build()?;
        cc_hamiltonian_terms.push(hamiltonian_term.clone());
        lag_terms.push(hamiltonian_term.clone());
        lag_terms.push(MatrixMul::new(vec![lambda_operator.clone(), hamiltonian_term])?);

        let mut multiplier_term = ExpAdjointMap::builder(
            cluster_operator.clone(),
            AdjointMap::new(
                vec![excitation_operators.clone()],
                unperturbed_hamiltonian.clone(),
                Some(false),
            )?,
        )
        .left_action(false)
        .max_fold(4)
        .build()?;
        multiplier_terms.push(multiplier_term.clone());
        multiplier_terms.push(MatrixMul::new(vec![lambda_operator.clone(), multiplier_term])?);

        for oper in perturbing_operators {
            hamiltonian_term = ExpAdjointMap::builder(cluster_operator.clone(), oper.clone())
                .left_action(false)
                .max_fold(4)
                .build()?;
            cc_hamiltonian_terms.push(hamiltonian_term.clone());
            lag_terms.push(hamiltonian_term.clone());
            lag_terms.push(MatrixMul::new(vec![lambda_operator.clone(), hamiltonian_term])?);

            multiplier_term = ExpAdjointMap::builder(
                cluster_operator.clone(),
                AdjointMap::new(vec![excitation_operators.clone()], oper.clone(), Some(false))?,
            )
            .left_action(false)
            .max_fold(4)
            .build()?;
            multiplier_terms.push(multiplier_term.clone());
            multiplier_terms.push(MatrixMul::new(vec![lambda_operator.clone(), multiplier_term])?);
        }

        let cc_hamiltonian = MatrixAdd::new(cc_hamiltonian_terms)?;
        let rhs_multiplier = MatrixAdd::new(multiplier_terms)?;

        // Perform -i*d/dt (backward) on coupled-cluster amplitudes
        let dt_cc_amplitudes =
            TemporumOperator::builder(cc_amplitudes.clone()).is_forward(false).build()?;

        // Here, we should have an inner product (`DotProduct`) between
        // Lagrangian multipliers and the time-differentiated coupled-cluster
        // amplitudes instead of `MatrixMul`. But it will be problematic for
        // the sum of `lag_terms` unless we make both `ExpAdjointMap` and
        // `AdjointMap` be scalar (a bit weird too) or wrapped in another
        // scalar `Expr` like `ExpectationValue` (unnecessary layer for users).
        lag_terms.push(MatrixMul::new(vec![multipliers.clone(), dt_cc_amplitudes])?);
        let lagrangian_expr = MatrixAdd::new(lag_terms)?;

        // Users may accidentally provide duplicated perturbation operators,
        // but it does not matter for the field `perturbing_operators`
        // because we use the field only for removing undifferentiated
        // perturbation operators.
        Ok(Self {
            perturbing_operators: perturbing_operators.iter().cloned().collect(),
            cc_amplitudes,
            multipliers,
            cc_hamiltonian,
            rhs_multiplier,
            lagrangian_expr,
        })
    }

    // Builds time-averaged quasi-energy Lagrangian for coupled-cluster models
    // with orbital relaxation.
    //
    // Using one- and two-particle density matrices
    //#[inline]
    //pub fn new_orbital_relaxed(
    //) -> Result<Self, TinnedError> {
    //}

    // Returns right-hand side (RHS) of the (linear) response equation.
    // `rsp_parameter`, which can be the type of `WfnParameter`
    // (coupled-cluster amplitudes), `LagMultiplier` (Lagrangian multipliers),
    // or `ResidueParameter` (for residues).
    //
    // (1) For types `WfnParameter` and `LagMultiplier`, we simply follow, for
    //     example, Equations (28) and (29), J. Phys. Chem. A 2025, 129, 3709-3721.
    //
    // (2) For type `ResidueParameter`, we need to check its field
    //     `parameter`, which must be either type WfnParameter` or `LagMultiplier`.
    //
    // 2a) If `parameter`'s derivative is equivalent to `perturbations` of
    //     `ResidueParameter`, we have a residue CC amplitude or Lagrangian
    //     multiplier, which may be solved from the left and right eigenvectors
    //     of the nonsymmetric Jacobian, and users should not call this method.
    //
    // 2b) If `parameter`'s derivative is a superchain of `perturbations`, we
    //     have a higher-order residue CC amplitude or Lagrangian multiplier.
    //     We need to remove all terms not containing `parameter` or its
    //     higher-order differentiated ones, and replace retained
    //     (un)differentiated `parameter`'s with corresponding residue
    //     `parameter`'s.
    //
    // Note that `rsp_parameter` should be a differentiated
    // `self.cc_amplitudes` or `self.multipliers`, otherwise the result will be
    // incorrect.
    #[inline]
    pub fn linear_response_rhs(
        &self,
        rsp_parameter: Arc<dyn Expr>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        // Undifferentiated perturbation operators and the perturbed
        // `rsp_parameter` itself will be removed from RHS
        let mut rhs_set = self.perturbing_operators.clone();
        rhs_set.insert(rsp_parameter.clone());

        if let Some(cc_amplitude) = downcast_from_arc::<WfnParameter>(&rsp_parameter) {
            let result = differentiate_expr(&self.cc_hamiltonian, cc_amplitude.derivative())?;

            result.remove(&rhs_set)
        } else if let Some(multiplier) = downcast_from_arc::<LagMultiplier>(&rsp_parameter) {
            let result = differentiate_expr(&self.rhs_multiplier, multiplier.derivative())?;

            result.remove(&rhs_set)
        } else if let Some(res_param) = downcast_from_arc::<ResidueParameter>(&rsp_parameter) {
            let (diff_rhs, unperturbed_param) = if let Some(cc_amplitude) =
                downcast_from_arc::<WfnParameter>(res_param.parameter())
            {
                // `ResidueParameter` ensures that `res_param.perturbations()`
                // is a subchain of `cc_amplitude.derivative()`, so we check if
                // the former is also a superchain of the latter.
                if cc_amplitude.derivative().is_superchain_vec(res_param.perturbations()) {
                    return Err(expression_error(
                        "linear_response_rhs() should not be called for a residue CC amplitude",
                        &rsp_parameter,
                        None,
                    ));
                }

                let result = differentiate_expr(&self.cc_hamiltonian, cc_amplitude.derivative())?;

                (result.remove(&rhs_set)?, self.cc_amplitudes.clone())
            } else if let Some(multiplier) =
                downcast_from_arc::<LagMultiplier>(&res_param.parameter())
            {
                if multiplier.derivative().is_superchain_vec(res_param.perturbations()) {
                    return Err(expression_error(
                        "linear_response_rhs() should not be called for a residue Lagrangian multiplier",
                        &rsp_parameter,
                        None,
                    ));
                }

                let result = differentiate_expr(&self.rhs_multiplier, multiplier.derivative())?;

                (result.remove(&rhs_set)?, self.multipliers.clone())
            } else {
                return Err(expression_error(
                    "Invalid parameter type of residue parameter",
                    &rsp_parameter,
                    None,
                ));
            };

            let residue_info: HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)> =
                std::iter::once((
                    res_param.excited_state().clone(),
                    (res_param.positive_frequency(), res_param.perturbations().to_vec()),
                ))
                .collect();

            let (residue_set, residue_map) =
                self.build_residue_parameters(&vec![unperturbed_param], &residue_info)?;

            diff_rhs.retain(&residue_set, false)?.replace(&residue_map, false)
        } else {
            return Err(expression_error(
                "Invalid type of response parameter",
                &rsp_parameter,
                None,
            ));
        }
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
        lagrangian.eliminate(&self.cc_amplitudes, exten_perturbations, min_wfn_order)
    }

    #[inline]
    fn eliminate_lag_multipliers(
        &self,
        lagrangian: &Arc<dyn Expr>,
        exten_perturbations: &[Arc<Perturbation>],
        min_multiplier_order: u32,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        lagrangian.eliminate(&self.multipliers, exten_perturbations, min_multiplier_order)
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
    fn get_wfn_parameter(&self) -> Vec<Arc<dyn Expr>> {
        vec![self.cc_amplitudes.clone()]
    }

    #[inline]
    fn get_lag_multiplier(&self) -> Vec<Arc<dyn Expr>> {
        vec![self.multipliers.clone()]
    }
}
