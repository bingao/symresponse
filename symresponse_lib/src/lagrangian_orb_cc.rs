use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tinned::{
    Add, AdjointMap, DotProduct, ExpAdjointMap, Expr, LagMultiplier, MatrixAdd, MatrixMul,
    NumberTolerance, Perturbation, ResidueParameter, SubExpr, TemporumOperator, TinnedError, Trace,
    WfnParameter, differentiate_expr, downcast_from_arc, expression_error, generic_error,
    is_expr_type,
};

use crate::lagrangian::Lagrangian;
use crate::lagrangian_internal::sealed::LagrangianInternal;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LagrangianOrbCc {
    // One-electron operator
    one_elec_operator: Arc<dyn Expr>,
    // Two-electron operator
    two_elec_operator: Arc<dyn Expr>,
    // Coupled-cluster amplitudes
    cc_amplitudes: Arc<dyn Expr>,
    // Coupled-cluster Lagrangian multipliers
    cc_multipliers: Arc<dyn Expr>,
    // Orbital rotation parameters (amplitudes)
    orb_rotation_parameters: Arc<dyn Expr>,
    // Brillouin condition multipliers
    brillouin_multipliers: Arc<dyn Expr>,
    // Symbol for one-electron density matrix
    one_density_matrix: Arc<dyn Expr>,
    // Symbol two one-electron density matrix
    two_density_matrix: Arc<dyn Expr>,
    lagrangian_expr: Arc<dyn Expr>,
}

impl LagrangianOrbCc {
    // Builds orbital-relaxed coupled-cluster Lagrangian
    pub fn new(
        one_elec_operator: Arc<dyn Expr>,
        two_elec_operator: Arc<dyn Expr>,
        cc_amplitudes: Arc<dyn Expr>,
        cc_excitation_operators: Arc<dyn Expr>,
        cc_multipliers: Arc<dyn Expr>,
        orb_rotation_parameters: Arc<dyn Expr>,
        // Orbital rotation generators, $\hat{a}^{\dagger)_{r}\hat{a}_{s}$
        orb_rotation_generators: Arc<dyn Expr>,
        brillouin_multipliers: Arc<dyn Expr>,
    ) -> Result<Self, TinnedError> {
        // Check types of coupled-cluster amplitudes, orbital rotation
        // parameters and generators, as well as Lagrangian multipliers
        if !is_expr_type::<WfnParameter>(&cc_amplitudes) {
            return Err(expression_error(
                "Invalid type of coupled-cluster amplitudes",
                &cc_amplitudes,
                None,
            ));
        }
        if !is_expr_type::<LagMultiplier>(&cc_multipliers) {
            return Err(expression_error(
                "Invalid type of coupled-cluster Lagrangian multipliers",
                &cc_multipliers,
                None,
            ));
        }
        if !is_expr_type::<WfnParameter>(&orb_rotation_parameters) {
            return Err(expression_error(
                "Invalid type of orbital rotation parameters",
                &orb_rotation_parameters,
                None,
            ));
        }
        if !is_expr_type::<LagMultiplier>(&brillouin_multipliers) {
            return Err(expression_error(
                "Invalid type of Brillouin condition multipliers",
                &brillouin_multipliers,
                None,
            ));
        }

        // Theoretically. the following dot products allow for swaping CC
        // excitation opertors and amplitudes/multipliers. But that does not
        // give us any benefit for symbolic differentiation and computation. We
        // simply set it as `false` here.
        let cluster_operator = DotProduct::new(
            cc_excitation_operators.clone(),
            false,
            cc_amplitudes.clone(),
            false,
            Some(false),
        )?;
        let cc_lambda_oper = DotProduct::new(
            cc_excitation_operators.clone(),
            true,
            cc_multipliers.clone(),
            false,
            Some(false),
        )?;
        // Orbital rotation operator, and we also set `allow_braket_swap` as `false`.
        let kappa_operator = DotProduct::new(
            orb_rotation_generators.clone(),
            true,
            orb_rotation_parameters.clone(),
            false,
            Some(false),
        )?;

        // Similarity-transformed orbital rotation generator, e^{kappa} * E_{pq} * e^{-kappa}
        let kappa_transformed_generator =
            ExpAdjointMap::builder(kappa_operator.clone(), orb_rotation_generators.clone())
                .left_action(true)
                .build()?;
        let similarity_transformed_generator =
            ExpAdjointMap::builder(cluster_operator.clone(), kappa_transformed_generator.clone())
                .left_action(false)
                .max_fold(4) //FIXME: or 2, or infinite?
                .build()?;

        // Set one-electron density matrix
        let one_density_expr = MatrixAdd::new(vec![
            similarity_transformed_generator.clone(),
            MatrixMul::new(vec![cc_lambda_oper.clone(), similarity_transformed_generator.clone()])?,
        ])?;
        let one_density_matrix =
            SubExpr::builder("one-electron-density", one_density_expr.clone()).build()?;

        // Set two-electron density matrix
        //FIXME: here we simply use the expression of one-electron density matrix
        let two_density_matrix =
            SubExpr::builder("two-electron-density", one_density_expr).build()?;

        // Similarity-transformed Hamiltonian, e^{kappa} * H * e^{-kappa}
        let kappa_transformed_hamiltonian = MatrixAdd::new(vec![
            ExpAdjointMap::builder(kappa_operator.clone(), one_elec_operator.clone())
                .left_action(true)
                .build()?,
            ExpAdjointMap::builder(kappa_operator.clone(), two_elec_operator.clone())
                .left_action(true)
                .build()?,
        ])?;

        // Set up the Lagrangian
        let lagrangian_expr = Add::new(vec![
            Trace::new(MatrixMul::new(vec![
                one_density_matrix.clone(),
                one_elec_operator.clone(),
            ])?)?,
            Trace::new(MatrixMul::new(vec![
                two_density_matrix.clone(),
                two_elec_operator.clone(),
            ])?)?,
            DotProduct::new(
                // [\hat{a}^{\dagger)_{r}\hat{a}_{s}, e^{kappa} * H * e^{-kappa}]
                AdjointMap::new(
                    vec![orb_rotation_generators.clone()],
                    kappa_transformed_hamiltonian.clone(),
                    Some(false),
                )?,
                true,
                brillouin_multipliers.clone(),
                false,
                Some(true),
            )?,
        ])?;

        // Users may accidentally provide duplicated perturbation operators,
        // but it does not matter for the field `perturbing_operators`
        // because we use the field only for removing undifferentiated
        // perturbation operators.
        Ok(Self {
            one_elec_operator,
            two_elec_operator,
            cc_amplitudes,
            cc_multipliers,
            orb_rotation_parameters,
            brillouin_multipliers,
            one_density_matrix,
            two_density_matrix,
            lagrangian_expr,
        })
    }

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
    // J. Chem. Phys. 92, 4924-4940 (Apr. 1990)
    // notes: equations (10)-(13)
    #[inline]
    pub fn linear_response_rhs(
        &self,
        rsp_parameter: Arc<dyn Expr>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        Err(generic_error("Sum of extensive perturbations' frequencies failed", None))
    }
}

impl LagrangianInternal for LagrangianOrbCc {
    #[inline]
    fn eliminate_wfn_parameter(
        &self,
        lagrangian: &Arc<dyn Expr>,
        exten_perturbations: &[Arc<Perturbation>],
        min_wfn_order: u32,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        // See J. Chem. Phys. 92, 4924-4940
        let result = lagrangian.eliminate(
            &self.orb_rotation_parameters,
            exten_perturbations,
            min_wfn_order,
        )?;
        result.eliminate(&self.cc_amplitudes, exten_perturbations, min_wfn_order)
    }

    #[inline]
    fn eliminate_lag_multipliers(
        &self,
        lagrangian: &Arc<dyn Expr>,
        exten_perturbations: &[Arc<Perturbation>],
        min_multiplier_order: u32,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        // See J. Chem. Phys. 92, 4924-4940
        let result = lagrangian.eliminate(
            &self.brillouin_multipliers,
            exten_perturbations,
            min_multiplier_order,
        )?;
        result.eliminate(&self.cc_multipliers, exten_perturbations, min_multiplier_order)
        //FIXME: remove unperturbed `brillouin_multipliers`!
    }
}

// at_zero_strength(), e^{kappa} = 1 when kappa is not differentiated

impl Lagrangian for LagrangianOrbCc {
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
        vec![self.cc_amplitudes.clone(), self.orb_rotation_parameters.clone()]
    }

    #[inline]
    fn get_lag_multiplier(&self) -> Vec<Arc<dyn Expr>> {
        vec![self.cc_multipliers.clone(), self.brillouin_multipliers.clone()]
    }
}
