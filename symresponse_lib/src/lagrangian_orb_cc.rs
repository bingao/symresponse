use std::sync::Arc;

use tinned::{
    Add, AdjointMap, AdjointMode, DotProduct, ExcitationOperator, ExpAdjointMap, Expr,
    HermitianTranspose, LagMultiplier, MatrixAdd, MatrixMul, Number, NumberTolerance, Perturbation,
    SubExpr, TinnedError, Trace, WfnParameter, differentiate_expr, downcast_from_arc,
    expression_error, is_expr_type,
};

use crate::lagrangian::Lagrangian;
use crate::lagrangian_internal::sealed::LagrangianInternal;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LagrangianOrbCc {
    // One-electron operator
    one_elec_matrix: Arc<dyn Expr>,
    // Two-electron operator
    two_elec_matrix: Arc<dyn Expr>,
    // Coupled-cluster amplitude
    cc_amplitude: Arc<dyn Expr>,
    // Coupled-cluster operator
    cluster_operator: Arc<dyn Expr>,
    // Lambda operator or de-excitation operator
    cc_lambda_operator: Arc<dyn Expr>,
    // Response equation for coupled-cluster amplitude
    cc_amplitude_equation: Arc<dyn Expr>,
    // Coupled-cluster Lagrangian multiplier
    cc_multiplier: Arc<dyn Expr>,
    // Response equation for coupled-cluster Lagrangian multiplier
    cc_multiplier_equation: Arc<dyn Expr>,
    // Orbital rotation parameter (amplitude)
    orb_rot_parameter: Arc<dyn Expr>,
    // Orbital rotation operator
    kappa_operator: Arc<dyn Expr>,
    // Brillouin equation,
    brillouin_equation: Arc<dyn Expr>,
    // Brillouin condition multiplier
    brillouin_multiplier: Arc<dyn Expr>,
    // Response equation for Brillouin condition multiplier
    brillouin_multiplier_equation: Arc<dyn Expr>,
    // One-electron density matrix
    one_elec_density: Arc<dyn Expr>,
    // Two-electron density matrix
    two_elec_density: Arc<dyn Expr>,
    lagrangian_expr: Arc<dyn Expr>,
}

impl LagrangianOrbCc {
    // Maximum commutator order
    #[inline]
    pub fn max_commutator_order() -> u32 {
        4
    }

    // Build one- or two-electron density matrix
    #[inline]
    pub fn build_electron_density_matrix(
        name: impl Into<String>,
        kappa_operator: &Arc<dyn Expr>,
        excitation_operator: &Arc<dyn Expr>,
        cluster_operator: &Arc<dyn Expr>,
        cc_lambda_operator: &Arc<dyn Expr>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        // Similarity-transformed single or double excitation operator
        // e^{kappa} * E_{pq} * e^{-kappa}, or e^{kappa} * e_{pqrs} * e^{-kappa}
        let kappa_transformed_oper = ExpAdjointMap::builder(
            kappa_operator.clone(),
            excitation_operator.clone(),
            Some(false),
        )
        .left_action(true)
        .build()?;
        let cc_transformed_oper =
            ExpAdjointMap::builder(cluster_operator.clone(), kappa_transformed_oper, Some(true))
                .left_action(false)
                .max_commutator_order(Self::max_commutator_order())
                .build()?;
        // Set one- or two-electron density matrix
        let density_expr = MatrixAdd::new(vec![
            cc_transformed_oper.clone(),
            MatrixMul::new(vec![cc_lambda_operator.clone(), cc_transformed_oper])?,
        ])?;

        Ok(SubExpr::new(name, density_expr))
    }

    // Builds orbital-relaxed coupled-cluster Lagrangian
    pub fn new(
        one_elec_matrix: Arc<dyn Expr>,
        // Single excitation $E_{pq}$
        single_excitation_operator: Arc<dyn Expr>,
        two_elec_matrix: Arc<dyn Expr>,
        // Double excitation $e_{pqrs} = E_{pq}E_{rs} - \delta_{qr}E_{ps}$
        double_excitation_operator: Arc<dyn Expr>,
        cc_amplitude: Arc<dyn Expr>,
        cc_excitation_operator: Arc<dyn Expr>,
        cc_multiplier: Arc<dyn Expr>,
        orb_rot_parameter: Arc<dyn Expr>,
        // Orbital rotation generator, $E_{pq}-E_{qp}$
        orb_rot_generator: Arc<dyn Expr>,
        brillouin_multiplier: Arc<dyn Expr>,
    ) -> Result<Self, TinnedError> {
        // Checks types of inputs
        if !is_expr_type::<ExcitationOperator>(&single_excitation_operator) {
            return Err(expression_error(
                "Invalid type of single excitation operator",
                &single_excitation_operator,
                None,
            ));
        }
        if !is_expr_type::<ExcitationOperator>(&double_excitation_operator) {
            return Err(expression_error(
                "Invalid type of double excitation operator",
                &double_excitation_operator,
                None,
            ));
        }
        if !is_expr_type::<WfnParameter>(&cc_amplitude) {
            return Err(expression_error(
                "Invalid type of coupled-cluster amplitude",
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
                "Invalid type of coupled-cluster Lagrangian multiplier",
                &cc_multiplier,
                None,
            ));
        }
        if !is_expr_type::<WfnParameter>(&orb_rot_parameter) {
            return Err(expression_error(
                "Invalid type of orbital rotation parameter",
                &orb_rot_parameter,
                None,
            ));
        }
        if orb_rot_parameter.has_unperturbed_term() {
            return Err(expression_error(
                "Non-perturbing orbital rotation parameter",
                &orb_rot_parameter,
                None,
            ));
        }
        if !is_expr_type::<ExcitationOperator>(&orb_rot_generator) {
            return Err(expression_error(
                "Invalid type of orbital rotation generator",
                &orb_rot_generator,
                None,
            ));
        }
        if !is_expr_type::<LagMultiplier>(&brillouin_multiplier) {
            return Err(expression_error(
                "Invalid type of Brillouin condition multiplier",
                &brillouin_multiplier,
                None,
            ));
        }

        // Theoretically, the following dot products allow for swapping CC
        // excitation operator and amplitude/multiplier. But that does not
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
        // Orbital rotation operator, and we also set `allow_braket_swap` as `false`.
        let kappa_operator = DotProduct::new(
            orb_rot_generator.clone(),
            true,
            orb_rot_parameter.clone(),
            false,
            Some(false),
        )?;

        // Set one- and two-electron density matrices
        let one_elec_density = Self::build_electron_density_matrix(
            "one-electron-density",
            &kappa_operator,
            &single_excitation_operator,
            &cluster_operator,
            &cc_lambda_operator,
        )?;

        let two_elec_density = Self::build_electron_density_matrix(
            "two-electron-density",
            &kappa_operator,
            &double_excitation_operator,
            &cluster_operator,
            &cc_lambda_operator,
        )?;

        // Similarity-transformed Hamiltonian, e^{kappa} * H * e^{-kappa}
        let kappa_transformed_hamiltonian = MatrixAdd::new(vec![
            ExpAdjointMap::builder(kappa_operator.clone(), one_elec_matrix.clone(), Some(false))
                .left_action(true)
                .build()?,
            ExpAdjointMap::builder(kappa_operator.clone(), two_elec_matrix.clone(), Some(false))
                .left_action(true)
                .build()?,
        ])?;

        // [E_{pq}-E_{qp}, e^{kappa} * H * e^{-kappa}], equation (41),
        // J. Chem. Phys. 92, 4924-4940 (1990)
        let brillouin_equation = AdjointMap::new(
            vec![orb_rot_generator.clone()],
            kappa_transformed_hamiltonian.clone(),
            Some(true),
            Some(AdjointMode::Symmetrized),
        )?;

        let max_commutator_order = Self::max_commutator_order();

        // Response equation for coupled-cluster amplitude, equation (42),
        // J. Chem. Phys. 92, 4924-4940 (1990)
        let cc_amplitude_equation = MatrixMul::new(vec![
            HermitianTranspose::new(cc_excitation_operator.clone())?,
            ExpAdjointMap::builder(
                cluster_operator.clone(),
                kappa_transformed_hamiltonian.clone(),
                Some(true),
            )
            .left_action(false)
            .max_commutator_order(max_commutator_order)
            .build()?,
        ])?;

        // [e^{kappa} * H * e^{-kappa}, tau]
        let cc_transformed_hamiltonian = ExpAdjointMap::builder(
            cluster_operator.clone(),
            AdjointMap::new(
                vec![cc_excitation_operator.clone()],
                kappa_transformed_hamiltonian.clone(),
                Some(false),
                Some(AdjointMode::Commutative),
            )?,
            Some(true),
        )
        .left_action(false)
        .max_commutator_order(max_commutator_order)
        .build()?;

        // Response equation for coupled-cluster Lagrangian multiplier,
        // equation (43), J. Chem. Phys. 92, 4924-4940 (1990)
        let cc_multiplier_equation = MatrixAdd::new(vec![
            cc_transformed_hamiltonian.clone(),
            MatrixMul::new(vec![cc_lambda_operator.clone(), cc_transformed_hamiltonian])?,
        ])?;

        // Derivative of e^{kappa} * H * e^{-kappa} with respect to kappa,
        // which can be written as e^{\tilde{kappa}} [E_{pq}-E_{qp}, H],
        // but which is not a regular exponential adjoint map, or its expansion
        // should use symmetrized nested commutators
        let diff_kappa_transformed_hamiltonian = ExpAdjointMap::builder(
            kappa_operator.clone(),
            AdjointMap::new(
                vec![orb_rot_generator.clone()],
                MatrixAdd::new(vec![one_elec_matrix.clone(), two_elec_matrix.clone()])?,
                Some(true),
                Some(AdjointMode::Symmetrized),
            )?,
            Some(false),
        )
        .left_action(true)
        .build()?;

        // e^{kappa} * (e^{\tilde{kappa}} [E_{pq}-E_{qp}, H]) * e^{-kappa}
        let cc_transformed_diff_hamiltonian = ExpAdjointMap::builder(
            cluster_operator.clone(),
            diff_kappa_transformed_hamiltonian.clone(),
            Some(true),
        )
        .left_action(false)
        .max_commutator_order(max_commutator_order)
        .build()?;

        // Response equation for Brillouin condition multiplier, equation (44),
        // J. Chem. Phys. 92, 4924-4940 (1990)
        let brillouin_multiplier_equation = MatrixAdd::new(vec![
            cc_transformed_diff_hamiltonian.clone(),
            MatrixMul::new(vec![cc_lambda_operator.clone(), cc_transformed_diff_hamiltonian])?,
            DotProduct::new(
                AdjointMap::new(
                    vec![orb_rot_generator],
                    diff_kappa_transformed_hamiltonian,
                    Some(true),
                    Some(AdjointMode::Symmetrized),
                )?,
                false,
                brillouin_multiplier.clone(),
                false,
                Some(false),
            )?,
        ])?;

        // Sets up the Lagrangian
        let lagrangian_expr = Add::new(vec![
            Trace::new(MatrixMul::new(vec![one_elec_density.clone(), one_elec_matrix.clone()])?)?,
            Trace::new(MatrixMul::new(vec![two_elec_density.clone(), two_elec_matrix.clone()])?)?,
            DotProduct::new(
                brillouin_equation.clone(),
                false,
                brillouin_multiplier.clone(),
                false,
                Some(true),
            )?,
        ])?;

        Ok(Self {
            one_elec_matrix,
            two_elec_matrix,
            cc_amplitude,
            cluster_operator,
            cc_lambda_operator,
            cc_amplitude_equation,
            cc_multiplier,
            cc_multiplier_equation,
            orb_rot_parameter,
            kappa_operator,
            brillouin_equation,
            brillouin_multiplier,
            brillouin_multiplier_equation,
            one_elec_density,
            two_elec_density,
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
    // `self.cc_amplitude` or `self.multipliers`, otherwise the result will be
    // incorrect.
    // J. Chem. Phys. 92, 4924-4940 (Apr. 1990)
    // notes: equations (10)-(13)
    #[inline]
    pub fn linear_response_rhs(
        &self,
        rsp_parameter: &Arc<dyn Expr>,
        num_tol: Option<NumberTolerance>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        let (equation, derivative, sign) = if let Some(parameter) =
            downcast_from_arc::<WfnParameter>(rsp_parameter)
        {
            if parameter.match_one_self(&self.cc_amplitude, true) {
                (&self.cc_amplitude_equation, parameter.derivative(), 1)
            } else if parameter.match_one_self(&self.orb_rot_parameter, true) {
                (&self.brillouin_equation, parameter.derivative(), -1)
            } else {
                return Err(expression_error(
                    "Invalid wave function parameter",
                    rsp_parameter,
                    None,
                ));
            }
        } else if let Some(multiplier) = downcast_from_arc::<LagMultiplier>(rsp_parameter) {
            if multiplier.match_one_self(&self.cc_multiplier, true) {
                (&self.cc_multiplier_equation, multiplier.derivative(), -1)
            } else if multiplier.match_one_self(&self.brillouin_multiplier, true) {
                (&self.brillouin_multiplier_equation, multiplier.derivative(), -1)
            } else {
                return Err(expression_error("Invalid Lagrangian multiplier", rsp_parameter, None));
            }
        } else {
            return Err(expression_error(
                "Invalid type of response parameter",
                rsp_parameter,
                None,
            ));
        };

        let rhs = differentiate_expr(equation, derivative)?
            .substitute_zero_perturbations(num_tol)?
            .remove_one(rsp_parameter)?;

        if sign < 0 {
            MatrixMul::new(vec![Number::minus_one(), rhs])
        } else {
            Ok(rhs)
        }
    }

    #[inline]
    pub fn cluster_operator(&self) -> &Arc<dyn Expr> {
        &self.cluster_operator
    }

    #[inline]
    pub fn cc_lambda_operator(&self) -> &Arc<dyn Expr> {
        &self.cc_lambda_operator
    }

    #[inline]
    pub fn kappa_operator(&self) -> &Arc<dyn Expr> {
        &self.kappa_operator
    }

    #[inline]
    pub fn brillouin_equation(&self) -> &Arc<dyn Expr> {
        &self.brillouin_equation
    }

    #[inline]
    pub fn one_elec_density(&self) -> &Arc<dyn Expr> {
        &self.one_elec_density
    }

    #[inline]
    pub fn two_elec_density(&self) -> &Arc<dyn Expr> {
        &self.two_elec_density
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
            self.orb_rot_parameter.clone(),
            exten_perturbations,
            min_wfn_order,
        )?;
        result.eliminate(self.cc_amplitude.clone(), exten_perturbations, min_wfn_order)
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
            self.brillouin_multiplier.clone(),
            exten_perturbations,
            min_multiplier_order,
        )?;
        result.eliminate(self.cc_multiplier.clone(), exten_perturbations, min_multiplier_order)
    }
}

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
    fn get_wfn_parameters(&self) -> Vec<Arc<dyn Expr>> {
        vec![self.cc_amplitude.clone(), self.orb_rot_parameter.clone()]
    }

    #[inline]
    fn get_lagrangian_multipliers(&self) -> Vec<Arc<dyn Expr>> {
        vec![self.cc_multiplier.clone(), self.brillouin_multiplier.clone()]
    }
}
