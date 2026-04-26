use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use num_rational::Rational64;

use tinned::{
    Add, AoTwoElecEnergy, AoTwoElecMatrix, BasisTimeEvolution, ExchCorrEnergy, ExchCorrPotential,
    Expr, LagMultiplier, MatrixAdd, MatrixMul, NonElecFunction, Number, NumberTolerance,
    OneElecMatrix, Perturbation, ResidueParameter, SubExpr, TimeEvolution, TinnedError, Trace,
    WfnParameter, ZeroOperator, anticommutator, commutator, differentiate_expr, downcast_from_arc,
    expression_error, generic_error, is_expr_type, is_zero_expr, s_anticommutator, s_commutator,
    subtract_exprs,
};

use crate::lagrangian::Lagrangian;
use crate::lagrangian_internal::sealed::LagrangianInternal;
use crate::types::LinearRhsInput;

/// Controls whether DAO response functions are written in simplified and
/// (almost) symmetric forms. See Sec. IV G 3, J. Chem. Phys. 2008, 129, 214108.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SymmetrizeMode {
    /// Always symmetrize
    Always,
    /// Never symmetrize
    Never,
    /// Symmetrize only when all perturbations are extensive
    Auto,
}

impl Default for SymmetrizeMode {
    fn default() -> Self {
        Self::Auto
    }
}

/// Time-averaged quasienergy derivative Lagrangian for density-matrix
/// response theory in an atomic-orbital basis.
///
/// The Lagrangian can be built for either non-orthonormal basis sets, when an
/// overlap matrix is provided, or orthonormal basis sets, when no overlap matrix
/// is provided.
///
/// `LagrangianDao` builds and stores the symbolic ingredients needed to compute
/// DAO response functions and residues through the [`Lagrangian`] interface.
///
/// The stored expression contains:
///
/// - perturbation (usually named as perturbation `a`) used to form the derivative Lagrangian,
/// - atomic-orbital density matrix as wave function parameters,
/// - optional overlap matrix,
/// - differentiated generalized energy with respect to perturbation `a`,
/// - generalized Fock matrix,
/// - optional generalized energy-weighted density matrix when the overlap matrix is present,
/// - time-dependent self-consistent-field (TDSCF) equation and idempotency
///   constraints, as well as their Lagrangian multipliers,
/// - symmetrization mode,
/// - the full time-averaged quasienergy derivative Lagrangian.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LagrangianDao {
    perturbation_a: Arc<Perturbation>,
    density_matrix: Arc<dyn Expr>,
    overlap_matrix: Option<Arc<dyn Expr>>,
    fock_matrix: Arc<dyn Expr>,
    generalized_energy_a: Arc<dyn Expr>,
    general_ew_density: Option<Arc<dyn Expr>>,
    tdscf_multiplier_proxy: Arc<dyn Expr>,
    tdscf_multiplier: Arc<dyn Expr>,
    tdscf_equation: Arc<dyn Expr>,
    idemp_multiplier_proxy: Arc<dyn Expr>,
    idemp_multiplier: Arc<dyn Expr>,
    idempotency: Arc<dyn Expr>,
    symmetrize_mode: SymmetrizeMode,
    lagrangian_expr: Arc<dyn Expr>,
}

impl LagrangianDao {
    /// Builds a time-averaged quasienergy derivative Lagrangian by following
    /// equation (11), J. Phys. Chem. A 2025, 129, 3709-3721.
    ///
    /// If `overlap_matrix` is `Some`, the Lagrangian is built for a
    /// non-orthonormal basis. If `overlap_matrix` is `None`, the Lagrangian is
    /// built for an orthonormal basis.
    ///
    /// # Arguments
    ///
    /// * `perturbation_a` - Perturbation used to form the derivative
    ///   Lagrangian.
    /// * `density_matrix` - Atomic-orbital density matrix. It must be a
    ///   `tinned::WfnParameter`.
    /// * `overlap_matrix` - Optional overlap matrix. If present, it must be a
    ///   `tinned::OneElecMatrix`.
    /// * `one_elec_operators` - One-electron matrix contributions. Each entry
    ///   must be either a `tinned::OneElecMatrix` or a
    ///   `tinned::BasisTimeEvolution`.
    /// * `two_elec_operator` - Optional two-electron matrix contribution. If
    ///   present, it must be a `tinned::AoTwoElecMatrix`.
    /// * `xc_energy` - Optional exchange-correlation energy contribution. If
    ///   present, it must be a `tinned::ExchCorrEnergy`.
    /// * `xc_potential` - Optional exchange-correlation potential contribution.
    ///   If present, it must be a `tinned::ExchCorrPotential`.
    /// * `h_nuc` - Optional nuclear contribution. If present, it must be a
    ///   `tinned::NonElecFunction`.
    /// * `symmetrize_mode` - Optional symmetrization mode. If omitted,
    ///   `SymmetrizeMode::Auto` is used.
    /// * `num_tol` - Optional numerical tolerance used to determine whether a
    ///   `tinned::Number` should be treated as zero.
    ///
    /// # Errors
    ///
    /// Returns an error if any supplied expression has an unsupported type, or
    /// if symbolic construction of the generalized energy, Fock matrix,
    /// constraints, multipliers, or final Lagrangian fails.
    pub fn new(
        perturbation_a: Arc<Perturbation>,
        density_matrix: Arc<dyn Expr>,
        overlap_matrix: Option<Arc<dyn Expr>>,
        one_elec_operators: &[Arc<dyn Expr>],
        two_elec_operator: Option<Arc<dyn Expr>>,
        xc_energy: Option<Arc<dyn Expr>>,
        xc_potential: Option<Arc<dyn Expr>>,
        h_nuc: Option<Arc<dyn Expr>>,
        symmetrize_mode: Option<SymmetrizeMode>,
        num_tol: Option<NumberTolerance>,
    ) -> Result<Self, TinnedError> {
        let (generalized_energy, fock_matrix) = Self::build_energy_and_fock(
            &density_matrix,
            one_elec_operators,
            &two_elec_operator,
            &xc_energy,
            &xc_potential,
            &h_nuc,
        )?;

        // Constraints and Lagrangian terms
        let mut lag_terms = Vec::with_capacity(3);
        let (
            general_ew_density,
            tdscf_multiplier_expr,
            tdscf_equation_expr,
            idemp_multiplier_expr,
            idempotency_expr,
        ) = match &overlap_matrix {
            Some(overlap) => {
                let (
                    general_ew_density,
                    pulay_term,
                    tdscf_multiplier_expr,
                    tdscf_equation_expr,
                    idemp_multiplier_expr,
                    idempotency_expr,
                ) = Self::build_constraints_with_overlap(
                    perturbation_a.clone(),
                    &density_matrix,
                    overlap,
                    &fock_matrix,
                    num_tol,
                )?;

                if let Some(expr) = pulay_term {
                    lag_terms.push(expr);
                }

                (
                    general_ew_density,
                    tdscf_multiplier_expr,
                    tdscf_equation_expr,
                    idemp_multiplier_expr,
                    idempotency_expr,
                )
            },
            None => {
                let (
                    tdscf_multiplier_expr,
                    tdscf_equation_expr,
                    idemp_multiplier_expr,
                    idempotency_expr,
                ) = Self::build_lagrangian_constraints(
                    perturbation_a.clone(),
                    &density_matrix,
                    &fock_matrix,
                )?;

                (
                    None,
                    tdscf_multiplier_expr,
                    tdscf_equation_expr,
                    idemp_multiplier_expr,
                    idempotency_expr,
                )
            },
        };

        let tdscf_multiplier_name = "tdscf-multiplier";
        let tdscf_multiplier = SubExpr::new(tdscf_multiplier_name, tdscf_multiplier_expr);
        let tdscf_equation = SubExpr::new("tdscf-equation", tdscf_equation_expr);

        let idemp_multiplier_name = "idempotency-multiplier";
        let idemp_multiplier = SubExpr::new(idemp_multiplier_name, idemp_multiplier_expr);
        let idempotency = SubExpr::new("idempotency-constraint", idempotency_expr);

        // Make proxies of Lagrangian multipliers for elimination
        let tdscf_multiplier_proxy = LagMultiplier::builder(tdscf_multiplier_name).build()?;
        let idemp_multiplier_proxy = LagMultiplier::builder(idemp_multiplier_name).build()?;

        lag_terms.push(Trace::new(MatrixMul::new(vec![
            tdscf_multiplier_proxy.clone(),
            tdscf_equation.clone(),
        ])?)?);
        lag_terms.push(Trace::new(MatrixMul::new(vec![
            idemp_multiplier_proxy.clone(),
            subtract_exprs(idempotency.clone(), density_matrix.clone())?,
        ])?)?);

        // The first term in Equation (98), J. Chem. Phys. 129, 214108 (2008)
        let density_a = density_matrix.differentiate(perturbation_a.clone())?;
        let generalized_energy_a = SubExpr::new(
            "generalized-energy-a",
            generalized_energy.differentiate(perturbation_a.clone())?.remove_one(&density_a)?,
        );

        // The time-averaged quasienergy derivative Lagrangian
        let lagrangian_expr = subtract_exprs(generalized_energy_a.clone(), Add::new(lag_terms)?)?;

        Ok(Self {
            perturbation_a,
            density_matrix,
            overlap_matrix,
            fock_matrix,
            generalized_energy_a,
            general_ew_density,
            tdscf_multiplier_proxy,
            tdscf_multiplier,
            tdscf_equation,
            idemp_multiplier_proxy,
            idemp_multiplier,
            idempotency,
            symmetrize_mode: symmetrize_mode.unwrap_or_default(),
            lagrangian_expr,
        })
    }

    // Build generalized energy and Fock matrix
    #[inline]
    fn build_energy_and_fock(
        density_matrix: &Arc<dyn Expr>,
        one_elec_operators: &[Arc<dyn Expr>],
        two_elec_operator: &Option<Arc<dyn Expr>>,
        xc_energy: &Option<Arc<dyn Expr>>,
        xc_potential: &Option<Arc<dyn Expr>>,
        h_nuc: &Option<Arc<dyn Expr>>,
    ) -> Result<(Arc<dyn Expr>, Arc<dyn Expr>), TinnedError> {
        if !is_expr_type::<WfnParameter>(density_matrix) {
            return Err(expression_error("Invalid type of density matrix", density_matrix, None));
        }

        let mut energy_terms = Vec::with_capacity(one_elec_operators.len() + 3);
        let mut fock_terms = Vec::with_capacity(one_elec_operators.len() + 2);

        for op in one_elec_operators {
            if !is_expr_type::<OneElecMatrix>(op) && !is_expr_type::<BasisTimeEvolution>(op) {
                return Err(expression_error("Invalid type of one-electron matrix", op, None));
            }

            fock_terms.push(op.clone());

            let matmul = MatrixMul::new(vec![op.clone(), density_matrix.clone()])?;
            let trace = Trace::new(matmul)?;
            energy_terms.push(trace);
        }

        if let Some(op) = two_elec_operator {
            if let Some(two_elec_op) = downcast_from_arc::<AoTwoElecMatrix>(op) {
                energy_terms.push(AoTwoElecEnergy::builder_from_operator(two_elec_op).build()?);
            } else {
                return Err(expression_error("Invalid two-electron matrix", op, None));
            }
            fock_terms.push(op.clone());
        }

        if let Some(energy) = xc_energy {
            if !is_expr_type::<ExchCorrEnergy>(energy) {
                return Err(expression_error(
                    "Invalid type of exchange-correlation energy functional",
                    energy,
                    None,
                ));
            }

            energy_terms.push(energy.clone());
        }
        if let Some(op) = xc_potential {
            if !is_expr_type::<ExchCorrPotential>(op) {
                return Err(expression_error(
                    "Invalid type of exchange-correlation functional derivative matrix",
                    op,
                    None,
                ));
            }

            fock_terms.push(op.clone());
        }
        if let Some(energy) = h_nuc {
            if !is_expr_type::<NonElecFunction>(energy) {
                return Err(expression_error(
                    "Invalid type of nuclear contributions",
                    energy,
                    None,
                ));
            }

            energy_terms.push(energy.clone());
        }

        let generalized_energy = Add::new(energy_terms)?;

        let fock_expr = MatrixAdd::new(fock_terms)?;
        let fock_matrix = SubExpr::new("generalized-fock-matrix", fock_expr);

        Ok((generalized_energy, fock_matrix))
    }

    // Apply the time-evolution operator i*\frac{\partial}{\partial t} to a given expression
    #[inline]
    fn apply_time_evolution(expr: &Arc<dyn Expr>) -> Result<Arc<dyn Expr>, TinnedError> {
        TimeEvolution::builder(expr.clone()).is_forward(true).build()
    }

    // Build generalized energy-weighted density matrix, Pulay term, TDSCF
    // equation and idempotency constraints with corresponding multipliers
    fn build_constraints_with_overlap(
        perturbation_a: Arc<Perturbation>,
        density_matrix: &Arc<dyn Expr>,
        overlap_matrix: &Arc<dyn Expr>,
        fock_matrix: &Arc<dyn Expr>,
        num_tol: Option<NumberTolerance>,
    ) -> Result<
        (
            Option<Arc<dyn Expr>>,
            Option<Arc<dyn Expr>>,
            Arc<dyn Expr>,
            Arc<dyn Expr>,
            Arc<dyn Expr>,
            Arc<dyn Expr>,
        ),
        TinnedError,
    > {
        if !is_expr_type::<OneElecMatrix>(overlap_matrix) {
            return Err(expression_error("Invalid type of overlap matrix", overlap_matrix, None));
        }

        let minus_one = Number::minus_one();
        let one_half = Number::one_half();
        let negative_one_half = Number::from_rational(Rational64::new(-1, 2));

        // Equation (220), J. Chem. Phys. 129, 214108 (2008)
        let density_a = density_matrix.differentiate(perturbation_a.clone())?;
        let tdscf_multiplier_expr =
            s_commutator(density_a.clone(), density_matrix.clone(), overlap_matrix.clone())?;

        // Equation (229), J. Chem. Phys. 129, 214108 (2008)
        let density_t = Self::apply_time_evolution(density_matrix)?;
        let overlap_t = Self::apply_time_evolution(overlap_matrix)?;
        let tdscf_equation_expr = MatrixAdd::new(vec![
            MatrixMul::new(vec![
                fock_matrix.clone(),
                density_matrix.clone(),
                overlap_matrix.clone(),
            ])?,
            MatrixMul::new(vec![
                minus_one.clone(),
                overlap_matrix.clone(),
                density_matrix.clone(),
                fock_matrix.clone(),
            ])?,
            MatrixMul::new(vec![
                minus_one.clone(),
                overlap_matrix.clone(),
                density_t.clone(),
                overlap_matrix.clone(),
            ])?,
            MatrixMul::new(vec![
                negative_one_half.clone(),
                overlap_t.clone(),
                density_matrix.clone(),
                overlap_matrix.clone(),
            ])?,
            MatrixMul::new(vec![
                negative_one_half.clone(),
                overlap_matrix.clone(),
                density_matrix.clone(),
                overlap_t.clone(),
            ])?,
        ])?;

        // Equation (224), J. Chem. Phys. 129, 214108 (2008)
        let overlap_a = overlap_matrix.differentiate(perturbation_a.clone())?;
        let fock_a = fock_matrix.differentiate(perturbation_a)?;
        let (general_ew_density, pulay_term, idemp_multiplier_expr) =
            if is_zero_expr(&overlap_a, num_tol) {
                (
                    None,
                    None,
                    MatrixAdd::new(vec![
                        MatrixMul::new(vec![
                            fock_a.clone(),
                            density_matrix.clone(),
                            overlap_matrix.clone(),
                        ])?,
                        MatrixMul::new(vec![
                            overlap_matrix.clone(),
                            density_matrix.clone(),
                            fock_a.clone(),
                        ])?,
                        MatrixMul::new(vec![minus_one.clone(), fock_a])?,
                    ])?,
                )
            } else {
                // Equation (95), J. Chem. Phys. 129, 214108 (2008)
                let general_ew_density_expr = MatrixAdd::new(vec![
                    MatrixMul::new(vec![
                        density_matrix.clone(),
                        fock_matrix.clone(),
                        density_matrix.clone(),
                    ])?,
                    MatrixMul::new(vec![
                        one_half.clone(),
                        density_t.clone(),
                        overlap_matrix.clone(),
                        density_matrix.clone(),
                    ])?,
                    MatrixMul::new(vec![
                        negative_one_half.clone(),
                        density_matrix.clone(),
                        overlap_matrix.clone(),
                        density_t.clone(),
                    ])?,
                ])?;
                let general_ew_density = SubExpr::new(
                    "generalized-energy-weighted-density-matrix",
                    general_ew_density_expr,
                );
                // Pulay term
                let pulay_term = Trace::new(MatrixMul::new(vec![
                    overlap_a.clone(),
                    general_ew_density.clone(),
                ])?)?;
                let idemp_multiplier_expr = MatrixAdd::new(vec![
                    MatrixMul::new(vec![
                        fock_a.clone(),
                        density_matrix.clone(),
                        overlap_matrix.clone(),
                    ])?,
                    MatrixMul::new(vec![
                        minus_one.clone(),
                        fock_matrix.clone(),
                        density_matrix.clone(),
                        overlap_a.clone(),
                    ])?,
                    MatrixMul::new(vec![
                        one_half.clone(),
                        overlap_t.clone(),
                        density_matrix.clone(),
                        overlap_a.clone(),
                    ])?,
                    MatrixMul::new(vec![
                        overlap_matrix.clone(),
                        density_t.clone(),
                        overlap_a.clone(),
                    ])?,
                    MatrixMul::new(vec![
                        overlap_matrix.clone(),
                        density_matrix.clone(),
                        fock_a.clone(),
                    ])?,
                    MatrixMul::new(vec![
                        minus_one.clone(),
                        overlap_a.clone(),
                        density_matrix.clone(),
                        fock_matrix.clone(),
                    ])?,
                    MatrixMul::new(vec![
                        negative_one_half,
                        overlap_a.clone(),
                        density_matrix.clone(),
                        overlap_t,
                    ])?,
                    MatrixMul::new(vec![
                        minus_one.clone(),
                        overlap_a,
                        density_t,
                        overlap_matrix.clone(),
                    ])?,
                    MatrixMul::new(vec![minus_one, fock_a])?,
                ])?;
                (Some(general_ew_density), Some(pulay_term), idemp_multiplier_expr)
            };

        // First term of Equation (230), J. Chem. Phys. 129, 214108 (2008)
        let idempotency_expr = MatrixMul::new(vec![
            density_matrix.clone(),
            overlap_matrix.clone(),
            density_matrix.clone(),
        ])?;

        Ok((
            general_ew_density,
            pulay_term,
            tdscf_multiplier_expr,
            tdscf_equation_expr,
            idemp_multiplier_expr,
            idempotency_expr,
        ))
    }

    // Build TDSCF equation and idempotency constraints with corresponding
    // multipliers for orthonormal basis sets
    fn build_lagrangian_constraints(
        perturbation_a: Arc<Perturbation>,
        density_matrix: &Arc<dyn Expr>,
        fock_matrix: &Arc<dyn Expr>,
    ) -> Result<(Arc<dyn Expr>, Arc<dyn Expr>, Arc<dyn Expr>, Arc<dyn Expr>), TinnedError> {
        // D^{a}D-DD^{a}
        let density_a = density_matrix.differentiate(perturbation_a.clone())?;
        let tdscf_multiplier_expr = commutator(density_a.clone(), density_matrix.clone())?;
        // Y = FD-DF-i\frac{\partial D}{\partial t}
        let density_t = Self::apply_time_evolution(density_matrix)?;
        let tdscf_equation_expr =
            subtract_exprs(commutator(fock_matrix.clone(), density_matrix.clone())?, density_t)?;
        // F^{a}D+DF^{a}-F^{a}
        let fock_a = fock_matrix.differentiate(perturbation_a)?;
        let idemp_multiplier_expr =
            subtract_exprs(anticommutator(fock_a.clone(), density_matrix.clone())?, fock_a)?;
        // D*D
        let idempotency_expr =
            MatrixMul::new(vec![density_matrix.clone(), density_matrix.clone()])?;

        Ok((tdscf_multiplier_expr, tdscf_equation_expr, idemp_multiplier_expr, idempotency_expr))
    }

    /// Returns the particular solution of a perturbed density matrix.
    ///
    /// `freq_pert_density` must be either a differentiated
    /// `tinned::WfnParameter` derived from this Lagrangian's density matrix, or
    /// a `tinned::ResidueParameter` whose inner parameter is such a
    /// `tinned::WfnParameter`.
    ///
    /// For a `tinned::WfnParameter`, the particular solution is constructed
    /// from the differentiated idempotency constraint, see squations (14) and
    /// (16), J. Phys. Chem. A 2025, 129, 3709-3721.
    ///
    /// For a `tinned::ResidueParameter`, this method first checks the wrapped
    /// density parameter. If the wrapped parameter's derivative is equivalent
    /// to the residue perturbations, the particular solution is zero. If the
    /// wrapped parameter's derivative is a superchain of the residue
    /// perturbations, only terms containing the wrapped parameter or its
    /// higher-order derivatives are retained, and those terms are replaced by
    /// the corresponding residue density matrices.
    ///
    /// # Errors
    ///
    /// Returns an error if `freq_pert_density` has an unsupported type, if a
    /// residue parameter wraps an unsupported inner parameter, if
    /// `freq_pert_density` is not derived from this Lagrangian's atomic-orbital
    /// density matrix, or if symbolic construction fails.
    //
    //FIXME: add unit test for this function
    pub fn particular_density_solution(
        &self,
        freq_pert_density: &Arc<dyn Expr>,
        num_tol: Option<NumberTolerance>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        let rhs_input = if let Some(wfn_param) =
            downcast_from_arc::<WfnParameter>(freq_pert_density)
        {
            if !wfn_param.match_one_self(&self.density_matrix, true) {
                return Err(expression_error(
                    format!(
                        "Response parameter is not derived from this Lagrangian's density matrix {}",
                        &self.density_matrix
                    ),
                    freq_pert_density,
                    None,
                ));
            }

            LinearRhsInput {
                equation: &self.idempotency,
                derivative: wfn_param.derivative(),
                diff_parameter: freq_pert_density,
                residue_info: None,
            }
        } else if let Some(res_param) = downcast_from_arc::<ResidueParameter>(freq_pert_density) {
            let diff_parameter = res_param.parameter();

            let wfn = downcast_from_arc::<WfnParameter>(diff_parameter).ok_or_else(|| {
                expression_error(
                    "Invalid parameter type of residue density matrix",
                    freq_pert_density,
                    None,
                )
            })?;

            if !wfn.match_one_self(&self.density_matrix, true) {
                return Err(expression_error(
                    format!(
                        "Residue response parameter is not derived from this Lagrangian's density matrix {}",
                        &self.density_matrix
                    ),
                    freq_pert_density,
                    None,
                ));
            }

            // `ResidueParameter` ensures that `res_param.perturbations()`
            // is a subchain of `wfn.derivative()`, so we check if the
            // former is also a superchain of the latter.
            if wfn.derivative().is_superchain_vec(res_param.perturbations()) {
                return Ok(ZeroOperator::new());
            }

            LinearRhsInput {
                equation: &self.idempotency,
                derivative: wfn.derivative(),
                diff_parameter,
                residue_info: Some((res_param, self.density_matrix.clone())),
            }
        } else {
            return Err(expression_error(
                "Invalid type of density matrix",
                freq_pert_density,
                None,
            ));
        };

        let idemp_deriv = self.build_linear_rhs(rhs_input, num_tol)?;

        let anticomm_idemp_dm = if let Some(overlap) = &self.overlap_matrix {
            s_anticommutator(idemp_deriv.clone(), self.density_matrix.clone(), overlap.clone())?
        } else {
            anticommutator(idemp_deriv.clone(), self.density_matrix.clone())?
        };

        subtract_exprs(anticomm_idemp_dm, idemp_deriv)
    }

    /// Builds the right-hand side of a linear response equation for a perturbed
    /// density matrix.
    ///
    /// `freq_pert_density` must be either a differentiated
    /// `tinned::WfnParameter` derived from this Lagrangian's density matrix,
    /// or a `tinned::ResidueParameter` whose inner parameter is such a
    /// `tinned::WfnParameter`.
    ///
    /// `particular_solution` is the particular solution returned by
    /// `Self::particular_density_solution` for the same perturbed density
    /// matrix.
    ///
    /// For a `tinned::WfnParameter`, this method differentiates the TDSCF
    /// equation and replaces the perturbed density matrix by
    /// `particular_solution`, see equation (19), J. Phys. Chem. A 2025, 129,
    /// 3709-3721.
    ///
    /// For a `tinned::ResidueParameter`, this method checks the wrapped
    /// density parameter. If the wrapped parameter's derivative is equivalent
    /// to the residue perturbations, the same construction is used as for a
    /// `tinned::WfnParameter`.
    ///
    /// If the wrapped parameter's derivative is a superchain of the residue
    /// perturbations, only terms containing the wrapped parameter or its
    /// higher-order derivatives are retained, and those terms are replaced by
    /// the corresponding residue density matrices. Note that
    /// `particular_solution` should not be removed or replaced, and it is
    /// actually the particular solution for the higher-order residue density
    /// matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if `freq_pert_density` has an unsupported type, if a
    /// residue parameter wraps an unsupported inner parameter, if
    /// `freq_pert_density` is not derived from this Lagrangian's atomic-orbital
    /// density matrix, or if symbolic differentiation, substitution,
    /// retention, replacement, or construction fails.
    pub fn linear_response_rhs(
        &self,
        freq_pert_density: &Arc<dyn Expr>,
        particular_solution: Arc<dyn Expr>,
        num_tol: Option<NumberTolerance>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        let (tdscf_deriv, replacement_target) = if let Some(wfn_param) =
            downcast_from_arc::<WfnParameter>(freq_pert_density)
        {
            if !wfn_param.match_one_self(&self.density_matrix, true) {
                return Err(expression_error(
                    format!(
                        "Response parameter is not derived from this Lagrangian's density matrix {}",
                        &self.density_matrix
                    ),
                    freq_pert_density,
                    None,
                ));
            }

            (
                differentiate_expr(&self.tdscf_equation, wfn_param.derivative())?
                    .substitute_zero_perturbations(num_tol)?,
                freq_pert_density,
            )
        } else if let Some(res_param) = downcast_from_arc::<ResidueParameter>(freq_pert_density) {
            let diff_parameter = res_param.parameter();

            let wfn = downcast_from_arc::<WfnParameter>(diff_parameter).ok_or_else(|| {
                expression_error(
                    "Invalid parameter type of residue density matrix",
                    freq_pert_density,
                    None,
                )
            })?;

            if !wfn.match_one_self(&self.density_matrix, true) {
                return Err(expression_error(
                    format!(
                        "Residue response parameter is not derived from this Lagrangian's density matrix {}",
                        &self.density_matrix
                    ),
                    freq_pert_density,
                    None,
                ));
            }

            // Clean `TimeEvolution` and unperturbed
            // `BasisTimeEvolution` objects first, then replace the
            // differentiated density by `particular_solution`; otherwise a
            // differentiated `TimeEvolution` (reflected by its
            // differentiated argument) will be incorrectly replaced by
            // an undifferentiated one and cleaned.
            let result = differentiate_expr(&self.tdscf_equation, wfn.derivative())?
                .substitute_zero_perturbations(num_tol)?;

            // `ResidueParameter` ensures that `res_param.perturbations()`
            // is a subchain of `wfn.derivative()`, so we check if the
            // former is also a superchain of the latter. Since we do not
            // replace the sum of frequencies of perturbations by the
            // excitation energy, nothing is different for the right-hand
            // side of the residue.
            if wfn.derivative().is_superchain_vec(res_param.perturbations()) {
                (result, diff_parameter)
            } else {
                let residue_relations = HashMap::from([(
                    res_param.excited_state().clone(),
                    (res_param.positive_frequency(), res_param.perturbations().to_vec()),
                )]);

                let residue_setup = self
                    .build_residue_parameters(&[self.density_matrix.clone()], &residue_relations)?;

                (
                    result
                        .retain_all(residue_setup.diff_params(), true)?
                        .replace_all(residue_setup.residue_map(), true)?,
                    freq_pert_density,
                )
            }
        } else {
            return Err(expression_error(
                "Invalid type of density matrix",
                freq_pert_density,
                None,
            ));
        };

        tdscf_deriv.replace_one(replacement_target, particular_solution, false)
    }

    /// Returns the perturbation `a` used to build the derivative Lagrangian.
    #[inline]
    pub fn perturbation_a(&self) -> &Arc<Perturbation> {
        &self.perturbation_a
    }

    /// Returns the optional overlap matrix.
    #[inline]
    pub fn overlap_matrix(&self) -> Option<&Arc<dyn Expr>> {
        self.overlap_matrix.as_ref()
    }

    /// Returns the generalized Fock matrix.
    #[inline]
    pub fn fock_matrix(&self) -> &Arc<dyn Expr> {
        &self.fock_matrix
    }

    /// Returns the differentiated generalized energy.
    #[inline]
    pub fn generalized_energy_a(&self) -> &Arc<dyn Expr> {
        &self.generalized_energy_a
    }

    /// Returns the optional generalized energy-weighted density matrix.
    #[inline]
    pub fn general_ew_density(&self) -> Option<&Arc<dyn Expr>> {
        self.general_ew_density.as_ref()
    }

    /// Returns the TDSCF multiplier expression.
    #[inline]
    pub fn tdscf_multiplier(&self) -> &Arc<dyn Expr> {
        &self.tdscf_multiplier
    }

    /// Returns the TDSCF equation expression.
    #[inline]
    pub fn tdscf_equation(&self) -> &Arc<dyn Expr> {
        &self.tdscf_equation
    }

    /// Returns the idempotency multiplier expression.
    #[inline]
    pub fn idemp_multiplier(&self) -> &Arc<dyn Expr> {
        &self.idemp_multiplier
    }

    /// Returns the idempotency constraint expression.
    #[inline]
    pub fn idempotency(&self) -> &Arc<dyn Expr> {
        &self.idempotency
    }

    /// Returns the symmetrization mode.
    #[inline]
    pub fn symmetrize_mode(&self) -> SymmetrizeMode {
        self.symmetrize_mode
    }
}

impl LagrangianInternal for LagrangianDao {
    #[inline]
    fn get_extra_perturbations(&self) -> Vec<Arc<Perturbation>> {
        vec![self.perturbation_a.clone()]
    }

    //FIXME: Refer to some equations in our manuscript later
    fn post_differentiation(
        &self,
        lagrangian: &Arc<dyn Expr>,
        exten_perturbations: &[Arc<Perturbation>],
        inten_perturbations: &[Arc<Perturbation>],
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        let do_symmetrize: bool = match self.symmetrize_mode {
            SymmetrizeMode::Always => true,
            SymmetrizeMode::Never => false,
            // We perform symmetrization when there is no intensive perturbations
            SymmetrizeMode::Auto => inten_perturbations.is_empty(),
        };

        if !do_symmetrize {
            return Ok(lagrangian.clone());
        }

        let fock_deriv =
            self.do_differentiation(&self.fock_matrix, exten_perturbations, inten_perturbations)?;
        let mut max_fs_derivs = HashSet::from([fock_deriv.clone()]);

        let dens_deriv = self.density_matrix.differentiate(self.perturbation_a.clone())?;
        let mut simplified_terms = vec![Trace::new(MatrixMul::new(vec![fock_deriv, dens_deriv])?)?];

        if let Some(overlap) = &self.overlap_matrix {
            let overlap_deriv =
                self.do_differentiation(&overlap, exten_perturbations, inten_perturbations)?;
            max_fs_derivs.insert(overlap_deriv.clone());
            if let Some(gew_density) = &self.general_ew_density {
                let gew_density_deriv = gew_density.differentiate(self.perturbation_a.clone())?;
                simplified_terms.push(Trace::new(MatrixMul::new(vec![
                    Number::minus_one(),
                    overlap_deriv,
                    gew_density_deriv,
                ])?)?);
            }
        }

        // Removes terms containing maximum order derivatives of Fock and overlap matrices
        simplified_terms.push(lagrangian.remove_all(&max_fs_derivs)?);

        Add::new(simplified_terms)
    }

    #[inline]
    fn eliminate_wfn_parameter(
        &self,
        lagrangian: &Arc<dyn Expr>,
        exten_perturbations: &[Arc<Perturbation>],
        min_wfn_order: u32,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        lagrangian.eliminate(self.density_matrix.clone(), exten_perturbations, min_wfn_order)
    }

    #[inline]
    fn eliminate_lag_multipliers(
        &self,
        lagrangian: &Arc<dyn Expr>,
        exten_perturbations: &[Arc<Perturbation>],
        min_multiplier_order: u32,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        let mut result = lagrangian
            .eliminate(
                self.tdscf_multiplier_proxy.clone(),
                exten_perturbations,
                min_multiplier_order,
            )
            .map_err(|e| {
                generic_error("Elimination of TDSCF multiplier failed", Some(Box::new(e)))
            })?;

        result = result
            .eliminate(
                self.idemp_multiplier_proxy.clone(),
                exten_perturbations,
                min_multiplier_order,
            )
            .map_err(|e| {
                generic_error("Elimination of idempotency multiplier failed", Some(Box::new(e)))
            })?;

        // Replace multiplier proxies with their representation including differentiated ones
        let multiplier_map: HashMap<Arc<dyn Expr>, Arc<dyn Expr>> = HashMap::from([
            (self.tdscf_multiplier_proxy.clone(), self.tdscf_multiplier.clone()),
            (self.idemp_multiplier_proxy.clone(), self.idemp_multiplier.clone()),
        ]);

        result.replace_all(&multiplier_map, true)
    }
}

impl Lagrangian for LagrangianDao {
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
        vec![self.density_matrix.clone()]
    }

    // We return an empty vector because we actually use ansatze of Lagrangian multipliers
    #[inline]
    fn get_lagrangian_multipliers(&self) -> Vec<Arc<dyn Expr>> {
        Vec::new()
    }
}
