use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use num_rational::Rational64;

use tinned::{
    Add, AoTwoElecEnergy, AoTwoElecMatrix, ExchCorrEnergy, ExchCorrPotential, Expr, LagMultiplier,
    MatrixAdd, MatrixMul, NonElecFunction, Number, NumberTolerance, OneElecMatrix, PertMultichain,
    Perturbation, ResidueParameter, SubExpr, TemporumOperator, TemporumOverlap, TinnedError, Trace,
    WfnParameter, ZeroOperator, anticommutator, commutator, differentiate_expr, downcast_from_arc,
    expression_error, generic_error, is_expr_type, is_zero_expr, s_anticommutator, s_commutator,
    subtract_exprs, sum_pert_frequencies,
};

use crate::lagrangian::Lagrangian;
use crate::lagrangian_internal::sealed::LagrangianInternal;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SymmetrizeMode {
    // (i) Always symmetrize
    Always,
    // (ii) Never symmetrize
    Never,
    // (iii) Symmetrize only when all perturbations are extensive
    Auto,
}

impl Default for SymmetrizeMode {
    fn default() -> Self {
        Self::Auto
    }
}

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
    lagrangian_expr: Arc<dyn Expr>,
    symmetrized_mode: SymmetrizeMode,
}

impl LagrangianDao {
    // Builds time-averaged quasi-energy derivative Lagrangian with
    // either non-orthonormal or orthonormal (when `overlap_matrix` is `None`)
    // basis sets.
    pub fn new(
        perturbation_a: Arc<Perturbation>,
        density_matrix: Arc<dyn Expr>,
        overlap_matrix: Option<Arc<dyn Expr>>,
        one_elec_operators: &[Arc<dyn Expr>],
        two_elec_operator: Option<Arc<dyn Expr>>,
        xc_energy: Option<Arc<dyn Expr>>,
        xc_potential: Option<Arc<dyn Expr>>,
        h_nuc: Option<Arc<dyn Expr>>,
        symmetrized_mode: Option<SymmetrizeMode>,
        num_tol: Option<NumberTolerance>,
    ) -> Result<Self, TinnedError> {
        let (generalized_energy, fock_matrix) = Self::build_energy_and_fock(
            density_matrix.clone(),
            one_elec_operators,
            two_elec_operator,
            xc_energy,
            xc_potential,
            h_nuc,
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
                    density_matrix.clone(),
                    overlap.clone(),
                    fock_matrix.clone(),
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
                    density_matrix.clone(),
                    fock_matrix.clone(),
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
        let tdscf_multiplier =
            SubExpr::builder(tdscf_multiplier_name, tdscf_multiplier_expr).build()?;
        let tdscf_equation = SubExpr::builder("tdscf-equation", tdscf_equation_expr).build()?;

        let idemp_multiplier_name = "idempotency-multiplier";
        let idemp_multiplier =
            SubExpr::builder(idemp_multiplier_name, idemp_multiplier_expr).build()?;
        let idempotency = SubExpr::builder("idempotency-constraint", idempotency_expr).build()?;

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
        let density_a = density_matrix.differentiate(&perturbation_a)?;
        let dens_a_set: HashSet<Arc<dyn Expr>> = [density_a].into_iter().collect();
        // Note that we differentiate the energy with respect to perturbation
        // `a`. We expect that users provide us a list of reasonable
        // `one_elec_operators` for the computation of response
        // functions and residues. We therefore do not need to remove
        // undifferentiated `one_elec_operators` in the method
        // `at_zero_strength()` as that of `LagrangianCc`.
        let generalized_energy_a = SubExpr::builder(
            "generalized-energy-a",
            generalized_energy.differentiate(&perturbation_a)?.remove(&dens_a_set)?,
        )
        .build()?;

        // The time-averaged quasi-energy derivative Lagrangian
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
            lagrangian_expr,
            symmetrized_mode: symmetrized_mode.unwrap_or_default(),
        })
    }

    // Build generalized energy and Fock matrix
    #[inline]
    fn build_energy_and_fock(
        density_matrix: Arc<dyn Expr>,
        one_elec_operators: &[Arc<dyn Expr>],
        two_elec_operator: Option<Arc<dyn Expr>>,
        xc_energy: Option<Arc<dyn Expr>>,
        xc_potential: Option<Arc<dyn Expr>>,
        h_nuc: Option<Arc<dyn Expr>>,
    ) -> Result<(Arc<dyn Expr>, Arc<dyn Expr>), TinnedError> {
        if !is_expr_type::<WfnParameter>(&density_matrix) {
            return Err(expression_error("Invalid type of density matrix", &density_matrix, None));
        }

        let mut energy_terms = Vec::with_capacity(one_elec_operators.len() + 3);
        let mut fock_terms = Vec::with_capacity(one_elec_operators.len() + 2);

        for op in one_elec_operators {
            if !is_expr_type::<OneElecMatrix>(&op) && !is_expr_type::<TemporumOverlap>(&op) {
                return Err(expression_error("Invalid type of one-electron matrix", &op, None));
            }

            fock_terms.push(op.clone());

            let matmul = MatrixMul::new(vec![op.clone(), density_matrix.clone()])?;
            let trace = Trace::new(matmul)?;
            energy_terms.push(trace);
        }

        if let Some(op) = &two_elec_operator {
            if let Some(two_elec_op) = downcast_from_arc::<AoTwoElecMatrix>(op) {
                energy_terms.push(AoTwoElecEnergy::builder_from_operator(two_elec_op).build()?);
            } else {
                return Err(expression_error("Invalid two-electron matrix", op, None));
            }
            fock_terms.push(op.clone());
        }

        if let Some(energy) = &xc_energy {
            if !is_expr_type::<ExchCorrEnergy>(&energy) {
                return Err(expression_error(
                    "Invalid type of exchange-correlation energy functional",
                    &energy,
                    None,
                ));
            }

            energy_terms.push(energy.clone());
        }
        if let Some(op) = &xc_potential {
            if !is_expr_type::<ExchCorrPotential>(&op) {
                return Err(expression_error(
                    "Invalid type of exchange-correlation functional derivative matrix",
                    &op,
                    None,
                ));
            }

            fock_terms.push(op.clone());
        }
        if let Some(energy) = &h_nuc {
            if !is_expr_type::<NonElecFunction>(&energy) {
                return Err(expression_error(
                    "Invalid type of nuclear contributions",
                    &energy,
                    None,
                ));
            }

            energy_terms.push(energy.clone());
        }

        let generalized_energy = Add::new(energy_terms)?;

        let fock_expr = MatrixAdd::new(fock_terms)?;
        let fock_matrix = SubExpr::builder("generalized-fock-matrix", fock_expr).build()?;

        Ok((generalized_energy, fock_matrix))
    }

    // Apply the time-evolution operator i*\frac{\partial}{\partial t} to a given expression
    #[inline]
    fn apply_time_evolution(expr: Arc<dyn Expr>) -> Result<Arc<dyn Expr>, TinnedError> {
        TemporumOperator::builder(expr).is_forward(true).build()
    }

    // Build generalized energy-weighted density matrix, Pulay term, TDSCF
    // equation and idempotency constraints with corresponding multipliers
    #[inline]
    fn build_constraints_with_overlap(
        perturbation_a: Arc<Perturbation>,
        density_matrix: Arc<dyn Expr>,
        overlap_matrix: Arc<dyn Expr>,
        fock_matrix: Arc<dyn Expr>,
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
        if !is_expr_type::<OneElecMatrix>(&overlap_matrix) {
            return Err(expression_error("Invalid type of overlap matrix", &overlap_matrix, None));
        }

        let minus_one = Number::minus_one();
        let one_half = Number::one_half();
        let negative_one_half = Number::from_rational(Rational64::new(-1, 2));

        // Equation (220), J. Chem. Phys. 129, 214108 (2008)
        let density_a = density_matrix.differentiate(&perturbation_a)?;
        let tdscf_multiplier_expr =
            s_commutator(density_a.clone(), density_matrix.clone(), overlap_matrix.clone())?;

        // Equation (229), J. Chem. Phys. 129, 214108 (2008)
        let density_t = Self::apply_time_evolution(density_matrix.clone())?;
        let overlap_t = Self::apply_time_evolution(overlap_matrix.clone())?;
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
        let overlap_a = overlap_matrix.differentiate(&perturbation_a)?;
        let fock_a = fock_matrix.differentiate(&perturbation_a)?;
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
                let general_ew_density = SubExpr::builder(
                    "generalized-energy-weighted-density-matrix",
                    general_ew_density_expr,
                )
                .build()?;
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
    #[inline]
    fn build_lagrangian_constraints(
        perturbation_a: Arc<Perturbation>,
        density_matrix: Arc<dyn Expr>,
        fock_matrix: Arc<dyn Expr>,
    ) -> Result<(Arc<dyn Expr>, Arc<dyn Expr>, Arc<dyn Expr>, Arc<dyn Expr>), TinnedError> {
        // D^{a}D-DD^{a}
        let density_a = density_matrix.differentiate(&perturbation_a)?;
        let tdscf_multiplier_expr = commutator(density_a.clone(), density_matrix.clone())?;
        // Y = FD-DF-i\frac{\partial D}{\partial t}
        let density_t = Self::apply_time_evolution(density_matrix.clone())?;
        let tdscf_equation_expr =
            subtract_exprs(commutator(fock_matrix.clone(), density_matrix.clone())?, density_t)?;
        // F^{a}D+DF^{a}-F^{a}
        let fock_a = fock_matrix.differentiate(&perturbation_a)?;
        let idemp_multiplier_expr =
            subtract_exprs(anticommutator(fock_a.clone(), density_matrix.clone())?, fock_a)?;
        // D*D
        let idempotency_expr =
            MatrixMul::new(vec![density_matrix.clone(), density_matrix.clone()])?;

        Ok((tdscf_multiplier_expr, tdscf_equation_expr, idemp_multiplier_expr, idempotency_expr))
    }

    // Returns the particular solution of a perturbed density matrix
    // `density_freq`, which can be the type of `WfnParameter` or
    // `ResidueParameter`.
    //
    // (1) For the type `WfnParameter`, we simply follow, for example,
    //     Equations (14) and (16), J. Phys. Chem. A 2025, 129, 3709-3721.
    //
    // (2) For the type `ResidueParameter`, we need to check its field
    //     `parameter`, which must be the type WfnParameter`.
    //
    // 2a) If `parameter`'s derivative is equivalent to `perturbations` of
    //     `ResidueParameter`, we have a residue density matrix and a
    //     ZeroOperator will return because the particular solution does not
    //     contribute to the residue density matrix.
    //
    // 2b) If `parameter`'s derivative is a superchain of `perturbations`, we
    //     have a higher-order residue density matrix. We need to remove all
    //     terms not containing `parameter` or its higher-order differentiated
    //     ones, and replace retained (un)differentiated `parameter`'s with
    //     corresponding residue density matrices.
    //
    // Note that `density_freq` should be a differentiated
    // `self.density_matrix`, otherwise the result will be incorrect.
    #[inline]
    pub fn particular_density_solution(
        &self,
        density_freq: Arc<dyn Expr>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        let idemp_deriv: Arc<dyn Expr> = if let Some(wfn_param) =
            downcast_from_arc::<WfnParameter>(&density_freq)
        {
            let set: HashSet<Arc<dyn Expr>> = [density_freq.clone()].into_iter().collect();
            differentiate_expr(&self.idempotency, wfn_param.derivative())?.remove(&set)?
        } else if let Some(res_param) = downcast_from_arc::<ResidueParameter>(&density_freq) {
            if let Some(wfn) = downcast_from_arc::<WfnParameter>(res_param.parameter()) {
                // `ResidueParameter` ensures that `res_param.perturbations()`
                // is a subchain of `wfn.derivative()`, so we check if the
                // former is also a superchain of the latter.
                if wfn.derivative().is_superchain_vec(res_param.perturbations()) {
                    return Ok(ZeroOperator::new());
                }

                let set: HashSet<Arc<dyn Expr>> = [wfn.clone_expr()].into_iter().collect();
                let result =
                    differentiate_expr(&self.idempotency, wfn.derivative())?.remove(&set)?;

                let residue_info: HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)> =
                    std::iter::once((
                        res_param.excited_state().clone(),
                        (res_param.positive_frequency(), res_param.perturbations().to_vec()),
                    ))
                    .collect();
                let (residue_set, residue_map) = self
                    .build_residue_parameters(&vec![self.density_matrix.clone()], &residue_info)?;

                result.retain(&residue_set, true)?.replace(&residue_map, true)?
            } else {
                return Err(expression_error(
                    "Invalid parameter type of residue density matrix",
                    &density_freq,
                    None,
                ));
            }
        } else {
            return Err(expression_error("Invalid type of density matrix", &density_freq, None));
        };

        let anticomm_idemp_dm = if let Some(overlap) = &self.overlap_matrix {
            s_anticommutator(idemp_deriv.clone(), self.density_matrix.clone(), overlap.clone())?
        } else {
            anticommutator(idemp_deriv.clone(), self.density_matrix.clone())?
        };

        subtract_exprs(anticomm_idemp_dm, idemp_deriv)
    }

    // Returns right-hand side (RHS) of the (linear) response equation.
    // `density_freq`, which can be the type of `WfnParameter` or
    // `ResidueParameter`. `density_part` is the particular solution from the
    // method `particular_density_solution()`.
    //
    // (1) For the type `WfnParameter`, we simply follow, for example, Equation
    //     (19), J. Phys. Chem. A 2025, 129, 3709-3721.
    //
    // (2) For the type `ResidueParameter`, we need to check its field
    //     `parameter`, which must be the type WfnParameter`.
    //
    // 2a) If `parameter`'s derivative is equivalent to `perturbations` of
    //     `ResidueParameter`, we have a residue density matrix, and the
    //     procedure is the same as (1). The RHS should be interpreted as that
    //     the sum of perturbations' frequencies is equal to, or close to the
    //     excitation energy, or its negative value depending on the field
    //     `positive_frequency` of `ResidueParameter`.
    //
    // 2b) If `parameter`'s derivative is a superchain of `perturbations`, we
    //     have a higher-order residue density matrix. We need to remove all
    //     terms not containing `parameter` or its higher-order differentiated
    //     ones, and replace retained (un)differentiated `parameter`'s with
    //     corresponding residue density matrices. Note that `density_part`
    //     should not be removed or replaced, and it is actually particular
    //     solution for the higher-order residue density matrix.
    //
    // Note that `density_freq` should be a differentiated
    // `self.density_matrix`, otherwise the result will be incorrect.
    #[inline]
    pub fn linear_response_rhs(
        &self,
        density_freq: Arc<dyn Expr>,
        density_part: Arc<dyn Expr>,
        num_tol: Option<NumberTolerance>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        let tdscf_deriv: Arc<dyn Expr> = if let Some(wfn_param) =
            downcast_from_arc::<WfnParameter>(&density_freq)
        {
            differentiate_expr(&self.tdscf_equation, wfn_param.derivative())?
        } else if let Some(res_param) = downcast_from_arc::<ResidueParameter>(&density_freq) {
            if let Some(wfn) = downcast_from_arc::<WfnParameter>(res_param.parameter()) {
                // `ResidueParameter` ensures that `res_param.perturbations()`
                // is a subchain of `wfn.derivative()`, so we check if the
                // former is also a superchain of the latter. Since we do not
                // replace the sum of frequencies of perturbations by the
                // excitation energy, nothing is different for the right-hand
                // side of the residue.
                if wfn.derivative().is_superchain_vec(res_param.perturbations()) {
                    let result = differentiate_expr(&self.tdscf_equation, wfn.derivative())?;
                    let dens_map: HashMap<Arc<dyn Expr>, Arc<dyn Expr>> =
                        std::iter::once((res_param.parameter().clone(), density_part)).collect();
                    // Clean `TemporumOperator` and unperturbed
                    // `TemporumOverlap` objects first, then replace the
                    // differentiated density by `density_part`; otherwise a
                    // differentiated `TemporumOperator` (reflected by its
                    // differentiated argument) will be incorrectly replaced by
                    // an undifferentiated one and cleaned.
                    return result.apply_zero_rules(num_tol)?.replace(&dens_map, false);
                } else {
                    let result = differentiate_expr(&self.tdscf_equation, wfn.derivative())?;

                    let residue_info: HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)> =
                        std::iter::once((
                            res_param.excited_state().clone(),
                            (res_param.positive_frequency(), res_param.perturbations().to_vec()),
                        ))
                        .collect();
                    let (residue_set, residue_map) = self.build_residue_parameters(
                        &vec![self.density_matrix.clone()],
                        &residue_info,
                    )?;

                    result.retain(&residue_set, true)?.replace(&residue_map, true)?
                }
            } else {
                return Err(expression_error(
                    "Invalid parameter type of residue density matrix",
                    &density_freq,
                    None,
                ));
            }
        } else {
            return Err(expression_error("Invalid type of density matrix", &density_freq, None));
        };

        let dens_map: HashMap<Arc<dyn Expr>, Arc<dyn Expr>> =
            std::iter::once((density_freq, density_part)).collect();

        tdscf_deriv.apply_zero_rules(num_tol)?.replace(&dens_map, false)
    }

    #[inline]
    pub fn perturbation_a(&self) -> &Arc<Perturbation> {
        &self.perturbation_a
    }

    #[inline]
    pub fn overlap_matrix(&self) -> Option<&Arc<dyn Expr>> {
        self.overlap_matrix.as_ref()
    }

    #[inline]
    pub fn fock_matrix(&self) -> &Arc<dyn Expr> {
        &self.fock_matrix
    }

    #[inline]
    pub fn generalized_energy_a(&self) -> &Arc<dyn Expr> {
        &self.generalized_energy_a
    }

    #[inline]
    pub fn general_ew_density(&self) -> Option<&Arc<dyn Expr>> {
        self.general_ew_density.as_ref()
    }

    #[inline]
    pub fn tdscf_multiplier(&self) -> &Arc<dyn Expr> {
        &self.tdscf_multiplier
    }

    #[inline]
    pub fn tdscf_equation(&self) -> &Arc<dyn Expr> {
        &self.tdscf_equation
    }

    #[inline]
    pub fn idemp_multiplier(&self) -> &Arc<dyn Expr> {
        &self.idemp_multiplier
    }

    #[inline]
    pub fn idempotency(&self) -> &Arc<dyn Expr> {
        &self.idempotency
    }

    #[inline]
    pub fn symmetrized_mode(&self) -> SymmetrizeMode {
        self.symmetrized_mode
    }
}

impl LagrangianInternal for LagrangianDao {
    #[inline]
    fn is_non_zero_sum_freqs(
        &self,
        exten_perturbations: &[Arc<Perturbation>],
        inten_perturbations: &[Arc<Perturbation>],
        num_tol: Option<NumberTolerance>,
    ) -> Result<bool, TinnedError> {
        let freq_sum_ext = sum_pert_frequencies(&exten_perturbations).map_err(|e| {
            generic_error("Sum of extensive perturbations' frequencies failed", Some(Box::new(e)))
        })?;
        let freq_sum_int = sum_pert_frequencies(&inten_perturbations).map_err(|e| {
            generic_error("Sum of intensive perturbations' frequencies failed", Some(Box::new(e)))
        })?;
        // Here we need to inlcude the frequency of `perturbation_a`, which can
        // either be extensive or intensive
        let terms = vec![freq_sum_ext, freq_sum_int, self.perturbation_a.frequency().clone()];

        let total_freq = Add::new(terms).map_err(|e| {
            generic_error("Sum of all perturbations' frequencies failed", Some(Box::new(e)))
        })?;

        Ok(!is_zero_expr(&total_freq, num_tol))
    }

    #[inline]
    fn post_differentiation(
        &self,
        lagrangian: &Arc<dyn Expr>,
        exten_perturbations: &[Arc<Perturbation>],
        inten_perturbations: &[Arc<Perturbation>],
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        let do_symmetrize: bool = match self.symmetrized_mode {
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
        let mut max_fs_derivs: HashSet<Arc<dyn Expr>> = [fock_deriv.clone()].into_iter().collect();

        let dens_deriv = self.density_matrix.differentiate(&self.perturbation_a)?;
        let mut simplified_terms = vec![Trace::new(MatrixMul::new(vec![fock_deriv, dens_deriv])?)?];

        if let Some(overlap) = &self.overlap_matrix {
            let overlap_deriv =
                self.do_differentiation(&overlap, exten_perturbations, inten_perturbations)?;
            max_fs_derivs.insert(overlap_deriv.clone());
            if let Some(gew_density) = &self.general_ew_density {
                let gew_density_deriv = gew_density.differentiate(&self.perturbation_a)?;
                simplified_terms.push(Trace::new(MatrixMul::new(vec![
                    Number::minus_one(),
                    overlap_deriv,
                    gew_density_deriv,
                ])?)?);
            }
        }

        // Removes terms containing maximum order derivatives of Fock and overlap matrices
        simplified_terms.push(lagrangian.remove(&max_fs_derivs)?);

        Add::new(simplified_terms)
    }

    #[inline]
    fn validate_residue_info(
        &self,
        exten_perturbations: &[Arc<Perturbation>],
        inten_perturbations: &[Arc<Perturbation>],
        residue_info: &HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)>,
    ) -> bool {
        let mut all_perturbations: Vec<Arc<Perturbation>> = exten_perturbations.to_vec();
        all_perturbations.extend(inten_perturbations.to_vec());
        all_perturbations.push(self.perturbation_a.clone());

        let all_pert_chain = PertMultichain::from_slice(&all_perturbations);

        for (_excited_state, (_positive_frequency, perturbations)) in residue_info {
            if !all_pert_chain.is_subchain_vec(perturbations) {
                return false;
            }
        }

        true
    }

    #[inline]
    fn eliminate_wfn_parameter(
        &self,
        lagrangian: &Arc<dyn Expr>,
        exten_perturbations: &[Arc<Perturbation>],
        min_wfn_order: u32,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        lagrangian.eliminate(&self.density_matrix, exten_perturbations, min_wfn_order)
    }

    #[inline]
    fn eliminate_lag_multipliers(
        &self,
        lagrangian: &Arc<dyn Expr>,
        exten_perturbations: &[Arc<Perturbation>],
        min_multiplier_order: u32,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        let mut result = lagrangian
            .eliminate(&self.tdscf_multiplier_proxy, exten_perturbations, min_multiplier_order)
            .map_err(|e| {
                generic_error("Elimination of TDSCF multiplier failed", Some(Box::new(e)))
            })?;

        result = result
            .eliminate(&self.idemp_multiplier_proxy, exten_perturbations, min_multiplier_order)
            .map_err(|e| {
                generic_error("Elimination of idempotency multiplier failed", Some(Box::new(e)))
            })?;

        // Replace multiplier proxies with their representation including differentiated ones
        let multiplier_map: HashMap<Arc<dyn Expr>, Arc<dyn Expr>> = HashMap::from([
            (self.tdscf_multiplier_proxy.clone(), self.tdscf_multiplier.clone()),
            (self.idemp_multiplier_proxy.clone(), self.idemp_multiplier.clone()),
        ]);

        result.replace(&multiplier_map, true)
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
    fn get_wfn_parameter(&self) -> Vec<Arc<dyn Expr>> {
        vec![self.density_matrix.clone()]
    }

    // Here, we return the proxies of Lagrangian multipliers
    #[inline]
    fn get_lag_multiplier(&self) -> Vec<Arc<dyn Expr>> {
        vec![self.tdscf_multiplier_proxy.clone(), self.idemp_multiplier_proxy.clone()]
    }
}
