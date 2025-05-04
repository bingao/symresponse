use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use num_rational::Rational64;

use typetag;

use tinned::{
    Add, Expr, LagMultiplier, MatrixAdd, MatrixMul, Number, NumberTolerance, Perturbation,
    TemporumOperator, TinnedError, Trace, TwoElecEnergy, TwoElecOperator, WfnParameter,
    anticommutator, commutator, differentiate_expr, downcast_from_arc, expression_error,
    generic_error, is_expr_type, is_zero_expr, s_anticommutator, s_commutator, subtract_exprs,
    sum_pert_frequencies,
};

use crate::lagrangian::Lagrangian;
use crate::lagrangian_internal::sealed::LagrangianInternal;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LagrangianDao {
    perturbation_a: Option<Arc<Perturbation>>,
    density_matrix: Arc<dyn Expr>,
    overlap_matrix: Option<Arc<dyn Expr>>,
    //generalized_energy: Arc<dyn Expr>,
    //fock_matrix: Arc<dyn Expr>,
    //ewd_matrix: Arc<dyn Expr>,
    lambda: Arc<dyn Expr>,
    tdscf_multiplier: Arc<dyn Expr>,
    tdscf_equation: Arc<dyn Expr>,
    zeta: Arc<dyn Expr>,
    idemp_multiplier: Arc<dyn Expr>,
    idempotency: Arc<dyn Expr>,
    lagrangian_dao: Arc<dyn Expr>,
}

impl LagrangianDao {
    pub fn new(
        perturbation_a: Arc<Perturbation>,
        density_matrix: Arc<dyn Expr>,
        overlap_matrix: Option<Arc<dyn Expr>>,
        one_elec_operators: &[Arc<dyn Expr>],
        two_elec_operator: Option<Arc<dyn Expr>>,
        xc_energy: Option<Arc<dyn Expr>>,
        xc_potential: Option<Arc<dyn Expr>>,
        h_nuc: Option<Arc<dyn Expr>>,
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

        // |i\frac{\partial `D_`}{\partial t}>
        let density_t = TemporumOperator::builder(density_matrix.clone())
            .is_forward(true)
            .build()?;
        let density_a = density_matrix.differentiate(&perturbation_a)?;
        let fock_a = fock_matrix.differentiate(&perturbation_a)?;

        // Constraint and Lagrangian terms
        let mut lag_terms = Vec::with_capacity(3);

        let (lambda, tdscf_equation, zeta, idempotency) = if let Some(overlap) = &overlap_matrix {
            // |i\frac{\partial `S`}{\partial t}>
            let overlap_t = TemporumOperator::builder(overlap.clone())
                .is_forward(true)
                .build()?;
            let minus_one = Number::minus_one();
            let one_half = Number::one_half();
            let negative_one_half = Number::from_rational(Rational64::new(-1, 2));

            // Equation (220), J. Chem. Phys. 129, 214108 (2008)
            let lambda = s_commutator(density_a.clone(), density_matrix.clone(), overlap.clone())?;

            // Equation (229), J. Chem. Phys. 129, 214108 (2008)
            let tdscf_equation = MatrixAdd::new(vec![
                MatrixMul::new(vec![
                    fock_matrix.clone(),
                    density_matrix.clone(),
                    overlap.clone(),
                ])?,
                MatrixMul::new(vec![
                    minus_one.clone(),
                    overlap.clone(),
                    density_matrix.clone(),
                    fock_matrix.clone(),
                ])?,
                MatrixMul::new(vec![
                    minus_one.clone(),
                    overlap.clone(),
                    density_t.clone(),
                    overlap.clone(),
                ])?,
                MatrixMul::new(vec![
                    negative_one_half.clone(),
                    overlap_t.clone(),
                    density_matrix.clone(),
                    overlap.clone(),
                ])?,
                MatrixMul::new(vec![
                    negative_one_half.clone(),
                    overlap.clone(),
                    density_matrix.clone(),
                    overlap_t.clone(),
                ])?,
            ])?;

            // Equation (224), J. Chem. Phys. 129, 214108 (2008)
            let overlap_a = overlap.differentiate(&perturbation_a)?;
            let zeta = if is_zero_expr(&overlap_a, num_tol) {
                MatrixAdd::new(vec![
                    MatrixMul::new(vec![
                        fock_a.clone(),
                        density_matrix.clone(),
                        overlap.clone(),
                    ])?,
                    MatrixMul::new(vec![
                        overlap.clone(),
                        density_matrix.clone(),
                        fock_a.clone(),
                    ])?,
                    MatrixMul::new(vec![minus_one.clone(), fock_a])?,
                ])?
            } else {
                // Equation (95), J. Chem. Phys. 129, 214108 (2008)
                let ewd_matrix = MatrixAdd::new(vec![
                    MatrixMul::new(vec![
                        density_matrix.clone(),
                        fock_matrix.clone(),
                        density_matrix.clone(),
                    ])?,
                    MatrixMul::new(vec![
                        one_half.clone(),
                        density_t.clone(),
                        overlap.clone(),
                        density_matrix.clone(),
                    ])?,
                    MatrixMul::new(vec![
                        negative_one_half.clone(),
                        density_matrix.clone(),
                        overlap.clone(),
                        density_t.clone(),
                    ])?,
                ])?;
                // Pulay term
                lag_terms.push(Trace::new(MatrixMul::new(vec![
                    overlap_a.clone(),
                    ewd_matrix,
                ])?)?);

                MatrixAdd::new(vec![
                    MatrixMul::new(vec![
                        fock_a.clone(),
                        density_matrix.clone(),
                        overlap.clone(),
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
                    MatrixMul::new(vec![overlap.clone(), density_t.clone(), overlap_a.clone()])?,
                    MatrixMul::new(vec![
                        overlap.clone(),
                        density_matrix.clone(),
                        fock_a.clone(),
                    ])?,
                    MatrixMul::new(vec![
                        minus_one.clone(),
                        overlap_a.clone(),
                        density_matrix.clone(),
                        fock_matrix,
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
                        overlap.clone(),
                    ])?,
                    MatrixMul::new(vec![minus_one, fock_a])?,
                ])?
            };

            // First term of Equation (230), J. Chem. Phys. 129, 214108 (2008)
            let idempotency = MatrixMul::new(vec![
                density_matrix.clone(),
                overlap.clone(),
                density_matrix.clone(),
            ])?;

            (lambda, tdscf_equation, zeta, idempotency)
        } else {
            // = D^{a}D-DD^{a}
            let lambda = commutator(density_a.clone(), density_matrix.clone())?;
            // Y = FD-DF-i\frac{\partial D}{\partial t}
            let tdscf_equation = subtract_exprs(
                commutator(fock_matrix.clone(), density_matrix.clone())?,
                density_t,
            )?;
            // = F^{a}D+DF^{a}-F^{a}
            let zeta = subtract_exprs(
                anticommutator(fock_a.clone(), density_matrix.clone())?,
                fock_a,
            )?;
            // = D*D
            let idempotency = MatrixMul::new(vec![density_matrix.clone(), density_matrix.clone()])?;

            (lambda, tdscf_equation, zeta, idempotency)
        };

        // Make "artificial" Lagrangian multipliers for elimination
        let tdscf_multiplier = LagMultiplier::builder("tdscf-multiplier").build()?;
        let idemp_multiplier = LagMultiplier::builder("idemp_multiplier").build()?;

        lag_terms.push(Trace::new(MatrixMul::new(vec![
            tdscf_multiplier.clone(),
            tdscf_equation.clone(),
        ])?)?);
        lag_terms.push(Trace::new(MatrixMul::new(vec![
            idemp_multiplier.clone(),
            subtract_exprs(idempotency.clone(), density_matrix.clone())?,
        ])?)?);

        // The first term in Equation (98), J. Chem. Phys. 129, 214108 (2008)
        let set: HashSet<Arc<dyn Expr>> = [density_a].into_iter().collect();
        let generalized_energy_a = generalized_energy
            .differentiate(&perturbation_a)?
            .remove(&set)?;

        // The time-averaged quasi-energy derivative Lagrangian
        let lagrangian_dao = subtract_exprs(generalized_energy_a, Add::new(lag_terms)?)?;

        Ok(Self {
            perturbation_a: Some(perturbation_a),
            density_matrix,
            overlap_matrix,
            lambda,
            tdscf_multiplier,
            tdscf_equation,
            zeta,
            idemp_multiplier,
            idempotency,
            lagrangian_dao,
        })
    }

    //#[inline]
    //pub fn new_static(
    //    density_matrix: Arc<dyn Expr>,
    //    overlap_matrix: Option<Arc<dyn Expr>>,
    //    one_elec_operators: &[Arc<dyn Expr>],
    //    two_elec_operator: Option<Arc<dyn Expr>>,
    //    xc_energy: Option<Arc<dyn Expr>>,
    //    xc_potential: Option<Arc<dyn Expr>>,
    //    h_nuc: Option<Arc<dyn Expr>>,
    //) -> Result<Self, TinnedError> {
    //}

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
            return Err(expression_error(
                "Invalid type of density matrix",
                &density_matrix,
                None,
            ));
        }

        let mut energy_terms = Vec::with_capacity(one_elec_operators.len() + 3);
        let mut fock_terms = Vec::with_capacity(one_elec_operators.len() + 2);

        for op in one_elec_operators {
            fock_terms.push(op.clone());

            let matmul = MatrixMul::new(vec![op.clone(), density_matrix.clone()])?;
            let trace = Trace::new(matmul)?;
            energy_terms.push(trace);
        }

        if let Some(op) = &two_elec_operator {
            if let Some(two_elec_op) = downcast_from_arc::<TwoElecOperator>(op) {
                energy_terms.push(TwoElecEnergy::builder_from_operator(two_elec_op).build()?);
            } else {
                return Err(expression_error("Invalid two-electron operator", op, None));
            }
            fock_terms.push(op.clone());
        }

        if let Some(energy) = &xc_energy {
            energy_terms.push(energy.clone());
        }
        if let Some(op) = &xc_potential {
            fock_terms.push(op.clone());
        }
        if let Some(energy) = &h_nuc {
            energy_terms.push(energy.clone());
        }

        let generalized_energy = Add::new(energy_terms)?;
        let fock_matrix = MatrixAdd::new(fock_terms)?;

        Ok((generalized_energy, fock_matrix))
    }

    #[inline]
    pub fn particular_density_solution(
        &self,
        density_freq: Arc<dyn Expr>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        if let Some(density_omega) = downcast_from_arc::<WfnParameter>(&density_freq) {
            let set: HashSet<Arc<dyn Expr>> = [density_freq].into_iter().collect();
            let diff_idempotency =
                differentiate_expr(&self.idempotency, density_omega.derivative())?.remove(&set)?;
            let anticomm_idemp_dm = if let Some(overlap) = &self.overlap_matrix {
                s_anticommutator(
                    diff_idempotency.clone(),
                    self.density_matrix.clone(),
                    overlap.clone(),
                )?
            } else {
                anticommutator(diff_idempotency.clone(), self.density_matrix.clone())?
            };

            subtract_exprs(anticomm_idemp_dm, diff_idempotency)
        } else {
            Err(expression_error(
                "Invalid type density matrix",
                &density_freq,
                None,
            ))
        }
    }

    #[inline]
    pub fn linear_response_rhs(
        &self,
        density_freq: Arc<dyn Expr>,
        density_part: Arc<dyn Expr>,
    ) -> Result<Arc<dyn Expr>, TinnedError> {
        if let Some(density_omega) = downcast_from_arc::<WfnParameter>(&density_freq) {
            let diff_tdscf = differentiate_expr(&self.tdscf_equation, density_omega.derivative())?;
            let map: HashMap<Arc<dyn Expr>, Arc<dyn Expr>> =
                std::iter::once((density_freq, density_part)).collect();

            Ok(diff_tdscf.replace(&map))
        } else {
            Err(expression_error(
                "Invalid type density matrix",
                &density_freq,
                None,
            ))
        }
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
        // Here we need to inlcude the frequency of `perturbation_a`, which can
        // either be extensive or intensive
        let mut terms = vec![freq_sum_ext, freq_sum_int];

        if let Some(pert) = &self.perturbation_a {
            terms.push(pert.frequency().clone());
        }

        let total_freq = Add::new(terms).map_err(|e| {
            generic_error(
                "Sum of all perturbations' frequencies failed",
                Some(Box::new(e)),
            )
        })?;

        Ok(!is_zero_expr(&total_freq, num_tol))
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
            .eliminate(
                &self.tdscf_multiplier,
                exten_perturbations,
                min_multiplier_order,
            )
            .map_err(|e| {
                generic_error("Elimination of TDSCF multiplier failed", Some(Box::new(e)))
            })?;

        result = result
            .eliminate(
                &self.idemp_multiplier,
                exten_perturbations,
                min_multiplier_order,
            )
            .map_err(|e| {
                generic_error(
                    "Elimination of idempotency multiplier failed",
                    Some(Box::new(e)),
                )
            })?;

        // Replace "artificial" multipliers with real differentiated ones
        let replacements: HashMap<Arc<dyn Expr>, Arc<dyn Expr>> = HashMap::from([
            (self.tdscf_multiplier.clone(), self.lambda.clone()),
            (self.idemp_multiplier.clone(), self.zeta.clone()),
        ]);

        Ok(result.replace(&replacements))
    }
}

#[typetag::serde]
impl Lagrangian for LagrangianDao {
    #[inline]
    fn get_lagrangian(&self) -> &Arc<dyn Expr> {
        &self.lagrangian_dao
    }
}
