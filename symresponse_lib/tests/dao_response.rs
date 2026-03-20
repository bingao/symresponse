use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianDao, SymmetrizeMode};
use tinned::{
    Add, Expr, MatrixMul, Mul, Number, OneElecOperator, PertMultichain, Perturbation, Symbol,
    TemporumOverlap, TinnedError, Trace, TwoElecOperator, WfnParameter, differentiate_expr,
};

// Magnetic circular dichroism
// J. Chem. Phys. 135, 024112 (2011)
#[test]
fn lao_quadratic_response() -> Result<(), TinnedError> {
    // `a` and `c` are electric perturbations with frequency as -omega and +omega
    let freq_el = Symbol::new("omega");
    let pert_a = Perturbation::new("a", Mul::new(vec![Number::minus_one(), freq_el.clone()])?);
    let pert_c = Perturbation::new("c", freq_el);
    // `b` is a magnetic perturbation with frequency as zero
    let pert_b = Perturbation::new("b", Number::zero());

    let density_matrix = WfnParameter::builder("D").build()?;

    // Since we use London atomic orbitals, all operators depend on the
    // magnetic perturbation and can be differentiated infinitely
    let perturbing_b_deps = PertMultichain::from_map(BTreeMap::from([(pert_b.clone(), 99)]));

    let overlap_matrix =
        OneElecOperator::builder("S").dependencies(perturbing_b_deps.clone()).build()?;
    let one_elec_hamiltonian =
        OneElecOperator::builder("h").dependencies(perturbing_b_deps.clone()).build()?;
    let t_matrix = TemporumOverlap::builder(perturbing_b_deps.clone()).build()?;

    // Perturbing operator of Equation (B2), which can be differentiated with
    // respect to electric perturbations only once
    let perturbing_deps = PertMultichain::from_map(BTreeMap::from([
        (pert_a.clone(), 1),
        (pert_b.clone(), 99),
        (pert_c.clone(), 1),
    ]));
    let perturbing_indep_perts = BTreeSet::from([pert_a.clone(), pert_c.clone()]);
    let perturbing_oper = OneElecOperator::builder("V")
        .dependencies(perturbing_deps)
        .independent_perturbations(perturbing_indep_perts)
        .is_perturbing(true)
        .build()?;

    let one_elec_opers = vec![one_elec_hamiltonian, perturbing_oper.clone(), t_matrix];

    let two_elec_operator = TwoElecOperator::builder("G", density_matrix.clone())
        .dependencies(perturbing_b_deps)
        .build()?;

    // We ignore exchange-correlation functional in this simple example.
    // Equation (B1) is obtained by symmetrization so we set `symmetrized_mode` as `Always`.
    let lag = LagrangianDao::new(
        pert_a.clone(),
        density_matrix.clone(),
        Some(overlap_matrix),
        &one_elec_opers,
        Some(two_elec_operator),
        None,
        None,
        None,
        Some(SymmetrizeMode::Always),
        None,
    )?;

    // We treat all perturbations as externsive ones
    let exten_perturbations = vec![pert_b.clone(), pert_c.clone()];
    let inten_perturbations = Vec::new();

    // Using `min_wfn_extern = 0` means it will be determined by SymResponse as
    // the next integer of the floor function of the half number of extensive
    // perturbations
    let response =
        lag.response_function(&exten_perturbations, &inten_perturbations, 0, false, None)?;

    // F^{bc}
    let fock_matrix_bc = differentiate_expr(lag.fock_matrix(), &exten_perturbations)?;
    let max_fock_derivs: HashSet<Arc<dyn Expr>> = [fock_matrix_bc.clone()].into_iter().collect();

    // Differentiated generalized energy of Equation (B1)
    //let generalized_energy_abc = Trace::new(MatrixMul::new(vec![
    //    perturbing_oper_a.differentiate(&pert_a)?.differentiate(&pert_b)?,
    //    density_matrix.differentiate(&pert_c)?,
    //])?)?;
    let generalized_energy_abc =
        differentiate_expr(lag.generalized_energy_a(), &exten_perturbations)?
            .remove(&max_fock_derivs)?;
    // TDSCF equation of Equation (B1)
    let tdscf_equation_bc =
        differentiate_expr(lag.tdscf_equation(), &exten_perturbations)?.remove(&max_fock_derivs)?;
    // Idempotency constraint of Equation (B1)
    let idempotency_bc =
        differentiate_expr(lag.idempotency(), &exten_perturbations)?.remove(&max_fock_derivs)?;
    // Construct terms of Equation (B1)
    let mut expected_response = Add::new(vec![
        generalized_energy_abc,
        //  F^{bc} * D^{a}
        Trace::new(MatrixMul::new(vec![fock_matrix_bc, density_matrix.differentiate(&pert_a)?])?)?,
        Mul::new(vec![
            Number::minus_one(),
            Add::new(vec![
                // TDSCF equation constraint
                Trace::new(MatrixMul::new(vec![
                    lag.tdscf_multiplier_expr().clone(),
                    tdscf_equation_bc,
                ])?)?,
                // Idempotency constraint
                Trace::new(MatrixMul::new(vec![
                    lag.idemp_multiplier_expr().clone(),
                    idempotency_bc,
                ])?)?,
            ])?,
        ])?,
    ])?
    .eliminate(&density_matrix, &exten_perturbations, 2)?;

    // Elimiate multipliers for TDSCF equation and dempotency constraints
    let multipliers = lag.get_lag_multiplier();
    for multiplier in &multipliers {
        expected_response = expected_response.eliminate(multiplier, &exten_perturbations, 1)?;
    }

    // Final result of Equation (B1)
    expected_response = expected_response.apply_zero_rules(None)?;

    //FIXME: Maybe we should construct the expected response function from inputs we sent to LagrangianDao::new()?
    assert_eq!(&response, &expected_response);

    //let json = serde_json::to_string(&response).unwrap();
    //println!("Response function = {}", json);

    Ok(())
}
