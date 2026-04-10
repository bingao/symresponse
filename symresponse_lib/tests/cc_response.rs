use std::collections::BTreeMap;
use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianCc};
use tinned::{
    ExcitationOperator, ExpAdjointMap, LagMultiplier, MatrixAdd, MatrixMul, OneElecMatrix,
    PertMultichain, Perturbation, Symbol, TinnedError, WfnParameter, differentiate_expr,
};

#[test]
fn cc_linear_response() -> Result<(), TinnedError> {
    let unperturbed_hamiltonian = OneElecMatrix::builder("H0").build()?;

    let freq_a = Symbol::new("omega_a");
    let pert_a = Perturbation::new("a", freq_a);
    let pert_a_deps = PertMultichain::from_map(BTreeMap::from([(pert_a.clone(), 1)]));
    let perturbing_oper_a =
        OneElecMatrix::builder("Va").is_perturbing(true).dependencies(pert_a_deps).build()?;

    let freq_b = Symbol::new("omega_b");
    let pert_b = Perturbation::new("b", freq_b);
    let pert_b_deps = PertMultichain::from_map(BTreeMap::from([(pert_b.clone(), 1)]));
    let perturbing_oper_b =
        OneElecMatrix::builder("Vb").is_perturbing(true).dependencies(pert_b_deps).build()?;

    let perturbing_operators = vec![perturbing_oper_a.clone(), perturbing_oper_b.clone()];

    let cc_amplitude = WfnParameter::builder("t").build()?;
    let cc_excitation_operator = ExcitationOperator::new("tau");
    let cc_multiplier = LagMultiplier::builder("tbar").build()?;

    let lag = LagrangianCc::new(
        unperturbed_hamiltonian.clone(),
        &perturbing_operators,
        cc_amplitude.clone(),
        cc_excitation_operator,
        cc_multiplier.clone(),
    )?;

    let exten_perturbations = vec![pert_a.clone(), pert_b.clone()];
    let inten_perturbations: Vec<Arc<Perturbation>> = Vec::new();

    // Using `min_wfn_exten = 0` means 2n+1 and 2n+2 rules
    let result =
        lag.response_function(&exten_perturbations, &inten_perturbations, 0, false, None)?;

    //match serde_json::to_string(&result) {
    //    Ok(json) => println!("L^{{ab}} = {}", json),
    //    Err(err) => {
    //        eprintln!("Serialization of L^{{ab}} failed: {err}");
    //    },
    //}

    // Build reference response function, equation (S1), J. Phys. Chem. A 2025, 129, 3709-3721.
    let cluster_operator = lag.cluster_operator().clone();
    let de_excitation_operator = lag.de_excitation_operator().clone();
    let max_commutator_order = LagrangianCc::max_commutator_order();
    // Similarity-transformed operators
    let st_unperturbed_hamiltonian = ExpAdjointMap::builder(
        cluster_operator.clone(),
        unperturbed_hamiltonian.clone(),
        Some(true),
    )
    .left_action(false)
    .max_commutator_order(max_commutator_order)
    .build()?;
    let st_perturbing_oper_a =
        ExpAdjointMap::builder(cluster_operator.clone(), perturbing_oper_a, Some(true))
            .left_action(false)
            .max_commutator_order(max_commutator_order)
            .build()?;
    let st_perturbing_oper_b =
        ExpAdjointMap::builder(cluster_operator.clone(), perturbing_oper_b, Some(true))
            .left_action(false)
            .max_commutator_order(max_commutator_order)
            .build()?;

    let expected_response = differentiate_expr(
        &MatrixAdd::new(vec![
            st_unperturbed_hamiltonian.clone(),
            st_perturbing_oper_a.clone(),
            st_perturbing_oper_b.clone(),
            MatrixMul::new(vec![de_excitation_operator.clone(), st_unperturbed_hamiltonian])?,
            MatrixMul::new(vec![de_excitation_operator.clone(), st_perturbing_oper_a])?,
            MatrixMul::new(vec![de_excitation_operator, st_perturbing_oper_b])?,
        ])?,
        &exten_perturbations,
    )?
    .eliminate(&cc_amplitude, &exten_perturbations, 2)?
    .eliminate(&cc_multiplier, &exten_perturbations, 1)?
    .substitute_zero_perturbations(None)?;

    //match serde_json::to_string(&expected_response) {
    //    Ok(json) => println!("Expected L^{{ab}} = {}", json),
    //    Err(err) => {
    //        eprintln!("Serialization of expected L^{{ab}} failed: {err}");
    //    },
    //}

    assert_eq!(&result, &expected_response);

    Ok(())
}
