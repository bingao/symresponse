use std::collections::BTreeMap;
use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianCc};
use tinned::{
    AdjointMap, AdjointMode, ExcitationOperator, ExpAdjointMap, LagMultiplier, MatrixAdd,
    MatrixMul, Number, OneElecMatrix, PertMultichain, Perturbation, Symbol, TinnedError,
    WfnParameter, differentiate_expr,
};

#[test]
fn cc_linear_quadratic_response() -> Result<(), TinnedError> {
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

    let freq_c = Symbol::new("omega_c");
    let pert_c = Perturbation::new("c", freq_c);
    let pert_c_deps = PertMultichain::from_map(BTreeMap::from([(pert_c.clone(), 1)]));
    let perturbing_oper_c =
        OneElecMatrix::builder("Vc").is_perturbing(true).dependencies(pert_c_deps).build()?;

    let perturbing_operators =
        vec![perturbing_oper_a.clone(), perturbing_oper_b.clone(), perturbing_oper_c.clone()];

    let cc_amplitude = WfnParameter::builder("t").build()?;
    let cc_excitation_operator = ExcitationOperator::new("tau");
    let cc_multiplier = LagMultiplier::builder("tbar").build()?;

    let lag = LagrangianCc::new(
        unperturbed_hamiltonian.clone(),
        &perturbing_operators,
        cc_amplitude.clone(),
        cc_excitation_operator.clone(),
        cc_multiplier.clone(),
    )?;

    let mut exten_perturbations = vec![pert_a.clone(), pert_b.clone()];
    let inten_perturbations: Vec<Arc<Perturbation>> = Vec::new();

    // Using `min_wfn_exten = 0` means 2n+1 and 2n+2 rules
    let linear_response =
        lag.response_function(&exten_perturbations, &inten_perturbations, 0, false, None)?;

    //match serde_json::to_string(&linear_response) {
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
        ExpAdjointMap::builder(cluster_operator.clone(), perturbing_oper_a.clone(), Some(true))
            .left_action(false)
            .max_commutator_order(max_commutator_order)
            .build()?;
    let st_perturbing_oper_b =
        ExpAdjointMap::builder(cluster_operator.clone(), perturbing_oper_b.clone(), Some(true))
            .left_action(false)
            .max_commutator_order(max_commutator_order)
            .build()?;
    let st_perturbing_oper_c =
        ExpAdjointMap::builder(cluster_operator.clone(), perturbing_oper_c.clone(), Some(true))
            .left_action(false)
            .max_commutator_order(max_commutator_order)
            .build()?;

    let lagrangian_energy = MatrixAdd::new(vec![
        st_unperturbed_hamiltonian.clone(),
        st_perturbing_oper_a.clone(),
        st_perturbing_oper_b.clone(),
        st_perturbing_oper_c.clone(),
        MatrixMul::new(vec![de_excitation_operator.clone(), st_unperturbed_hamiltonian])?,
        MatrixMul::new(vec![de_excitation_operator.clone(), st_perturbing_oper_a.clone()])?,
        MatrixMul::new(vec![de_excitation_operator.clone(), st_perturbing_oper_b])?,
        MatrixMul::new(vec![de_excitation_operator.clone(), st_perturbing_oper_c])?,
    ])?;
    let expected_linear_response = differentiate_expr(&lagrangian_energy, &exten_perturbations)?
        .eliminate(&cc_amplitude, &exten_perturbations, 2)?
        .eliminate(&cc_multiplier, &exten_perturbations, 1)?
        .substitute_zero_perturbations(None)?;

    //match serde_json::to_string(&expected_linear_response) {
    //    Ok(json) => println!("Expected L^{{ab}} = {}", json),
    //    Err(err) => {
    //        eprintln!("Serialization of expected L^{{ab}} failed: {err}");
    //    },
    //}

    assert_eq!(&linear_response, &expected_linear_response);

    // Check the right-hand side of first-order differentiated coupled-cluster amplitude
    let cc_amplitude_a = cc_amplitude.differentiate(&pert_a)?;
    let rhs_cc_amplitude = lag.linear_response_rhs(&cc_amplitude_a, None)?;
    let expected_rhs_cc_amplitude = MatrixMul::new(vec![
        Number::minus_one(),
        st_perturbing_oper_a
            .differentiate(&pert_a)?
            .eliminate(&cc_amplitude, &exten_perturbations, 2)?
            .substitute_zero_perturbations(None)?,
    ])?;

    assert_eq!(&rhs_cc_amplitude, &expected_rhs_cc_amplitude);

    exten_perturbations.push(pert_c.clone());

    let quadratic_response =
        lag.response_function(&exten_perturbations, &inten_perturbations, 0, false, None)?;

    //match serde_json::to_string(&quadratic_response) {
    //    Ok(json) => println!("L^{{abc}} = {}", json),
    //    Err(err) => {
    //        eprintln!("Serialization of L^{{abc}} failed: {err}");
    //    },
    //}

    // Build reference response function, equation (S2), J. Phys. Chem. A 2025, 129, 3709-3721.
    let expected_quadratic_response = differentiate_expr(&lagrangian_energy, &exten_perturbations)?
        .eliminate(&cc_amplitude, &exten_perturbations, 2)?
        .eliminate(&cc_multiplier, &exten_perturbations, 2)?
        .substitute_zero_perturbations(None)?;

    //match serde_json::to_string(&expected_quadratic_response) {
    //    Ok(json) => println!("Expected L^{{abc}} = {}", json),
    //    Err(err) => {
    //        eprintln!("Serialization of expected L^{{abc}} failed: {err}");
    //    },
    //}

    assert_eq!(&quadratic_response, &expected_quadratic_response);

    // Get the right-hand side of first-order differentiated Lagrangian multiplier
    let cc_multiplier_a = cc_multiplier.differentiate(&pert_a)?;
    let rhs_cc_multiplier = lag.linear_response_rhs(&cc_multiplier_a, None)?;

    //match serde_json::to_string(&rhs_cc_multiplier) {
    //    Ok(json) => println!("RHS of lambda^{{a}} = {}", json),
    //    Err(err) => {
    //        eprintln!("Serialization of lambda^{{a}} failed: {err}");
    //    },
    //}

    // Similarity-transformed commutator between operators and the excitation operator
    let st_comm_unperturbed_hamiltonian = ExpAdjointMap::builder(
        cluster_operator.clone(),
        AdjointMap::new(
            vec![cc_excitation_operator.clone()],
            unperturbed_hamiltonian,
            Some(false),
            Some(AdjointMode::Commutative),
        )?,
        Some(true),
    )
    .left_action(false)
    .max_commutator_order(max_commutator_order)
    .build()?;
    let st_comm_perturbing_oper_a = ExpAdjointMap::builder(
        cluster_operator.clone(),
        AdjointMap::new(
            vec![cc_excitation_operator.clone()],
            perturbing_oper_a,
            Some(false),
            Some(AdjointMode::Commutative),
        )?,
        Some(true),
    )
    .left_action(false)
    .max_commutator_order(max_commutator_order)
    .build()?;
    let st_comm_perturbing_oper_b = ExpAdjointMap::builder(
        cluster_operator.clone(),
        AdjointMap::new(
            vec![cc_excitation_operator.clone()],
            perturbing_oper_b,
            Some(false),
            Some(AdjointMode::Commutative),
        )?,
        Some(true),
    )
    .left_action(false)
    .max_commutator_order(max_commutator_order)
    .build()?;
    let st_comm_perturbing_oper_c = ExpAdjointMap::builder(
        cluster_operator.clone(),
        AdjointMap::new(
            vec![cc_excitation_operator.clone()],
            perturbing_oper_c,
            Some(false),
            Some(AdjointMode::Commutative),
        )?,
        Some(true),
    )
    .left_action(false)
    .max_commutator_order(max_commutator_order)
    .build()?;

    let cc_multiplier_equation = MatrixAdd::new(vec![
        st_comm_unperturbed_hamiltonian.clone(),
        st_comm_perturbing_oper_a.clone(),
        st_comm_perturbing_oper_b.clone(),
        st_comm_perturbing_oper_c.clone(),
        MatrixMul::new(vec![de_excitation_operator.clone(), st_comm_unperturbed_hamiltonian])?,
        MatrixMul::new(vec![de_excitation_operator.clone(), st_comm_perturbing_oper_a])?,
        MatrixMul::new(vec![de_excitation_operator.clone(), st_comm_perturbing_oper_b])?,
        MatrixMul::new(vec![de_excitation_operator, st_comm_perturbing_oper_c])?,
    ])?;

    let expected_rhs_cc_multiplier = MatrixMul::new(vec![
        Number::minus_one(),
        cc_multiplier_equation
            .differentiate(&pert_a)?
            .eliminate(&cc_amplitude, &exten_perturbations, 2)?
            .eliminate(&cc_multiplier, &exten_perturbations, 1)?
            .substitute_zero_perturbations(None)?,
    ])?;

    //match serde_json::to_string(&expected_rhs_cc_multiplier) {
    //    Ok(json) => println!("Expected RHS of lambda^{{a}} = {}", json),
    //    Err(err) => {
    //        eprintln!("Serialization of expected lambda^{{a}} failed: {err}");
    //    },
    //}

    assert_eq!(&rhs_cc_multiplier, &expected_rhs_cc_multiplier);

    Ok(())
}
