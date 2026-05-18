use std::collections::HashMap;
use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianCc};
use tinned::{
    ExcitationOperator, Expr, LagMultiplier, MatrixAdd, MatrixMul, OneElecMatrix, Perturbation,
    ResidueParameter, TinnedError, WfnParameter, differentiate_expr,
};

mod common;
use common::{make_cc_st_operator, make_perturbing_operator};

#[test]
fn cc_first_order_lr_residue() -> Result<(), TinnedError> {
    let unperturbed_hamiltonian = OneElecMatrix::builder("H0").build()?;

    let (pert_a, perturbing_oper_a) = make_perturbing_operator("Va", "a", "omega_a")?;
    let (pert_b, perturbing_oper_b) = make_perturbing_operator("Vb", "b", "omega_b")?;

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

    let excited_state = WfnParameter::builder("Xf").build()?;
    let residue_relations: HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)> =
        HashMap::from([(excited_state.clone(), (true, vec![pert_b.clone()]))]);

    // Using `min_wfn_exten_order = 0` means 2n+1 and 2n+2 rules
    let residue = lag.residue(
        &exten_perturbations,
        &inten_perturbations,
        0,
        &residue_relations,
        false,
        None,
    )?;

    //match serde_json::to_string(&residue) {
    //    Ok(json) => println!("Residue = {}", json),
    //    Err(err) => {
    //        eprintln!("Serialization of residue failed: {err}");
    //    },
    //}

    // Residue couple-cluster amplitude
    let amplitude_b = cc_amplitude.differentiate(pert_b.clone())?;
    let res_amplitude_b =
        ResidueParameter::builder(vec![pert_b], excited_state, amplitude_b.clone())
            .positive_frequency(true)
            .build()?;

    // First term of equation (57), J. Chem. Phys., 108, 8331-8354 (1998)
    let cluster_operator = lag.cluster_operator().clone();
    let cc_lambda_operator = lag.cc_lambda_operator().clone();
    let max_commutator_order = LagrangianCc::max_commutator_order();

    // Similarity-transformed operators
    let st_unperturbed_hamiltonian = make_cc_st_operator(
        cluster_operator.clone(),
        unperturbed_hamiltonian,
        max_commutator_order,
    )?;
    let st_perturbing_oper_a =
        make_cc_st_operator(cluster_operator.clone(), perturbing_oper_a, max_commutator_order)?;
    let st_perturbing_oper_b =
        make_cc_st_operator(cluster_operator, perturbing_oper_b, max_commutator_order)?;

    let lagrangian_energy = MatrixAdd::new(vec![
        st_unperturbed_hamiltonian.clone(),
        st_perturbing_oper_a.clone(),
        st_perturbing_oper_b.clone(),
        MatrixMul::new(vec![cc_lambda_operator.clone(), st_unperturbed_hamiltonian])?,
        MatrixMul::new(vec![cc_lambda_operator.clone(), st_perturbing_oper_a])?,
        MatrixMul::new(vec![cc_lambda_operator, st_perturbing_oper_b])?,
    ])?;

    let expected_residue = differentiate_expr(&lagrangian_energy, &exten_perturbations)?
        .eliminate(cc_amplitude, &exten_perturbations, 2)?
        .eliminate(cc_multiplier, &exten_perturbations, 1)?
        .substitute_zero_perturbations(None)?
        .retain_one(&amplitude_b, true)?
        .replace_one(&amplitude_b, res_amplitude_b, true)?;

    assert_eq!(&residue, &expected_residue);

    Ok(())
}

// Two-photon transition moments from the cubic response function
#[test]
fn cc_2p_tme() -> Result<(), TinnedError> {
    let unperturbed_hamiltonian = OneElecMatrix::builder("H0").build()?;

    let (pert_a, perturbing_oper_a) = make_perturbing_operator("Va", "a", "omega_a")?;
    let (pert_b, perturbing_oper_b) = make_perturbing_operator("Vb", "b", "omega_b")?;
    let (pert_c, perturbing_oper_c) = make_perturbing_operator("Vc", "c", "omega_c")?;
    let (pert_d, perturbing_oper_d) = make_perturbing_operator("Vd", "d", "omega_d")?;

    let perturbing_operators = vec![
        perturbing_oper_a.clone(),
        perturbing_oper_b.clone(),
        perturbing_oper_c.clone(),
        perturbing_oper_d.clone(),
    ];

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

    let exten_perturbations = vec![pert_a.clone(), pert_b.clone(), pert_c, pert_d];
    let inten_perturbations: Vec<Arc<Perturbation>> = Vec::new();

    let excited_state = WfnParameter::builder("Xf").build()?;
    let residue_relations: HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)> =
        HashMap::from([(excited_state.clone(), (true, vec![pert_a.clone(), pert_b.clone()]))]);

    // Using `min_wfn_exten_order = 0` means 2n+1 and 2n+2 rules
    let residue = lag.residue(
        &exten_perturbations,
        &inten_perturbations,
        0,
        &residue_relations,
        false,
        None,
    )?;

    //match serde_json::to_string(&residue) {
    //    Ok(json) => println!("Residue = {}", json),
    //    Err(err) => {
    //        eprintln!("Serialization of residue failed: {err}");
    //    },
    //}

    // Residue couple-cluster amplitude
    let pert_ab = vec![pert_a, pert_b];
    let amplitude_ab = differentiate_expr(&cc_amplitude, &pert_ab)?;
    let res_amplitude_ab = ResidueParameter::builder(pert_ab, excited_state, amplitude_ab.clone())
        .positive_frequency(true)
        .build()?;

    // The expected residue is obtained from the last five terms of equation
    // (22), Chem. Phys. Lett. 282 (1998) 139-146.
    let cluster_operator = lag.cluster_operator().clone();
    let cc_lambda_operator = lag.cc_lambda_operator().clone();
    let max_commutator_order = LagrangianCc::max_commutator_order();

    // Similarity-transformed operators
    let st_unperturbed_hamiltonian = make_cc_st_operator(
        cluster_operator.clone(),
        unperturbed_hamiltonian,
        max_commutator_order,
    )?;
    let st_perturbing_oper_a =
        make_cc_st_operator(cluster_operator.clone(), perturbing_oper_a, max_commutator_order)?;
    let st_perturbing_oper_b =
        make_cc_st_operator(cluster_operator.clone(), perturbing_oper_b, max_commutator_order)?;
    let st_perturbing_oper_c =
        make_cc_st_operator(cluster_operator.clone(), perturbing_oper_c, max_commutator_order)?;
    let st_perturbing_oper_d =
        make_cc_st_operator(cluster_operator, perturbing_oper_d, max_commutator_order)?;

    let lagrangian_energy = MatrixAdd::new(vec![
        st_unperturbed_hamiltonian.clone(),
        st_perturbing_oper_a.clone(),
        st_perturbing_oper_b.clone(),
        st_perturbing_oper_c.clone(),
        st_perturbing_oper_d.clone(),
        MatrixMul::new(vec![cc_lambda_operator.clone(), st_unperturbed_hamiltonian])?,
        MatrixMul::new(vec![cc_lambda_operator.clone(), st_perturbing_oper_a])?,
        MatrixMul::new(vec![cc_lambda_operator.clone(), st_perturbing_oper_b])?,
        MatrixMul::new(vec![cc_lambda_operator.clone(), st_perturbing_oper_c])?,
        MatrixMul::new(vec![cc_lambda_operator, st_perturbing_oper_d])?,
    ])?;

    let expected_residue = differentiate_expr(&lagrangian_energy, &exten_perturbations)?
        .eliminate(cc_amplitude, &exten_perturbations, 3)?
        .eliminate(cc_multiplier, &exten_perturbations, 2)?
        .substitute_zero_perturbations(None)?
        .retain_one(&amplitude_ab, true)?
        .replace_one(&amplitude_ab, res_amplitude_ab, true)?;

    //match serde_json::to_string(&expected_residue) {
    //    Ok(json) => println!("Expected residue = {}", json),
    //    Err(err) => {
    //        eprintln!("Serialization of expected residue failed: {err}");
    //    },
    //}

    assert_eq!(&residue, &expected_residue);

    Ok(())
}
