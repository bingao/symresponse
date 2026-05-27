use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianMcscf};
use tinned::{
    AdjointMap, AdjointMode, ExcitationOperator, MatrixAdd, MatrixMul, Number, OneElecMatrix,
    Perturbation, TinnedError, WfnParameter,
};

#[path = "helpers/perturbation.rs"]
mod perturbation_helpers;

use perturbation_helpers::make_perturbing_operator;

#[test]
fn mcscf_linear_response() -> Result<(), TinnedError> {
    let unperturbed_hamiltonian = OneElecMatrix::builder("H0").build()?;

    let (pert_a, perturbing_oper_a) = make_perturbing_operator("Va", "a", "omega_a")?;
    let (pert_b, perturbing_oper_b) = make_perturbing_operator("Vb", "b", "omega_b")?;

    let perturbing_operators = vec![perturbing_oper_a.clone(), perturbing_oper_b.clone()];

    let rotation_operators = ExcitationOperator::new("T");
    let rotation_parameters = WfnParameter::builder("beta").is_perturbing(true).build()?;

    let lag = LagrangianMcscf::new(
        unperturbed_hamiltonian.clone(),
        &perturbing_operators,
        rotation_operators.clone(),
        rotation_parameters.clone(),
    )?;

    let exten_perturbations = vec![pert_a.clone(), pert_b.clone()];
    let inten_perturbations: Vec<Arc<Perturbation>> = Vec::new();

    // Using `min_wfn_exten_order = 0` means 2n+1 and 2n+2 rules
    let linear_response =
        lag.response_function(&exten_perturbations, &inten_perturbations, 0, false, None)?;

    //match serde_json::to_string(&linear_response) {
    //    Ok(json) => println!("L^{{ab}} = {}", json),
    //    Err(err) => {
    //        eprintln!("Serialization of L^{{ab}} failed: {err}");
    //    },
    //}

    let lambda_operator = lag.lambda_operator().clone();
    let lambda_operator_a = lambda_operator.differentiate(pert_a.clone())?;
    let lambda_operator_b = lambda_operator.differentiate(pert_b.clone())?;
    let diff_perturbing_oper_a = perturbing_oper_a.differentiate(pert_a.clone())?;
    let diff_perturbing_oper_b = perturbing_oper_b.differentiate(pert_b.clone())?;

    // Equation (467), Chem. Rev. 2012, 112, 543-631
    let expected_linear_response = MatrixAdd::new(vec![
        MatrixMul::new(vec![
            Number::imaginary_unit(),
            AdjointMap::new(
                vec![lambda_operator_a.clone()],
                diff_perturbing_oper_b.clone(),
                Some(false),
                Some(AdjointMode::Symmetrized),
            )?,
        ])?,
        MatrixMul::new(vec![
            Number::imaginary_unit(),
            AdjointMap::new(
                vec![lambda_operator_b.clone()],
                diff_perturbing_oper_a,
                Some(false),
                Some(AdjointMode::Symmetrized),
            )?,
        ])?,
        MatrixMul::new(vec![
            Number::from_f64(-1.0),
            AdjointMap::new(
                vec![lambda_operator_a.clone(), lambda_operator_b.clone()],
                unperturbed_hamiltonian,
                Some(false),
                Some(AdjointMode::Symmetrized),
            )?,
        ])?,
        MatrixMul::new(vec![
            pert_a.frequency().clone(),
            AdjointMap::new(
                vec![lambda_operator_b],
                lambda_operator_a,
                Some(false),
                Some(AdjointMode::Symmetrized),
            )?,
        ])?,
    ])?;

    assert_eq!(&linear_response, &expected_linear_response);

    // Check the right-hand side of the first-differentiated rotation parameters
    let rotation_parameters_b = rotation_parameters.differentiate(pert_b)?;
    let rhs_rotation_parameters = lag.linear_response_rhs(&rotation_parameters_b, None)?;

    //match serde_json::to_string(&rhs_rotation_parameters) {
    //    Ok(json) => println!("RHS of beta^{{b}} = {}", json),
    //    Err(err) => {
    //        eprintln!("Serialization of RHS of beta^{{b}} failed: {err}");
    //    },
    //}

    // Equation (466) without the imaginary unit, Chem. Rev. 2012, 112, 543-631
    let expected_rhs = AdjointMap::new(
        vec![rotation_operators],
        diff_perturbing_oper_b,
        Some(true),
        Some(AdjointMode::Symmetrized),
    )?;

    assert_eq!(&rhs_rotation_parameters, &expected_rhs);

    Ok(())
}
