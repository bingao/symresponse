use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianMcscf};
use tinned::{ExcitationOperator, OneElecMatrix, Perturbation, TinnedError, WfnParameter};

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
        unperturbed_hamiltonian,
        &perturbing_operators,
        rotation_operators,
        rotation_parameters,
    )?;

    let exten_perturbations = vec![pert_a, pert_b];
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

    

    Ok(())
}
