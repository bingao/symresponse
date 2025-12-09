use serde_json;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianDao};
use tinned::{
    Expr, OneElecOperator, PertMultichain, Perturbation, Symbol, TemporumOverlap, TinnedError,
    TwoElecOperator, WfnParameter,
};

// First-order residue of the linear response function, equation (286), J. Chem. Phys. 129, 214108 (2008)
#[test]
fn test_linear_response_function() -> Result<(), TinnedError> {
    let freq_a = Symbol::new("omega_a");
    let pert_a = Perturbation::new("a", freq_a);
    let freq_b = Symbol::new("omega_b");
    let pert_b = Perturbation::new("b", freq_b);

    let density_matrix = WfnParameter::builder("D").build()?;
    let oper_deps =
        PertMultichain::from_map(BTreeMap::from([(pert_a.clone(), 9), (pert_b.clone(), 9)]));
    let overlap_matrix = OneElecOperator::builder("S").dependencies(oper_deps.clone()).build()?;

    let one_elec_hamiltonian =
        OneElecOperator::builder("h").dependencies(oper_deps.clone()).build()?;
    let perturbing_oper = OneElecOperator::builder("V").dependencies(oper_deps.clone()).build()?;
    let t_matrix = TemporumOverlap::builder(oper_deps.clone()).build()?;
    let one_elec_opers = vec![one_elec_hamiltonian, perturbing_oper, t_matrix];

    let two_elec_operator =
        TwoElecOperator::builder("G", density_matrix.clone()).dependencies(oper_deps).build()?;

    let lag = LagrangianDao::new(
        pert_a,
        density_matrix,
        Some(overlap_matrix),
        &one_elec_opers,
        Some(two_elec_operator),
        None,
        None,
        None,
        None,
    )?;

    let exten_perturbations = vec![pert_b.clone()];
    let inten_perturbations = Vec::new();

    let excited_state = WfnParameter::builder("Xq").build()?;
    let residue_info: HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)> =
        HashMap::from([(excited_state, (false, vec![pert_b]))]);

    // Using `min_wfn_extern = 3` removes all Lagrangian multipliers
    let residue =
        lag.residue(&exten_perturbations, &inten_perturbations, 3, &residue_info, false, None)?;

    // Reference residue
    let json = include_str!("data/dao_residue_linear.json");
    let result: Arc<dyn Expr> = serde_json::from_str(json)
        .expect("Failed to deserialize the first-order residue of the linear response function");

    assert_eq!(&residue, &result);

    Ok(())
}

// Three-photon transition matrix element between the ground state and the excited state
//#[test]
//fn test_3p_tme() {
//    LagrangianDao::new(
//        perturbation_a,
//        density_matrix,
//        overlap_matrix,
//        one_elec_operators,
//        two_elec_operator,
//        xc_energy,
//        xc_potential,
//        h_nuc,
//        num_tol,
//    )
//}
