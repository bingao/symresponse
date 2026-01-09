use serde_json;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianDao};
use tinned::{
    Expr, OneElecOperator, PertMultichain, Perturbation, ResidueParameter, Symbol, TemporumOverlap, TinnedError, TwoElecOperator, WfnParameter
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
        density_matrix.clone(),
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
        HashMap::from([(excited_state.clone(), (false, vec![pert_b.clone()]))]);

    // Using `min_wfn_extern = 3` removes all Lagrangian multipliers
    let residue =
        lag.residue(&exten_perturbations, &inten_perturbations, 3, &residue_info, false, None)?;

    // Reference residue
    let residue_json = include_str!("data/dao_linear_residue.json");
    let expected_residue: Arc<dyn Expr> = serde_json::from_str(residue_json)
        .expect("Failed to deserialize the first-order residue of the linear response function");

    assert_eq!(&residue, &expected_residue);

    // Check the right-hand side of the linear response equation (289), J. Chem. Phys. 129, 214108 (2008)
    let diff_dmat = density_matrix.differentiate(&pert_b)?;
    let density_freq = ResidueParameter::builder(
                        vec![pert_b.clone()],
                        excited_state.clone(),
                        diff_dmat,
                    )
                    .positive_frequency(false)
                    .build()?;
    let density_part = WfnParameter::builder("D_P").build()?;
    let rhs = lag.linear_response_rhs(density_freq, density_part, None)?;

    // Reference RHS
    let rhs_json = include_str!("data/dao_linear_rhs.json");
    let expected_rhs: Arc<dyn Expr> = serde_json::from_str(rhs_json)
        .expect("Failed to deserialize the right-hand side of the linear response equation");

    assert_eq!(&rhs, &expected_rhs);

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
