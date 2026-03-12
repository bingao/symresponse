use std::collections::BTreeMap;
use symresponse::{Lagrangian, LagrangianDao, SymmetrizeMode};
use tinned::{
    Mul, Number, OneElecOperator, PertMultichain, Perturbation, Symbol, TemporumOverlap,
    TinnedError, TwoElecOperator, WfnParameter,
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
    // magnetic perturbation
    let perturbing_b_deps = PertMultichain::from_map(BTreeMap::from([(pert_b.clone(), 9)]));

    let overlap_matrix =
        OneElecOperator::builder("S").dependencies(perturbing_b_deps.clone()).build()?;
    let one_elec_hamiltonian =
        OneElecOperator::builder("h").dependencies(perturbing_b_deps.clone()).build()?;
    let t_matrix = TemporumOverlap::builder(perturbing_b_deps.clone()).build()?;

    // We split the perturbing operator into three ones according to the three
    // perturbing fields, see Equation (B2)
    let perturbing_a_deps =
        PertMultichain::from_map(BTreeMap::from([(pert_a.clone(), 1), (pert_b.clone(), 9)]));
    let perturbing_oper_a =
        OneElecOperator::builder("Va").dependencies(perturbing_a_deps).build()?;
    let perturbing_oper_b =
        OneElecOperator::builder("Vb").dependencies(perturbing_b_deps.clone()).build()?;
    let perturbing_c_deps =
        PertMultichain::from_map(BTreeMap::from([(pert_b.clone(), 9), (pert_c.clone(), 1)]));
    let perturbing_oper_c =
        OneElecOperator::builder("Vc").dependencies(perturbing_c_deps).build()?;

    let one_elec_opers = vec![
        one_elec_hamiltonian,
        perturbing_oper_a,
        perturbing_oper_b,
        perturbing_oper_c,
        t_matrix,
    ];

    let two_elec_operator = TwoElecOperator::builder("G", density_matrix.clone())
        .dependencies(perturbing_b_deps)
        .build()?;

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

    let exten_perturbations = vec![pert_b.clone(), pert_c.clone()];
    let inten_perturbations = Vec::new();

    // Using `min_wfn_extern = 0` means it will be determined by SymResponse as
    // the next integer of the floor function of the half number of extensive
    // perturbations
    let response =
        lag.response_function(&exten_perturbations, &inten_perturbations, 0, false, None)?;

    let json_residue = serde_json::to_string(&response).unwrap();
    println!("response: {}", json_residue);

    Ok(())
}
