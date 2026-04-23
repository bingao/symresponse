use std::collections::{BTreeMap, BTreeSet};
use symresponse::{Lagrangian, LagrangianDao, SymmetrizeMode};
use tinned::{
    Add, AoTwoElecMatrix, BasisTimeEvolution, ExchCorrEnergy, ExchCorrPotential, MatrixMul, Mul,
    NonElecFunction, Number, OneElecMatrix, PertMultichain, Perturbation, Symbol, TinnedError,
    Trace, WfnParameter, differentiate_expr,
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
    let perturbing_b_deps = PertMultichain::from_map(BTreeMap::from([(pert_b.clone(), u32::MAX)]));

    let overlap_matrix =
        OneElecMatrix::builder("S").dependencies(perturbing_b_deps.clone()).build()?;
    let one_elec_hamiltonian =
        OneElecMatrix::builder("h").dependencies(perturbing_b_deps.clone()).build()?;
    let t_matrix = BasisTimeEvolution::builder(perturbing_b_deps.clone()).build()?;

    // Perturbing operator of Equation (B2), which can be differentiated with
    // respect to electric perturbations only once
    let perturbing_deps = PertMultichain::from_map(BTreeMap::from([
        (pert_a.clone(), 1),
        (pert_b.clone(), u32::MAX),
        (pert_c.clone(), 1),
    ]));
    let perturbing_indep_perts = BTreeSet::from([pert_a.clone(), pert_c.clone()]);
    let perturbing_oper = OneElecMatrix::builder("V")
        .is_perturbing(true)
        .dependencies(perturbing_deps)
        .independent_perturbations(perturbing_indep_perts)
        .build()?;

    let one_elec_opers = vec![one_elec_hamiltonian, perturbing_oper.clone(), t_matrix];

    let two_elec_operator = AoTwoElecMatrix::builder("G", density_matrix.clone())
        .dependencies(perturbing_b_deps)
        .build()?;

    let grid_weight = NonElecFunction::builder("w").build()?;
    let overlap_distribution = OneElecMatrix::builder("Omega").build()?;
    let xc_energy = ExchCorrEnergy::builder(
        "Exc",
        grid_weight.clone(),
        density_matrix.clone(),
        overlap_distribution.clone(),
    )
    .build()?;
    let xc_potential = ExchCorrPotential::builder(
        "Vxc",
        grid_weight,
        density_matrix.clone(),
        overlap_distribution,
    )
    .build()?;

    // We ignore exchange-correlation functional in this simple example.
    // Equation (B1) is obtained by symmetrization so we set `symmetrized_mode` as `Always`.
    let lag = LagrangianDao::new(
        pert_a.clone(),
        density_matrix.clone(),
        Some(overlap_matrix),
        &one_elec_opers,
        Some(two_elec_operator),
        Some(xc_energy),
        Some(xc_potential),
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

    // Differentiated generalized energy of Equation (B1)
    //let generalized_energy_abc = Trace::new(MatrixMul::new(vec![
    //    perturbing_oper_a.differentiate(&pert_a)?.differentiate(&pert_b)?,
    //    density_matrix.differentiate(&pert_c)?,
    //])?)?;
    let generalized_energy_abc =
        differentiate_expr(lag.generalized_energy_a(), &exten_perturbations)?
            .remove_one(&fock_matrix_bc)?;
    // TDSCF equation of Equation (B1)
    let tdscf_equation_bc = differentiate_expr(lag.tdscf_equation(), &exten_perturbations)?
        .remove_one(&fock_matrix_bc)?;
    // Idempotency constraint of Equation (B1)
    let idempotency_bc =
        differentiate_expr(lag.idempotency(), &exten_perturbations)?.remove_one(&fock_matrix_bc)?;
    // Construct terms of Equation (B1)
    let mut expected_response = Add::new(vec![
        generalized_energy_abc,
        //  F^{bc} * D^{a}
        Trace::new(MatrixMul::new(vec![fock_matrix_bc, density_matrix.differentiate(pert_a)?])?)?,
        Mul::new(vec![
            Number::minus_one(),
            Add::new(vec![
                // TDSCF equation constraint
                Trace::new(MatrixMul::new(vec![
                    lag.tdscf_multiplier().clone(),
                    tdscf_equation_bc,
                ])?)?,
                // Idempotency constraint
                Trace::new(MatrixMul::new(vec![lag.idemp_multiplier().clone(), idempotency_bc])?)?,
            ])?,
        ])?,
    ])?
    .eliminate(density_matrix, &exten_perturbations, 2)?;

    //FIXME: is this elimination necessary?
    // Elimiate multipliers for TDSCF equation and idempotency constraints
    for multiplier in lag.get_lagrangian_multipliers() {
        expected_response = expected_response.eliminate(multiplier, &exten_perturbations, 1)?;
    }

    // Final result of Equation (B1)
    expected_response = expected_response.substitute_zero_perturbations(None)?;

    //FIXME: Maybe we should construct the expected response function from inputs we sent to LagrangianDao::new()?
    assert_eq!(&response, &expected_response);

    //let json = serde_json::to_string(&response).unwrap();
    //println!("Response function = {}", json);

    Ok(())
}
