use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianDao, SymmetrizeMode};
use tinned::{
    Add, AoTwoElecMatrix, BasisTimeEvolution, Expr, MatrixMul, Mul, Number, OneElecMatrix,
    PertMultichain, Perturbation, ResidueParameter, Symbol, TinnedError, Trace, WfnParameter,
    differentiate_expr, subtract_exprs,
};

mod common;
use common::make_perturbing_operator;

// First-order residue of the linear response function, equation (286),
// J. Chem. Phys. 129, 214108 (2008)
#[test]
fn dao_first_order_lr_residue() -> Result<(), TinnedError> {
    let freq_a = Symbol::new("omega_a");
    let pert_a = Perturbation::new("a", freq_a);
    let freq_b = Symbol::new("omega_b");
    let pert_b = Perturbation::new("b", freq_b);

    let density_matrix = WfnParameter::builder("D").build()?;

    let oper_deps = PertMultichain::from_map(BTreeMap::from([
        (pert_a.clone(), u32::MAX),
        (pert_b.clone(), u32::MAX),
    ]));

    let overlap_matrix = OneElecMatrix::builder("S").dependencies(oper_deps.clone()).build()?;
    let one_elec_hamiltonian =
        OneElecMatrix::builder("h").dependencies(oper_deps.clone()).build()?;
    let perturbing_oper =
        OneElecMatrix::builder("V").is_perturbing(true).dependencies(oper_deps.clone()).build()?;
    let t_matrix = BasisTimeEvolution::builder(oper_deps.clone()).build()?;

    let one_elec_opers = vec![one_elec_hamiltonian, perturbing_oper, t_matrix];

    let two_elec_operator =
        AoTwoElecMatrix::builder("G", density_matrix.clone()).dependencies(oper_deps).build()?;

    let lag = LagrangianDao::new(
        pert_a.clone(),
        density_matrix.clone(),
        Some(overlap_matrix.clone()),
        &one_elec_opers,
        Some(two_elec_operator),
        None,
        None,
        None,
        Some(SymmetrizeMode::Never),
        None,
    )?;

    // Perturbation `a` is a bit special: it was already set in Lagrangian and
    // we should not specify it in external or internal perturbations here
    let exten_perturbations = vec![pert_b.clone()];
    let inten_perturbations: Vec<Arc<Perturbation>> = Vec::new();

    let excited_state = WfnParameter::builder("Xq").build()?;
    let residue_relations: HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)> =
        HashMap::from([(excited_state.clone(), (true, vec![pert_b.clone()]))]);

    // Using `min_wfn_extern = 3` removes all Lagrangian multipliers
    let residue = lag.residue(
        &exten_perturbations,
        &inten_perturbations,
        3,
        &residue_relations,
        false,
        None,
    )?;

    // Residue density matrix
    let density_b = density_matrix.differentiate(pert_b.clone())?;
    let res_density_b =
        ResidueParameter::builder(vec![pert_b.clone()], excited_state, density_b.clone())
            .positive_frequency(true)
            .build()?;

    // Reference residue, equation (286)
    let generalized_energy_ab = lag.generalized_energy_a().differentiate(pert_b.clone())?;
    let overlap_a = overlap_matrix.differentiate(pert_a)?;
    let general_ew_density =
        lag.general_ew_density().expect("Expect generalized energy-weighted density matrix");
    let general_ew_density_b = general_ew_density.differentiate(pert_b.clone())?;

    let expected_residue = subtract_exprs(
        generalized_energy_ab,
        Trace::new(MatrixMul::new(vec![overlap_a, general_ew_density_b])?)?,
    )?
    .eliminate(density_matrix, &exten_perturbations, 2)?
    .substitute_zero_perturbations(None)?
    .retain_one(&density_b, true)?
    .replace_one(&density_b, res_density_b.clone(), true)?;

    //match serde_json::to_string(&residue) {
    //    Ok(json) => println!("Residue = {}", json),
    //    Err(e) => {
    //        eprintln!("Display: {}", e);
    //        eprintln!("Debug: {:?}", e);
    //        eprintln!("Category: {:?}", e.classify());
    //        eprintln!("Line: {}, Column: {}", e.line(), e.column());
    //    }
    //}

    //match serde_json::to_string(&expected_residue) {
    //    Ok(json) => println!("Expected residue = {}", json),
    //    Err(e) => {
    //        eprintln!("Display: {}", e);
    //        eprintln!("Debug: {:?}", e);
    //        eprintln!("Category: {:?}", e.classify());
    //        eprintln!("Line: {}, Column: {}", e.line(), e.column());
    //    }
    //}

    assert_eq!(&residue, &expected_residue);

    // Get the right-hand side of the linear response equation
    let density_part = WfnParameter::builder("D_P").build()?;
    let rhs = lag.linear_response_rhs(&res_density_b, density_part.clone(), None)?;

    //let json_rhs = serde_json::to_string(&rhs).unwrap();
    //println!("RHS = {}", json_rhs);

    // Reference RHS, equation (289), J. Chem. Phys. 129, 214108 (2008)
    let expected_rhs = lag
        .tdscf_equation()
        .differentiate(pert_b)?
        .substitute_zero_perturbations(None)?
        .replace_one(&density_b, density_part, true)?;

    //let json_expected_rhs = serde_json::to_string(&expected_rhs).unwrap();
    //println!("Expected RHS = {}", json_expected_rhs);

    assert_eq!(&rhs, &expected_rhs);

    Ok(())
}

// Magnetic circular dichroism
// J. Chem. Phys. 135, 024112 (2011)
#[test]
fn lao_mcd() -> Result<(), TinnedError> {
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

    let one_elec_opers = vec![one_elec_hamiltonian, perturbing_oper, t_matrix];

    let two_elec_operator = AoTwoElecMatrix::builder("G", density_matrix.clone())
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
    let exten_perturbations = vec![pert_b, pert_c.clone()];
    let inten_perturbations = Vec::new();

    let excited_state_a = WfnParameter::builder("X-j").build()?;
    let excited_state_c = WfnParameter::builder("X+j").build()?;
    let residue_relations: HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)> =
        HashMap::from([
            (excited_state_a.clone(), (false, vec![pert_a.clone()])),
            (excited_state_c.clone(), (true, vec![pert_c.clone()])),
        ]);

    // Using `min_wfn_extern = 0` means it will be determined by SymResponse as
    // the next integer of the floor function of the half number of extensive
    // perturbations
    let residue = lag.residue(
        &exten_perturbations,
        &inten_perturbations,
        0,
        &residue_relations,
        false,
        None,
    )?;

    //let json_residue = serde_json::to_string(&residue).unwrap();
    //println!("Reside = {}", json_residue);

    // F^{bc}
    let fock_matrix_bc = differentiate_expr(lag.fock_matrix(), &exten_perturbations)?;

    // Residue density matrices
    let density_a = density_matrix.differentiate(pert_a.clone())?;
    let density_c = density_matrix.differentiate(pert_c.clone())?;
    let res_density_a = ResidueParameter::builder(vec![pert_a], excited_state_a, density_a.clone())
        .positive_frequency(false)
        .build()?;
    let res_density_c = ResidueParameter::builder(vec![pert_c], excited_state_c, density_c.clone())
        .positive_frequency(true)
        .build()?;
    let density_ac_sets =
        vec![HashSet::from([density_a.clone()]), HashSet::from([density_c.clone()])];

    let res_density_ac_map: HashMap<Arc<dyn Expr>, Arc<dyn Expr>> =
        HashMap::from([(density_a.clone(), res_density_a), (density_c, res_density_c)]);

    // Reference double residue, equation (B27)
    let generalized_energy_abc =
        differentiate_expr(lag.generalized_energy_a(), &exten_perturbations)?
            .remove_one(&fock_matrix_bc)?;
    let tdscf_equation_bc = differentiate_expr(lag.tdscf_equation(), &exten_perturbations)?
        .remove_one(&fock_matrix_bc)?;
    let idempotency_bc =
        differentiate_expr(lag.idempotency(), &exten_perturbations)?.remove_one(&fock_matrix_bc)?;
    let expected_residue = Add::new(vec![
        generalized_energy_abc,
        //  F^{bc} * D^{a}
        Trace::new(MatrixMul::new(vec![fock_matrix_bc, density_a])?)?,
        Mul::new(vec![
            Number::minus_one(),
            Add::new(vec![
                Trace::new(MatrixMul::new(vec![
                    lag.tdscf_multiplier().clone(),
                    tdscf_equation_bc,
                ])?)?,
                Trace::new(MatrixMul::new(vec![lag.idemp_multiplier().clone(), idempotency_bc])?)?,
            ])?,
        ])?,
    ])?
    .eliminate(density_matrix, &exten_perturbations, 2)?
    .substitute_zero_perturbations(None)?
    .retain_all(&density_ac_sets, true)?
    .replace_all(&res_density_ac_map, true)?;

    //let json_expected_residue = serde_json::to_string(&expected_residue).unwrap();
    //println!("Expected residue = {}", json_expected_residue);

    assert_eq!(&residue, &expected_residue);

    Ok(())
}

// Two-photon transition matrix element between the ground state and the excited state
// Equation (63), J. Chem. Phys. 134, 214104 (2011)
#[test]
fn dao_2p_tme() -> Result<(), TinnedError> {
    let density_matrix = WfnParameter::builder("D").build()?;

    let overlap_matrix = OneElecMatrix::builder("S").build()?;
    let one_elec_hamiltonian = OneElecMatrix::builder("h").build()?;

    let (pert_a, mu_a) = make_perturbing_operator("mu_a", "a", "omega_a")?;
    let (pert_b, mu_b) = make_perturbing_operator("mu_b", "b", "omega_b")?;
    let (pert_c, mu_c) = make_perturbing_operator("mu_c", "c", "omega_c")?;
    let (pert_d, mu_d) = make_perturbing_operator("mu_d", "d", "omega_d")?;

    let one_elec_opers = vec![one_elec_hamiltonian, mu_a, mu_b, mu_c, mu_d];

    let two_elec_operator = AoTwoElecMatrix::builder("G", density_matrix.clone()).build()?;

    let lag = LagrangianDao::new(
        pert_a.clone(),
        density_matrix.clone(),
        Some(overlap_matrix),
        &one_elec_opers,
        Some(two_elec_operator),
        None,
        None,
        None,
        Some(SymmetrizeMode::Never),
        None,
    )?;

    // Perturbation `a` is a bit special: it was already set in Lagrangian and
    // we should not specify it in external or internal perturbations here
    let exten_perturbations = vec![pert_b.clone(), pert_c.clone(), pert_d.clone()];
    let inten_perturbations: Vec<Arc<Perturbation>> = Vec::new();

    let excited_state = WfnParameter::builder("Xn").build()?;
    let residue_relations: HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)> =
        HashMap::from([(excited_state, (true, vec![pert_c.clone(), pert_d.clone()]))]);
    // The following is TPA between excited states, or excited state absorption
    //
    //let excited_state_f = WfnParameter::builder("Xf").build()?;
    //let excited_state_g = WfnParameter::builder("Xg").build()?;
    //let residue_relations: HashMap<Arc<dyn Expr>, (bool, Vec<Arc<Perturbation>>)> = HashMap::from([
    //    (excited_state_f, (false, vec![pert_c.clone()])),
    //    (excited_state_g, (true, vec![pert_d.clone()])),
    //]);

    // Equation (63) is the trace of product between the right-hand side of
    // X^{ab} and X^{cd}. The only possible response function expression is
    // equation (240) in J. Chem. Phys. 129, 214108 (2008).
    let residue = lag.residue(
        &exten_perturbations,
        &inten_perturbations,
        3,
        &residue_relations,
        false,
        None,
    )?;

    //FIXME: Not sure the correctness of equation (63), needs to figure out and compare
    println!("Reside = {}", residue);

    Ok(())
}
