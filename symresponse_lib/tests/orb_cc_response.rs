use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianOrbCc};
use tinned::{
    ExcitationOperator, LagMultiplier, Mul, Number, OneElecMatrix, PertMultichain, Perturbation,
    Symbol, TinnedError, TwoElecMatrix, WfnParameter,
};

#[test]
fn orb_cc_quadratic() -> Result<(), TinnedError> {
    let freq_a = Symbol::new("omega_a");
    let pert_a = Perturbation::new("a", freq_a);
    let freq_b = Symbol::new("omega_b");
    let pert_b = Perturbation::new("b", freq_b);
    let freq_c = Symbol::new("omega_c");
    let pert_c = Perturbation::new("c", freq_c);

    let oper_deps = PertMultichain::from_map(BTreeMap::from([
        (pert_a.clone(), 99),
        (pert_b.clone(), 99),
        (pert_c.clone(), 99),
    ]));

    let one_elec_matrix = OneElecMatrix::builder("h").dependencies(oper_deps.clone()).build()?;
    let single_excitation_operator = ExcitationOperator::new("E_{pq}");
    let two_elec_matrix = TwoElecMatrix::builder("g").dependencies(oper_deps.clone()).build()?;
    let double_excitation_operator = ExcitationOperator::new("e_{pqrs}");
    let cc_amplitude = WfnParameter::builder("t").build()?;
    let cc_excitation_operator = ExcitationOperator::new("tau");
    let cc_multiplier = LagMultiplier::builder("tbar").build()?;
    let orb_rot_parameter = WfnParameter::builder("kappa").is_perturbing(true).build()?;
    // Orbital rotation generator, $E_{pq}-E_{qp}$
    let orb_rot_generator = ExcitationOperator::new("E-");
    let brillouin_multiplier = LagMultiplier::builder("kbar").build()?;

    let lag = LagrangianOrbCc::new(
        one_elec_matrix.clone(),
        single_excitation_operator.clone(),
        two_elec_matrix.clone(),
        double_excitation_operator.clone(),
        cc_amplitude.clone(),
        cc_excitation_operator.clone(),
        cc_multiplier.clone(),
        orb_rot_parameter.clone(),
        orb_rot_generator.clone(),
        brillouin_multiplier.clone(),
    )?;

    let mut result = lag.get_lagrangian().clone();
    match serde_json::to_string(&result) {
        Ok(json) => println!("Lagrangian = {}\n", json),
        Err(err) => {
            eprintln!("Serialization of Lagrangian failed: {err}");
        },
    }

    let exten_perturbations = vec![pert_a.clone(), pert_a.clone(), pert_a.clone()];
    let inten_perturbations: Vec<Arc<Perturbation>> = Vec::new();

    // Using `min_wfn_exten = 0` means 2n+1 and 2n+2 rules
    result = lag.response_function(&exten_perturbations, &inten_perturbations, 0, false, None)?;
    match serde_json::to_string(&result) {
        Ok(json) => println!("L^{{aaa}} = {}", json),
        Err(err) => {
            eprintln!("Serialization of L^{{aaa}} failed: {err}");
        },
    }

    // Get one- and two-electron density matrices
    let one_electron_density = lag.one_electron_density();
    let two_electron_density = lag.two_electron_density();

    match serde_json::to_string(one_electron_density) {
        Ok(json) => println!("Unperturbed one-electron density matrix = {}\n", json),
        Err(err) => {
            eprintln!("Serialization of unperturbed one-electron density matrix failed: {err}");
        },
    }

    match serde_json::to_string(two_electron_density) {
        Ok(json) => println!("Unperturbed two-electron density matrix = {}\n", json),
        Err(err) => {
            eprintln!("Serialization of unperturbed two-electron density matrix failed: {err}");
        },
    }

    // Find all (un)perturbed one- and two-electron density matrices
    let one_electron_densities = result.find_superchains(one_electron_density);
    let two_electron_densities = result.find_superchains(two_electron_density);

    for (order, densities) in &one_electron_densities {
        println!("\norder = {}", order);
        for density in densities {
            match serde_json::to_string(density) {
                Ok(json) => println!("One-electron density matrix = {}\n", json),
                Err(err) => {
                    eprintln!("Serialization of one-electron density matrix failed: {err}");
                },
            }
        }
    }

    for (order, densities) in &two_electron_densities {
        println!("\norder = {}", order);
        for density in densities {
            match serde_json::to_string(density) {
                Ok(json) => println!("Two-electron density matrix = {}\n", json),
                Err(err) => {
                    eprintln!("Serialization of two-electron density matrix failed: {err}");
                },
            }
        }
    }

    // Find all (un)perturbed coupled-cluster amplitudes
    let cc_amplitudes = result.find_superchains(&cc_amplitude);

    for (order, amplitudes) in &cc_amplitudes {
        println!("\norder = {}", order);
        for amplitude in amplitudes {
            match serde_json::to_string(amplitude) {
                Ok(json) => println!("Cluster amplitude = {}\n", json),
                Err(err) => {
                    eprintln!("Serialization of cluster amplitude failed: {err}");
                },
            }
        }
    }

    // Find all (un)perturbed coupled-cluster multipliers
    let cc_multipliers = result.find_superchains(&cc_multiplier);

    for (order, multipliers) in &cc_multipliers {
        println!("\norder = {}", order);
        for multiplier in multipliers {
            match serde_json::to_string(multiplier) {
                Ok(json) => println!("Cluster multiplier = {}\n", json),
                Err(err) => {
                    eprintln!("Serialization of cluster multiplier failed: {err}");
                },
            }
        }
    }

    Ok(())
}
