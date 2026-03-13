use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianOrbCc};
use tinned::{
    LagMultiplier, Mul, Number, OneElecOperator, PertMultichain, Perturbation, Symbol, TinnedError,
    TwoElecOperator, WfnParameter,
};

#[test]
fn orb_cc_linear() -> Result<(), TinnedError> {
    let freq_a = Symbol::new("omega_a");
    let pert_a = Perturbation::new("a", freq_a);
    let freq_b = Symbol::new("omega_b");
    let pert_b = Perturbation::new("b", freq_b);
    let freq_c = Symbol::new("omega_c");
    let pert_c = Perturbation::new("c", freq_c);

    let oper_deps = PertMultichain::from_map(BTreeMap::from([
        (pert_a.clone(), 9),
        (pert_b.clone(), 9),
        (pert_c.clone(), 9),
    ]));

    let one_elec_operator =
        OneElecOperator::builder("h").dependencies(oper_deps.clone()).build()?;
    //FIXME: `two_elec_operator` is simply an operator without dependency on wave function parameter
    let two_elec_operator =
        OneElecOperator::builder("g").dependencies(oper_deps.clone()).build()?;
    let cc_amplitude = WfnParameter::builder("t").build()?;
    let cc_excitation_operator = OneElecOperator::builder("tau").build()?;
    let cc_multiplier = LagMultiplier::builder("tbar").build()?;
    let orb_rotation_parameter = WfnParameter::builder("kappa").build()?;
    // Orbital rotation generators, $\hat{a}^{\dagger)_{r}\hat{a}_{s}$
    let orb_rotation_generator = OneElecOperator::builder("E-").build()?;
    let brillouin_multiplier = LagMultiplier::builder("kbar").build()?;

    let lag = LagrangianOrbCc::new(
        one_elec_operator.clone(),
        two_elec_operator.clone(),
        cc_amplitude.clone(),
        cc_excitation_operator.clone(),
        cc_multiplier.clone(),
        orb_rotation_parameter.clone(),
        orb_rotation_generator.clone(),
        brillouin_multiplier.clone(),
    )?;

    let mut result = lag.get_lagrangian().clone();
    match serde_json::to_string(&result) {
        Ok(json) => println!("Lagrangian = {}\n", json),
        Err(err) => {
            eprintln!("Serialization of Lagrangian failed: {err}");
        },
    }

    let exten_perturbations = vec![pert_a.clone(), pert_b.clone(), pert_c.clone()];
    let inten_perturbations: Vec<Arc<Perturbation>> = Vec::new();

    // Using `min_wfn_extern = 3` removes all Lagrangian multipliers
    result = lag.response_function(&exten_perturbations, &inten_perturbations, 3, false, None)?;
    //match serde_json::to_string(&result) {
    //    Ok(json) => println!("L^{{abc}} = {}", json),
    //    Err(err) => {
    //        eprintln!("Serialization of L^{{abc}} failed: {err}");
    //    },
    //}

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

    Ok(())
}
