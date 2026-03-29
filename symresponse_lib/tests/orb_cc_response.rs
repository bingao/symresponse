use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianOrbCc};
use tinned::{
    ExcitationOperator, LagMultiplier, Mul, Number, OneElecMatrix, PertMultichain, Perturbation,
    Symbol, TinnedError, TwoElecMatrix, WfnParameter,
};

#[test]
fn orb_cc_quadratic() -> Result<(), TinnedError> {
    let freq_x = Symbol::new("omega_x");
    let pert_x = Perturbation::new("x", freq_x);
    let freq_y = Symbol::new("omega_y");
    let pert_y = Perturbation::new("y", freq_y);
    let freq_z = Symbol::new("omega_z");
    let pert_z = Perturbation::new("z", freq_z);

    let oper_deps = PertMultichain::from_map(BTreeMap::from([
        (pert_x.clone(), 99),
        (pert_y.clone(), 99),
        (pert_z.clone(), 99),
    ]));

    let one_elec_matrix = OneElecMatrix::builder("h").dependencies(oper_deps.clone()).build()?;
    let single_excitation_operator = ExcitationOperator::new("E_{pq}");
    let two_elec_matrix = TwoElecMatrix::builder("g").dependencies(oper_deps.clone()).build()?;
    let double_excitation_operator = ExcitationOperator::new("e_{pqrs}");
    let cc_amplitude = WfnParameter::builder("t").build()?;
    let cc_excitation_operator = ExcitationOperator::new("tau");
    let cc_multiplier = LagMultiplier::builder("tbar").build()?;
    // Undifferentiated orbital rotation parameter `orb_rot_parameter` should
    // be zero, so `is_perturbing` should be `true` here
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

    //let mut result = lag.get_lagrangian().clone();
    //match serde_json::to_string(&result) {
    //    Ok(json) => println!("Lagrangian = {}\n", json),
    //    Err(err) => {
    //        eprintln!("Serialization of Lagrangian failed: {err}");
    //    },
    //}

    let exten_perturbations = vec![pert_x.clone(), pert_y.clone(), pert_z.clone()];
    let inten_perturbations: Vec<Arc<Perturbation>> = Vec::new();

    // Using `min_wfn_exten = 0` means 2n+1 and 2n+2 rules
    let result =
        lag.response_function(&exten_perturbations, &inten_perturbations, 0, false, None)?;
    match serde_json::to_string(&result) {
        Ok(json) => println!("L^{{xyz}} = {}", json),
        Err(err) => {
            eprintln!("Serialization of L^{{xyz}} failed: {err}");
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

    // Find all perturbed orbital rotation parameters
    let orb_rot_parameters = result.find_superchains(&orb_rot_parameter);

    for (order, parameters) in &orb_rot_parameters {
        println!("\norder = {}", order);
        for parameter in parameters {
            match serde_json::to_string(parameter) {
                Ok(json) => println!("Orbital rotation parameter = {}\n", json),
                Err(err) => {
                    eprintln!("Serialization of orbital rotation parameter failed: {err}");
                },
            }
            let rhs_parameter = lag.linear_response_rhs(parameter.clone())?;
            match serde_json::to_string(&rhs_parameter) {
                Ok(json) => println!("RHS of orbital rotation parameter = {}\n", json),
                Err(err) => {
                    eprintln!("Serialization of RHS of orbital rotation parameter failed: {err}");
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
            let rhs_amplitude = lag.linear_response_rhs(amplitude.clone())?;
            match serde_json::to_string(&rhs_amplitude) {
                Ok(json) => println!("RHS of cluster amplitude = {}\n", json),
                Err(err) => {
                    eprintln!("Serialization of RHS of cluster amplitude failed: {err}");
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
            let rhs_multiplier = lag.linear_response_rhs(multiplier.clone())?;
            match serde_json::to_string(&rhs_multiplier) {
                Ok(json) => println!("RHS of cluster multiplier = {}\n", json),
                Err(err) => {
                    eprintln!("Serialization of RHS of cluster multiplier failed: {err}");
                },
            }
        }
    }

    // Find all (un)perturbed Brillouin condition multipliers
    let brillouin_multipliers = result.find_superchains(&brillouin_multiplier);

    for (order, multipliers) in &brillouin_multipliers {
        println!("\norder = {}", order);
        for multiplier in multipliers {
            match serde_json::to_string(multiplier) {
                Ok(json) => println!("Brillouin condition multiplier = {}\n", json),
                Err(err) => {
                    eprintln!("Serialization of Brillouin condition multiplier failed: {err}");
                },
            }
            let rhs_multiplier = lag.linear_response_rhs(multiplier.clone())?;
            match serde_json::to_string(&rhs_multiplier) {
                Ok(json) => println!("RHS of Brillouin condition multiplier = {}\n", json),
                Err(err) => {
                    eprintln!(
                        "Serialization of RHS of Brillouin condition multiplier failed: {err}"
                    );
                },
            }
        }
    }

    Ok(())
}
