use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianOrbCc};
use tinned::{
    Add, DotProduct, ExcitationOperator, LagMultiplier, MatrixMul, Mul, Number, OneElecMatrix,
    PertMultichain, Perturbation, Symbol, TinnedError, Trace, TwoElecMatrix, WfnParameter,
    differentiate_expr,
};

#[test]
fn orb_cc_linear_response() -> Result<(), TinnedError> {
    let freq_x = Symbol::new("omega_x");
    let pert_x = Perturbation::new("x", freq_x);
    let freq_y = Symbol::new("omega_y");
    let pert_y = Perturbation::new("y", freq_y);

    let oper_deps =
        PertMultichain::from_map(BTreeMap::from([(pert_x.clone(), 99), (pert_y.clone(), 99)]));

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

    //let cc_lagrangian = lag.get_lagrangian().clone();
    //match serde_json::to_string(&cc_lagrangian) {
    //    Ok(json) => println!("Lagrangian = {}\n", json),
    //    Err(err) => {
    //        eprintln!("Serialization of Lagrangian failed: {err}");
    //    },
    //}

    let exten_perturbations = vec![pert_x.clone(), pert_y.clone()];
    let inten_perturbations: Vec<Arc<Perturbation>> = Vec::new();

    // Using `min_wfn_exten = 0` means 2n+1 and 2n+2 rules
    let linear_response =
        lag.response_function(&exten_perturbations, &inten_perturbations, 0, false, None)?;

    //match serde_json::to_string(&linear_response) {
    //    Ok(json) => println!("L^{{xy}} = {}", json),
    //    Err(err) => {
    //        eprintln!("Serialization of L^{{xy}} failed: {err}");
    //    },
    //}

    // Get one- and two-electron density matrices
    let one_elec_density = lag.one_elec_density().clone();
    let two_elec_density = lag.two_elec_density().clone();

    //match serde_json::to_string(one_elec_density) {
    //    Ok(json) => println!("Unperturbed one-electron density matrix = {}\n", json),
    //    Err(err) => {
    //        eprintln!("Serialization of unperturbed one-electron density matrix failed: {err}");
    //    },
    //}

    //match serde_json::to_string(two_elec_density) {
    //    Ok(json) => println!("Unperturbed two-electron density matrix = {}\n", json),
    //    Err(err) => {
    //        eprintln!("Serialization of unperturbed two-electron density matrix failed: {err}");
    //    },
    //}

    // Get Brillouin equation
    let brillouin_equation = lag.brillouin_equation().clone();

    // Build reference linear response function
    let expected_linear_response = Add::new(vec![
        Trace::new(MatrixMul::new(vec![
            one_elec_density.clone(),
            differentiate_expr(&one_elec_matrix, &exten_perturbations)?,
        ])?)?,
        Trace::new(MatrixMul::new(vec![
            one_elec_density.differentiate(&pert_y)?,
            one_elec_matrix.differentiate(&pert_x)?,
        ])?)?,
        Trace::new(MatrixMul::new(vec![
            one_elec_density.differentiate(&pert_x)?,
            one_elec_matrix.differentiate(&pert_y)?,
        ])?)?,
        Trace::new(MatrixMul::new(vec![
            differentiate_expr(&one_elec_density, &exten_perturbations)?,
            one_elec_matrix.clone(),
        ])?)?,
        Trace::new(MatrixMul::new(vec![
            two_elec_density.clone(),
            differentiate_expr(&two_elec_matrix, &exten_perturbations)?,
        ])?)?,
        Trace::new(MatrixMul::new(vec![
            two_elec_density.differentiate(&pert_y)?,
            two_elec_matrix.differentiate(&pert_x)?,
        ])?)?,
        Trace::new(MatrixMul::new(vec![
            two_elec_density.differentiate(&pert_x)?,
            two_elec_matrix.differentiate(&pert_y)?,
        ])?)?,
        Trace::new(MatrixMul::new(vec![
            differentiate_expr(&two_elec_density, &exten_perturbations)?,
            two_elec_matrix.clone(),
        ])?)?,
        differentiate_expr(
            &DotProduct::new(
                brillouin_equation.clone(),
                false,
                brillouin_multiplier.clone(),
                false,
                Some(true),
            )?,
            &exten_perturbations,
        )?,
    ])?
    .eliminate(&orb_rot_parameter, &exten_perturbations, 2)?
    .eliminate(&cc_amplitude, &exten_perturbations, 2)?
    .eliminate(&brillouin_multiplier, &exten_perturbations, 1)?
    .eliminate(&cc_multiplier, &exten_perturbations, 1)?
    .substitute_zero_perturbations(None)?;

    assert_eq!(&linear_response, &expected_linear_response);

    //    // Find all (un)perturbed one- and two-electron density matrices
    //    let one_electron_densities = linear_response.find_superchains(one_elec_density);
    //    let two_electron_densities = linear_response.find_superchains(two_elec_density);
    //
    //    for (order, densities) in &one_electron_densities {
    //        println!("\norder = {}", order);
    //        for density in densities {
    //            match serde_json::to_string(density) {
    //                Ok(json) => println!("One-electron density matrix = {}\n", json),
    //                Err(err) => {
    //                    eprintln!("Serialization of one-electron density matrix failed: {err}");
    //                },
    //            }
    //        }
    //    }
    //
    //    for (order, densities) in &two_electron_densities {
    //        println!("\norder = {}", order);
    //        for density in densities {
    //            match serde_json::to_string(density) {
    //                Ok(json) => println!("Two-electron density matrix = {}\n", json),
    //                Err(err) => {
    //                    eprintln!("Serialization of two-electron density matrix failed: {err}");
    //                },
    //            }
    //        }
    //    }
    //
    //    // Find all perturbed orbital rotation parameters
    //    let orb_rot_parameters = linear_response.find_superchains(&orb_rot_parameter);
    //
    //    for (order, parameters) in &orb_rot_parameters {
    //        println!("\norder = {}", order);
    //        for parameter in parameters {
    //            match serde_json::to_string(parameter) {
    //                Ok(json) => println!("Orbital rotation parameter = {}\n", json),
    //                Err(err) => {
    //                    eprintln!("Serialization of orbital rotation parameter failed: {err}");
    //                },
    //            }
    //            let rhs_parameter = lag.linear_response_rhs(parameter.clone(), None)?;
    //            match serde_json::to_string(&rhs_parameter) {
    //                Ok(json) => println!("RHS of orbital rotation parameter = {}\n", json),
    //                Err(err) => {
    //                    eprintln!("Serialization of RHS of orbital rotation parameter failed: {err}");
    //                },
    //            }
    //        }
    //    }
    //
    //    // Find all (un)perturbed coupled-cluster amplitudes
    //    let cc_amplitudes = linear_response.find_superchains(&cc_amplitude);
    //
    //    for (order, amplitudes) in &cc_amplitudes {
    //        println!("\norder = {}", order);
    //        for amplitude in amplitudes {
    //            match serde_json::to_string(amplitude) {
    //                Ok(json) => println!("Cluster amplitude = {}\n", json),
    //                Err(err) => {
    //                    eprintln!("Serialization of cluster amplitude failed: {err}");
    //                },
    //            }
    //            let rhs_amplitude = lag.linear_response_rhs(amplitude.clone(), None)?;
    //            match serde_json::to_string(&rhs_amplitude) {
    //                Ok(json) => println!("RHS of cluster amplitude = {}\n", json),
    //                Err(err) => {
    //                    eprintln!("Serialization of RHS of cluster amplitude failed: {err}");
    //                },
    //            }
    //        }
    //    }
    //
    //    // Find all (un)perturbed coupled-cluster multipliers
    //    let cc_multipliers = linear_response.find_superchains(&cc_multiplier);
    //
    //    for (order, multipliers) in &cc_multipliers {
    //        println!("\norder = {}", order);
    //        for multiplier in multipliers {
    //            match serde_json::to_string(multiplier) {
    //                Ok(json) => println!("Cluster multiplier = {}\n", json),
    //                Err(err) => {
    //                    eprintln!("Serialization of cluster multiplier failed: {err}");
    //                },
    //            }
    //            let rhs_multiplier = lag.linear_response_rhs(multiplier.clone(), None)?;
    //            match serde_json::to_string(&rhs_multiplier) {
    //                Ok(json) => println!("RHS of cluster multiplier = {}\n", json),
    //                Err(err) => {
    //                    eprintln!("Serialization of RHS of cluster multiplier failed: {err}");
    //                },
    //            }
    //        }
    //    }
    //
    //    // Find all (un)perturbed Brillouin condition multipliers
    //    let brillouin_multipliers = linear_response.find_superchains(&brillouin_multiplier);
    //
    //    for (order, multipliers) in &brillouin_multipliers {
    //        println!("\norder = {}", order);
    //        for multiplier in multipliers {
    //            match serde_json::to_string(multiplier) {
    //                Ok(json) => println!("Brillouin condition multiplier = {}\n", json),
    //                Err(err) => {
    //                    eprintln!("Serialization of Brillouin condition multiplier failed: {err}");
    //                },
    //            }
    //            let rhs_multiplier = lag.linear_response_rhs(multiplier.clone(), None)?;
    //            match serde_json::to_string(&rhs_multiplier) {
    //                Ok(json) => println!("RHS of Brillouin condition multiplier = {}\n", json),
    //                Err(err) => {
    //                    eprintln!(
    //                        "Serialization of RHS of Brillouin condition multiplier failed: {err}"
    //                    );
    //                },
    //            }
    //        }
    //    }

    Ok(())
}
