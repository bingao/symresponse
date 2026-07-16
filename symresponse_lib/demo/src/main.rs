use anyhow::Context;
use std::collections::BTreeMap;
use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianOrbCc};
use tinned::{
    ExcitationOperator, LagMultiplier, OneElecMatrix, PertMultichain, Perturbation, Symbol,
    TwoElecMatrix, WfnParameter, walk_expr_postorder,
};

mod code_generator;
use code_generator::CodeGenerator;

fn main() -> anyhow::Result<()> {
    let freq_x = Symbol::new("omega_x");
    let pert_x = Perturbation::new("x", freq_x);
    let freq_y = Symbol::new("omega_y");
    let pert_y = Perturbation::new("y", freq_y);

    let oper_deps = PertMultichain::from_map(BTreeMap::from([
        (pert_x.clone(), u32::MAX),
        (pert_y.clone(), u32::MAX),
    ]));

    let one_elec_matrix = OneElecMatrix::builder("h").dependencies(oper_deps.clone()).build()?;
    let single_excitation_operator = ExcitationOperator::new("E_{pq}");
    let two_elec_matrix = TwoElecMatrix::builder("g").dependencies(oper_deps).build()?;
    let double_excitation_operator = ExcitationOperator::new("e_{pqrs}");
    let cc_amplitude = WfnParameter::builder("t").build()?;
    let cc_excitation_operator = ExcitationOperator::new("tau");
    let cc_multiplier = LagMultiplier::builder("tbar").build()?;
    // Undifferentiated orbital rotation parameter `orb_rot_parameter` should
    // be zero, so `is_perturbing` should be `true` here
    let orb_rot_parameter = WfnParameter::builder("kappa").is_perturbing(true).build()?;
    // Orbital rotation generator, E_{pq}-E_{qp}
    let orb_rot_generator = ExcitationOperator::new("E-");
    let brillouin_multiplier = LagMultiplier::builder("kbar").build()?;

    // Build Lagrangian
    let lag = LagrangianOrbCc::new(
        one_elec_matrix.clone(),
        single_excitation_operator,
        two_elec_matrix.clone(),
        double_excitation_operator,
        cc_amplitude.clone(),
        cc_excitation_operator.clone(),
        cc_multiplier.clone(),
        orb_rot_parameter.clone(),
        orb_rot_generator.clone(),
        brillouin_multiplier.clone(),
    )?;

    let exten_perturbations = vec![pert_x.clone(), pert_y.clone()];
    let inten_perturbations: Vec<Arc<Perturbation>> = Vec::new();

    // Using `min_wfn_exten_order = 0` means 2n+1 and 2n+2 rules
    let linear_response =
        lag.response_function(&exten_perturbations, &inten_perturbations, 0, false, None)?;

    match serde_json::to_string(&linear_response) {
        Ok(json) => println!("L^{{xy}} = {}\n", json),
        Err(err) => eprintln!("Serialization of L^{{xy}} failed: {err}"),
    }

    // Generate eT Fortran subroutine for the calculation of response function
    let mut code_generator = CodeGenerator::new().context("Failed to create code generator")?;

    //FIXME: we should change `one_elec_matrix` to `linear_response` after
    //`CodeGenerator` is completely implemented
    match walk_expr_postorder(&one_elec_matrix, &mut code_generator) {
        Ok(()) => println!("\nResponse function processed\n"),
        Err(err) => eprintln!("\nFailed to process response function: {err}"),
    }

    //FIXME: we may also need to consider how to handle unknown perturbed
    //parameters in the generated code
    code_generator
        .print_et_code("symresponse_demo")
        .context("Failed to generate Fortran subroutines for eT")?;

    code_generator.reset();

    // Find all perturbed orbital rotation parameters, unperturbed one is zero
    let orb_rot_parameters = linear_response.find_all(&orb_rot_parameter);

    for (order, parameters) in &orb_rot_parameters {
        println!("\nOrder of (un)perturbed orbital rotation parameters {}\n", *order);

        for parameter in parameters {
            match serde_json::to_string(parameter) {
                Ok(json) => println!("Orbital rotation parameter = {}\n", json),
                Err(err) => eprintln!("Serialization of orbital rotation parameter failed: {err}"),
            }

            // Get the right-hand side of the response equation of perturbed
            // orbital rotation parameters
            if *order >= 1 {
                let rhs_parameter = lag.linear_response_rhs(parameter, None)?;

                match serde_json::to_string(&rhs_parameter) {
                    Ok(json) => println!("RHS of orbital rotation parameter = {}\n", json),
                    Err(err) => eprintln!(
                        "Serialization of RHS of orbital rotation parameter failed: {err}"
                    ),
                }
            }
        }
    }

    // Find all (un)perturbed coupled-cluster amplitudes
    let cc_amplitudes = linear_response.find_all(&cc_amplitude);

    for (order, amplitudes) in &cc_amplitudes {
        println!("\nOrder of (un)perturbed coupled-cluster amplitudes {}\n", *order);

        for amplitude in amplitudes {
            match serde_json::to_string(amplitude) {
                Ok(json) => println!("Coupled-cluster amplitude = {}\n", json),
                Err(err) => eprintln!("Serialization of coupled-cluster amplitude failed: {err}"),
            }

            // Get the right-hand side of the response equation of perturbed
            // coupled-cluster amplitude
            if *order >= 1 {
                let rhs_amplitude = lag.linear_response_rhs(amplitude, None)?;

                match serde_json::to_string(&rhs_amplitude) {
                    Ok(json) => println!("RHS of coupled-cluster amplitude = {}\n", json),
                    Err(err) => {
                        eprintln!("Serialization of RHS of coupled-cluster amplitude failed: {err}")
                    },
                }
            }
        }
    }

    // Find all (un)perturbed coupled-cluster multipliers
    let cc_multipliers = linear_response.find_all(&cc_multiplier);

    for (order, multipliers) in &cc_multipliers {
        println!("\nOrder of (un)perturbed coupled-cluster multipliers {}\n", *order);

        for multiplier in multipliers {
            match serde_json::to_string(multiplier) {
                Ok(json) => println!("Coupled-cluster multiplier = {}\n", json),
                Err(err) => eprintln!("Serialization of coupled-cluster multiplier failed: {err}"),
            }

            let rhs_multiplier = lag.linear_response_rhs(multiplier, None)?;

            match serde_json::to_string(&rhs_multiplier) {
                Ok(json) => println!("RHS of coupled-cluster multiplier = {}\n", json),
                Err(err) => {
                    eprintln!("Serialization of RHS of coupled-cluster multiplier failed: {err}")
                },
            }
        }
    }

    // Find all (un)perturbed Brillouin condition multipliers
    let brillouin_multipliers = linear_response.find_all(&brillouin_multiplier);

    for (order, multipliers) in &brillouin_multipliers {
        println!("\nOrder of (un)perturbed Brillouin condition multipliers {}\n", *order);

        for multiplier in multipliers {
            match serde_json::to_string(multiplier) {
                Ok(json) => println!("Brillouin condition multiplier = {}\n", json),
                Err(err) => {
                    eprintln!("Serialization of Brillouin condition multiplier failed: {err}")
                },
            }

            let rhs_multiplier = lag.linear_response_rhs(multiplier, None)?;

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
