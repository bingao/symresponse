use std::collections::BTreeMap;
use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianOrbCc};
use tinned::{
    Add, AdjointMap, AdjointMode, DotProduct, ExcitationOperator, ExpAdjointMap,
    HermitianTranspose, LagMultiplier, MatrixAdd, MatrixMul, Number, OneElecMatrix, PertMultichain,
    Perturbation, SubExpr, Symbol, TinnedError, Trace, TwoElecMatrix, WfnParameter,
    differentiate_expr, downcast_from_arc, is_expr_type,
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

    // Get one- and two-electron density matrices. Note that they are not
    // evaluated at zero perturbation strength
    let one_elec_density = lag.one_elec_density().clone();
    let two_elec_density = lag.two_elec_density().clone();

    //match serde_json::to_string(&one_elec_density) {
    //    Ok(json) => println!("Unperturbed one-electron density matrix = {}\n", json),
    //    Err(err) => {
    //        eprintln!("Serialization of unperturbed one-electron density matrix failed: {err}");
    //    },
    //}

    //match serde_json::to_string(&two_elec_density) {
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

    // Find all (un)perturbed one-electron density matrices
    let one_elec_densities = linear_response.find_superchains(&one_elec_density);

    assert!(one_elec_densities.len() == 3);

    // Check all (un)perturbed one-electron density matrices
    for (order, densities) in &one_elec_densities {
        assert!(*order <= 2);
        match *order {
            0 => {
                assert!(densities.len() == 1);
                assert_eq!(
                    &one_elec_density.substitute_zero_perturbations(None)?,
                    densities.iter().next().unwrap()
                );
            },
            _ => {
                for density in densities {
                    assert!(is_expr_type::<SubExpr>(density));
                    let perturbations = downcast_from_arc::<SubExpr>(density).unwrap().derivative();
                    let one_density_deriv = differentiate_expr(&one_elec_density, perturbations)?;
                    assert_eq!(
                        &one_density_deriv
                            .eliminate(&orb_rot_parameter, &exten_perturbations, 2)?
                            .eliminate(&cc_amplitude, &exten_perturbations, 2)?
                            .eliminate(&brillouin_multiplier, &exten_perturbations, 1)?
                            .eliminate(&cc_multiplier, &exten_perturbations, 1)?
                            .substitute_zero_perturbations(None)?,
                        density
                    );
                    //match serde_json::to_string(density) {
                    //    Ok(json) => println!("One-electron density matrix = {}\n", json),
                    //    Err(err) => {
                    //        eprintln!("Serialization of one-electron density matrix failed: {err}");
                    //    },
                    //}
                }
            },
        }
    }

    // Find all (un)perturbed two-electron density matrices
    let two_elec_densities = linear_response.find_superchains(&two_elec_density);

    assert!(two_elec_densities.len() == 3);

    // Check all (un)perturbed two-electron density matrices
    for (order, densities) in &two_elec_densities {
        assert!(*order <= 2);
        match *order {
            0 => {
                assert!(densities.len() == 1);
                assert_eq!(
                    &two_elec_density.substitute_zero_perturbations(None)?,
                    densities.iter().next().unwrap()
                );
            },
            _ => {
                for density in densities {
                    assert!(is_expr_type::<SubExpr>(density));
                    let perturbations = downcast_from_arc::<SubExpr>(density).unwrap().derivative();
                    let two_density_deriv = differentiate_expr(&two_elec_density, perturbations)?;
                    assert_eq!(
                        &two_density_deriv
                            .eliminate(&orb_rot_parameter, &exten_perturbations, 2)?
                            .eliminate(&cc_amplitude, &exten_perturbations, 2)?
                            .eliminate(&brillouin_multiplier, &exten_perturbations, 1)?
                            .eliminate(&cc_multiplier, &exten_perturbations, 1)?
                            .substitute_zero_perturbations(None)?,
                        density
                    );
                    //match serde_json::to_string(density) {
                    //    Ok(json) => println!("Two-electron density matrix = {}\n", json),
                    //    Err(err) => {
                    //        eprintln!("Serialization of two-electron density matrix failed: {err}");
                    //    },
                    //}
                }
            },
        }
    }

    // Find all perturbed orbital rotation parameters, unperturbed one is zero
    let orb_rot_parameters = linear_response.find_superchains(&orb_rot_parameter);

    assert!(orb_rot_parameters.len() == 1);

    for (order, parameters) in &orb_rot_parameters {
        assert!(*order == 1);
        assert!(parameters.len() == 2);

        let orb_rot_parameter_x = orb_rot_parameter.differentiate(&pert_x)?;
        let orb_rot_parameter_y = orb_rot_parameter.differentiate(&pert_y)?;

        for parameter in parameters {
            assert!(
                parameter.clone() == orb_rot_parameter_x.clone()
                    || parameter.clone() == orb_rot_parameter_y.clone()
            );
        }

        // Check the right-hand side of the response equation of perturbed
        // orbital rotation parameters
        let rhs_parameter_x = lag.linear_response_rhs(&orb_rot_parameter_x, None)?;
        // Equation (51), J. Chem. Phys. 92, 4924-4940 (1990)
        assert_eq!(
            &rhs_parameter_x,
            &MatrixMul::new(vec![
                Number::minus_one(),
                AdjointMap::new(
                    vec![orb_rot_generator.clone()],
                    MatrixAdd::new(vec![one_elec_matrix.clone(), two_elec_matrix.clone()])?
                        .differentiate(&pert_x)?,
                    Some(true),
                    Some(AdjointMode::Symmetrized)
                )?
            ])?
        );
    }

    // Find all (un)perturbed coupled-cluster amplitudes
    let cc_amplitudes = linear_response.find_superchains(&cc_amplitude);

    assert!(cc_amplitudes.len() == 2);

    let kappa_operator_x = lag.kappa_operator().differentiate(&pert_x)?;
    // Equation (35), J. Chem. Phys. 92, 4924-4940 (1990)
    let j_operator_x = MatrixAdd::new(vec![
        one_elec_matrix.differentiate(&pert_x)?,
        two_elec_matrix.differentiate(&pert_x)?,
        AdjointMap::new(
            vec![kappa_operator_x.clone()],
            one_elec_matrix.clone(),
            Some(true),
            Some(AdjointMode::Symmetrized),
        )?,
        AdjointMap::new(
            vec![kappa_operator_x.clone()],
            two_elec_matrix.clone(),
            Some(true),
            Some(AdjointMode::Symmetrized),
        )?,
    ])?;

    for (order, amplitudes) in &cc_amplitudes {
        assert!(*order == 0 || *order == 1);

        if *order == 0 {
            assert!(amplitudes.len() == 1);
            assert_eq!(
                amplitudes.iter().next().expect("Unperturbed coupled-cluster amplitudes"),
                &cc_amplitude
            );
        } else {
            assert!(amplitudes.len() == 2);

            let cc_amplitude_x = cc_amplitude.differentiate(&pert_x)?;
            let cc_amplitude_y = cc_amplitude.differentiate(&pert_y)?;

            for amplitude in amplitudes {
                assert!(
                    amplitude.clone() == cc_amplitude_x.clone()
                        || amplitude.clone() == cc_amplitude_y.clone()
                );
            }

            // Check the right-hand side of the response equation of perturbed
            // coupled-cluster amplitude
            let rhs_amplitude_x = lag.linear_response_rhs(&cc_amplitude_x, None)?;
            assert!(is_expr_type::<MatrixMul>(&rhs_amplitude_x));

            let rhs = downcast_from_arc::<MatrixMul>(&rhs_amplitude_x).unwrap();
            let rhs_coefficient = rhs.coefficient();

            assert_eq!(rhs_coefficient, &Number::one());

            let rhs_factors = rhs.factors();
            assert!(rhs_factors.len() == 2);

            let (tau_dagger, kappa_transformed_hamiltonian_x) =
                if is_expr_type::<HermitianTranspose>(&rhs_factors[0]) {
                    (&rhs_factors[0], &rhs_factors[1])
                } else {
                    (&rhs_factors[1], &rhs_factors[0])
                };

            assert_eq!(tau_dagger, &HermitianTranspose::new(cc_excitation_operator.clone())?);
            assert!(is_expr_type::<ExpAdjointMap>(kappa_transformed_hamiltonian_x));

            let exp_ad_map =
                downcast_from_arc::<ExpAdjointMap>(kappa_transformed_hamiltonian_x).unwrap();
            assert_eq!(exp_ad_map.generator(), lag.cluster_operator());

            assert_eq!(exp_ad_map.result(), &j_operator_x);
        }
    }

    // Find all (un)perturbed coupled-cluster multipliers
    let cc_multipliers = linear_response.find_superchains(&cc_multiplier);

    assert!(cc_multipliers.len() == 1);

    for (order, multipliers) in &cc_multipliers {
        assert_eq!(*order, 0);
        for multiplier in multipliers {
            assert_eq!(multiplier, &cc_multiplier);

            let rhs_multiplier = lag.linear_response_rhs(multiplier, None)?;
            assert!(is_expr_type::<MatrixMul>(&rhs_multiplier));

            let rhs = downcast_from_arc::<MatrixMul>(&rhs_multiplier).unwrap();
            assert_eq!(rhs.coefficient(), &Number::minus_one());
            assert!(rhs.factors().len() == 1);
            assert!(is_expr_type::<ExpAdjointMap>(&rhs.factors()[0]));

            let exp_ad_map = downcast_from_arc::<ExpAdjointMap>(&rhs.factors()[0]).unwrap();
            assert_eq!(exp_ad_map.generator(), lag.cluster_operator());
            assert_eq!(
                exp_ad_map.result(),
                // Equation (49), J. Chem. Phys. 92, 4924-4940 (1990)
                &AdjointMap::new(
                    vec![cc_excitation_operator.clone()],
                    MatrixAdd::new(vec![one_elec_matrix.clone(), two_elec_matrix.clone()])?,
                    Some(false),
                    Some(AdjointMode::Commutative)
                )?,
            );
        }
    }

    // Test RHS of the first order coupled-cluster multiplier
    let cc_multiplier_x = cc_multiplier.differentiate(&pert_x)?;
    {
        let rhs_multiplier_x = lag.linear_response_rhs(&cc_multiplier_x, None)?;
        assert!(is_expr_type::<MatrixMul>(&rhs_multiplier_x));

        let rhs = downcast_from_arc::<MatrixMul>(&rhs_multiplier_x).unwrap();
        assert_eq!(rhs.coefficient(), &Number::minus_one());
        assert!(rhs.factors().len() == 1);
        assert!(is_expr_type::<MatrixAdd>(&rhs.factors()[0]));

        let terms = downcast_from_arc::<MatrixAdd>(&rhs.factors()[0]).unwrap().terms();
        assert!(terms.len() == 2);
        // Equation (53), J. Chem. Phys. 92, 4924-4940 (1990)
        let (exp_ad_map, matrix_mul) = if is_expr_type::<ExpAdjointMap>(&terms[0]) {
            assert!(is_expr_type::<MatrixMul>(&terms[1]));
            (
                downcast_from_arc::<ExpAdjointMap>(&terms[0]).unwrap(),
                downcast_from_arc::<MatrixMul>(&terms[1]).unwrap(),
            )
        } else {
            assert!(is_expr_type::<MatrixMul>(&terms[0]));
            (
                downcast_from_arc::<ExpAdjointMap>(&terms[1]).unwrap(),
                downcast_from_arc::<MatrixMul>(&terms[2]).unwrap(),
            )
        };
        assert_eq!(exp_ad_map.generator(), lag.cluster_operator());
        assert_eq!(
            exp_ad_map.result(),
            &MatrixAdd::new(vec![
                AdjointMap::new(
                    vec![
                        lag.cluster_operator().differentiate(&pert_x)?,
                        cc_excitation_operator.clone()
                    ],
                    MatrixAdd::new(vec![one_elec_matrix.clone(), two_elec_matrix.clone()])?,
                    Some(false),
                    Some(AdjointMode::Commutative),
                )?,
                AdjointMap::new(
                    vec![cc_excitation_operator.clone()],
                    j_operator_x.clone(),
                    Some(false),
                    Some(AdjointMode::Commutative),
                )?,
            ])?
        );
        assert_eq!(matrix_mul.coefficient(), &Number::one());
        assert!(matrix_mul.factors().len() == 2);
        if is_expr_type::<DotProduct>(&matrix_mul.factors()[0]) {
            assert!(is_expr_type::<ExpAdjointMap>(&matrix_mul.factors()[1]));
            assert_eq!(&matrix_mul.factors()[0], lag.de_excitation_operator());
            assert_eq!(
                exp_ad_map,
                downcast_from_arc::<ExpAdjointMap>(&matrix_mul.factors()[1]).unwrap()
            );
        } else {
            assert!(is_expr_type::<ExpAdjointMap>(&matrix_mul.factors()[0]));
            assert!(is_expr_type::<DotProduct>(&matrix_mul.factors()[1]));
            assert_eq!(&matrix_mul.factors()[1], lag.de_excitation_operator());
            assert_eq!(
                exp_ad_map,
                downcast_from_arc::<ExpAdjointMap>(&matrix_mul.factors()[0]).unwrap()
            );
        }
    }

    // Find all (un)perturbed Brillouin condition multipliers
    let brillouin_multipliers = linear_response.find_superchains(&brillouin_multiplier);

    assert!(brillouin_multipliers.len() == 1);

    for (order, multipliers) in &brillouin_multipliers {
        assert_eq!(*order, 0);
        for multiplier in multipliers {
            assert_eq!(multiplier, &brillouin_multiplier);

            let rhs_multiplier = lag.linear_response_rhs(multiplier, None)?;
            // Equation (50), J. Chem. Phys. 92, 4924-4940 (1990)
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

    // Test RHS of the first order Brillouin condition multiplier
    // Equation (54), J. Chem. Phys. 92, 4924-4940 (1990)

    Ok(())
}
