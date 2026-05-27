use std::collections::BTreeMap;
use std::sync::Arc;
use symresponse::{Lagrangian, LagrangianOrbCc};
use tinned::{
    Add, AdjointMap, AdjointMode, DotProduct, ExcitationOperator, ExpAdjointMap, Expr,
    LagMultiplier, MatrixAdd, MatrixMul, Number, OneElecMatrix, PertMultichain, Perturbation,
    SubExpr, Symbol, TinnedError, Trace, Transpose, TwoElecMatrix, WfnParameter,
    differentiate_expr, downcast_from_arc, is_expr_type,
};

#[inline]
fn normalize_response_expr(
    expr: Arc<dyn Expr>,
    orb_rot_parameter: Arc<dyn Expr>,
    cc_amplitude: Arc<dyn Expr>,
    brillouin_multiplier: Arc<dyn Expr>,
    cc_multiplier: Arc<dyn Expr>,
    exten_perturbations: &[Arc<Perturbation>],
) -> Result<Arc<dyn Expr>, TinnedError> {
    expr.eliminate(orb_rot_parameter, exten_perturbations, 2)?
        .eliminate(cc_amplitude, exten_perturbations, 2)?
        .eliminate(brillouin_multiplier, exten_perturbations, 1)?
        .eliminate(cc_multiplier, exten_perturbations, 1)?
        .substitute_zero_perturbations(None)
}

#[inline]
fn expect_subexpr_derivative<'a>(expr: &'a Arc<dyn Expr>) -> &'a PertMultichain {
    downcast_from_arc::<SubExpr>(expr).expect("expression must be SubExpr").derivative()
}

#[inline]
fn expect_matrix_mul(expr: &Arc<dyn Expr>) -> &MatrixMul {
    downcast_from_arc::<MatrixMul>(expr).expect("expression must be MatrixMul")
}

#[inline]
fn expect_matrix_add(expr: &Arc<dyn Expr>) -> &MatrixAdd {
    downcast_from_arc::<MatrixAdd>(expr).expect("expression must be MatrixAdd")
}

#[inline]
fn expect_exp_adjoint_map(expr: &Arc<dyn Expr>) -> &ExpAdjointMap {
    downcast_from_arc::<ExpAdjointMap>(expr).expect("expression must be ExpAdjointMap")
}

#[inline]
fn split_two_terms_by_types<'a, A: 'static, B: 'static>(
    terms: &'a [Arc<dyn Expr>],
) -> (&'a Arc<dyn Expr>, &'a Arc<dyn Expr>) {
    assert_eq!(terms.len(), 2);

    if is_expr_type::<A>(&terms[0]) {
        assert!(is_expr_type::<B>(&terms[1]));
        (&terms[0], &terms[1])
    } else {
        assert!(is_expr_type::<B>(&terms[0]));
        assert!(is_expr_type::<A>(&terms[1]));
        (&terms[1], &terms[0])
    }
}

#[inline]
fn assert_matrix_mul_has_factor_pair_unordered(
    matrix_mul: &MatrixMul,
    expected_left: &Arc<dyn Expr>,
    expected_right: &Arc<dyn Expr>,
) {
    assert_eq!(matrix_mul.coefficient(), &Number::one());
    assert_eq!(matrix_mul.factors().len(), 2);

    let left = &matrix_mul.factors()[0];
    let right = &matrix_mul.factors()[1];

    assert!(
        (left == expected_left && right == expected_right)
            || (left == expected_right && right == expected_left)
    );
}

#[inline]
fn build_bch_expansion(op: &ExpAdjointMap) -> Result<Arc<dyn Expr>, TinnedError> {
    let bch_expansion = op.bch_expansion().clone();
    let terms = bch_expansion.into_values().flatten().collect();
    MatrixAdd::new(terms)
}

#[test]
fn orb_cc_linear_response() -> Result<(), TinnedError> {
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

    let normalize = |expr: Arc<dyn Expr>| {
        normalize_response_expr(
            expr,
            orb_rot_parameter.clone(),
            cc_amplitude.clone(),
            brillouin_multiplier.clone(),
            cc_multiplier.clone(),
            &exten_perturbations,
        )
    };

    // Using `min_wfn_exten_order = 0` means 2n+1 and 2n+2 rules
    let linear_response =
        lag.response_function(&exten_perturbations, &inten_perturbations, 0, false, None)?;

    // Get one- and two-electron density matrices. Note that they are not
    // evaluated at zero perturbation strength
    let one_elec_density = lag.one_elec_density().clone();
    let two_elec_density = lag.two_elec_density().clone();

    let one_elec_matrix_x = one_elec_matrix.differentiate(pert_x.clone())?;
    let two_elec_matrix_x = two_elec_matrix.differentiate(pert_x.clone())?;

    let orb_rot_parameter_x = orb_rot_parameter.differentiate(pert_x.clone())?;
    let orb_rot_parameter_y = orb_rot_parameter.differentiate(pert_y.clone())?;

    let cc_amplitude_x = cc_amplitude.differentiate(pert_x.clone())?;
    let cc_amplitude_y = cc_amplitude.differentiate(pert_y.clone())?;

    let kappa_operator_x = lag.kappa_operator().differentiate(pert_x.clone())?;
    let cluster_operator_x = lag.cluster_operator().differentiate(pert_x.clone())?;

    // Build reference linear response function
    let expected_linear_response = normalize(Add::new(vec![
        Trace::new(MatrixMul::new(vec![
            one_elec_density.clone(),
            differentiate_expr(&one_elec_matrix, &exten_perturbations)?,
        ])?)?,
        Trace::new(MatrixMul::new(vec![
            one_elec_density.differentiate(pert_y.clone())?,
            one_elec_matrix_x.clone(),
        ])?)?,
        Trace::new(MatrixMul::new(vec![
            one_elec_density.differentiate(pert_x.clone())?,
            one_elec_matrix.differentiate(pert_y.clone())?,
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
            two_elec_density.differentiate(pert_y.clone())?,
            two_elec_matrix_x.clone(),
        ])?)?,
        Trace::new(MatrixMul::new(vec![
            two_elec_density.differentiate(pert_x.clone())?,
            two_elec_matrix.differentiate(pert_y)?,
        ])?)?,
        Trace::new(MatrixMul::new(vec![
            differentiate_expr(&two_elec_density, &exten_perturbations)?,
            two_elec_matrix.clone(),
        ])?)?,
        differentiate_expr(
            &DotProduct::new(
                lag.brillouin_equation().clone(),
                false,
                brillouin_multiplier.clone(),
                false,
                Some(true),
            )?,
            &exten_perturbations,
        )?,
    ])?)?;

    assert_eq!(&linear_response, &expected_linear_response);

    // Find all (un)perturbed one-electron density matrices
    let one_elec_densities = linear_response.find_all(&one_elec_density);

    assert_eq!(one_elec_densities.len(), 3);

    // Check all (un)perturbed one-electron density matrices
    for (order, densities) in &one_elec_densities {
        assert!(*order <= 2);
        match *order {
            0 => {
                assert_eq!(densities.len(), 1);
                assert_eq!(
                    &one_elec_density.substitute_zero_perturbations(None)?,
                    densities.iter().next().expect("unperturbed one-electron density")
                );
            },
            _ => {
                for density in densities {
                    assert!(is_expr_type::<SubExpr>(density));
                    let perturbations = expect_subexpr_derivative(density);
                    let one_density_deriv = differentiate_expr(&one_elec_density, perturbations)?;
                    assert_eq!(&normalize(one_density_deriv)?, density);
                }
            },
        }
    }

    // Find all (un)perturbed two-electron density matrices
    let two_elec_densities = linear_response.find_all(&two_elec_density);

    assert_eq!(two_elec_densities.len(), 3);

    // Check all (un)perturbed two-electron density matrices
    for (order, densities) in &two_elec_densities {
        assert!(*order <= 2);
        match *order {
            0 => {
                assert_eq!(densities.len(), 1);
                assert_eq!(
                    &two_elec_density.substitute_zero_perturbations(None)?,
                    densities.iter().next().expect("unperturbed two-electron density")
                );
            },
            _ => {
                for density in densities {
                    assert!(is_expr_type::<SubExpr>(density));
                    let perturbations = expect_subexpr_derivative(density);
                    let two_density_deriv = differentiate_expr(&two_elec_density, perturbations)?;
                    assert_eq!(&normalize(two_density_deriv)?, density);
                }
            },
        }
    }

    // Find all perturbed orbital rotation parameters, unperturbed one is zero
    let orb_rot_parameters = linear_response.find_all(&orb_rot_parameter);

    assert_eq!(orb_rot_parameters.len(), 1);

    let hamiltonian_operator =
        MatrixAdd::new(vec![one_elec_matrix.clone(), two_elec_matrix.clone()])?;
    let rhs_hartree_fock_x = AdjointMap::new(
        vec![orb_rot_generator.clone()],
        hamiltonian_operator.differentiate(pert_x.clone())?,
        Some(true),
        Some(AdjointMode::Symmetrized),
    )?;

    for (order, parameters) in &orb_rot_parameters {
        assert_eq!(*order, 1);
        assert_eq!(parameters.len(), 2);

        for parameter in parameters {
            assert!(parameter == &orb_rot_parameter_x || parameter == &orb_rot_parameter_y);
        }

        // Check the right-hand side of the response equation of perturbed
        // orbital rotation parameters
        let rhs_parameter_x = lag.linear_response_rhs(&orb_rot_parameter_x, None)?;
        // Equation (51), J. Chem. Phys. 92, 4924-4940 (1990)
        assert_eq!(
            &rhs_parameter_x,
            &MatrixMul::new(vec![Number::minus_one(), rhs_hartree_fock_x.clone()])?
        );
    }

    // Find all (un)perturbed coupled-cluster amplitudes
    let cc_amplitudes = linear_response.find_all(&cc_amplitude);

    assert_eq!(cc_amplitudes.len(), 2);

    // The first derivative of the Hamiltonian operator including orbital rotations.
    // Equation (35), J. Chem. Phys. 92, 4924-4940 (1990)
    let eff_hamiltonian_x = MatrixAdd::new(vec![
        one_elec_matrix_x.clone(),
        two_elec_matrix_x.clone(),
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
            assert_eq!(amplitudes.len(), 1);
            assert_eq!(
                amplitudes.iter().next().expect("unperturbed coupled-cluster amplitudes"),
                &cc_amplitude
            );
        } else {
            assert_eq!(amplitudes.len(), 2);

            for amplitude in amplitudes {
                assert!(amplitude == &cc_amplitude_x || amplitude == &cc_amplitude_y);
            }

            // Check the right-hand side of the response equation of perturbed
            // coupled-cluster amplitude
            let rhs_amplitude_x = lag.linear_response_rhs(&cc_amplitude_x, None)?;
            assert!(is_expr_type::<MatrixMul>(&rhs_amplitude_x));

            let rhs = expect_matrix_mul(&rhs_amplitude_x);
            let rhs_coefficient = rhs.coefficient();

            assert_eq!(rhs_coefficient, &Number::one());

            let rhs_factors = rhs.factors();
            assert_eq!(rhs_factors.len(), 2);

            let (tau_dagger, kappa_transformed_hamiltonian_x) =
                split_two_terms_by_types::<Transpose, ExpAdjointMap>(rhs_factors);

            assert_eq!(tau_dagger, &Transpose::new(cc_excitation_operator.clone(), true)?);

            let exp_ad_map = expect_exp_adjoint_map(kappa_transformed_hamiltonian_x);
            assert_eq!(exp_ad_map.generator(), lag.cluster_operator());
            assert_eq!(&build_bch_expansion(exp_ad_map)?, &eff_hamiltonian_x);
        }
    }

    // Find all (un)perturbed coupled-cluster multipliers
    let cc_multipliers = linear_response.find_all(&cc_multiplier);

    assert_eq!(cc_multipliers.len(), 1);

    for (order, multipliers) in &cc_multipliers {
        assert_eq!(*order, 0);
        for multiplier in multipliers {
            assert_eq!(multiplier, &cc_multiplier);

            let rhs_multiplier = lag.linear_response_rhs(multiplier, None)?;
            assert!(is_expr_type::<MatrixMul>(&rhs_multiplier));

            let rhs = expect_matrix_mul(&rhs_multiplier);
            assert_eq!(rhs.coefficient(), &Number::minus_one());
            assert_eq!(rhs.factors().len(), 1);

            let exp_ad_map = expect_exp_adjoint_map(&rhs.factors()[0]);
            assert_eq!(exp_ad_map.generator(), lag.cluster_operator());
            assert_eq!(
                &build_bch_expansion(exp_ad_map)?,
                // Equation (49), J. Chem. Phys. 92, 4924-4940 (1990)
                &AdjointMap::new(
                    vec![cc_excitation_operator.clone()],
                    hamiltonian_operator.clone(),
                    Some(false),
                    Some(AdjointMode::Commutative)
                )?,
            );
        }
    }

    // Test RHS of the first-order coupled-cluster multiplier
    {
        let cc_multiplier_x = cc_multiplier.differentiate(pert_x.clone())?;
        let rhs_multiplier_x = lag.linear_response_rhs(&cc_multiplier_x, None)?;
        assert!(is_expr_type::<MatrixMul>(&rhs_multiplier_x));

        let rhs = expect_matrix_mul(&rhs_multiplier_x);
        assert_eq!(rhs.coefficient(), &Number::minus_one());
        assert_eq!(rhs.factors().len(), 1);

        let terms = expect_matrix_add(&rhs.factors()[0]).terms();
        assert_eq!(terms.len(), 2);

        // Equation (53), J. Chem. Phys. 92, 4924-4940 (1990)
        let (exp_ad_map_term, matrix_mul_term) =
            split_two_terms_by_types::<ExpAdjointMap, MatrixMul>(terms);

        let exp_ad_map = expect_exp_adjoint_map(exp_ad_map_term);
        let matrix_mul = expect_matrix_mul(matrix_mul_term);

        assert_eq!(exp_ad_map.generator(), lag.cluster_operator());
        assert_eq!(
            &build_bch_expansion(exp_ad_map)?,
            &MatrixAdd::new(vec![
                // [[H^(0), tau], T^(1)] = [T^(1), [tau, H^(0)]]
                AdjointMap::new(
                    vec![cluster_operator_x.clone(), cc_excitation_operator.clone()],
                    hamiltonian_operator.clone(),
                    Some(false),
                    Some(AdjointMode::Commutative),
                )?,
                // [J^(1), tau] = - [tau, J^(1)]
                AdjointMap::new(
                    vec![cc_excitation_operator.clone()],
                    eff_hamiltonian_x.clone(),
                    Some(false),
                    Some(AdjointMode::Commutative),
                )?,
            ])?
        );

        assert_matrix_mul_has_factor_pair_unordered(
            matrix_mul,
            lag.cc_lambda_operator(),
            exp_ad_map_term,
        );
    }

    // Find all (un)perturbed Brillouin condition multipliers
    let brillouin_multipliers = linear_response.find_all(&brillouin_multiplier);

    assert_eq!(brillouin_multipliers.len(), 1);

    // Equation (8), J. Chem. Phys. 92, 4924-4940 (1990)
    let unperturbed_brillouin_equation = AdjointMap::new(
        vec![orb_rot_generator.clone()],
        hamiltonian_operator.clone(),
        Some(true),
        Some(AdjointMode::Symmetrized),
    )?;

    for (order, multipliers) in &brillouin_multipliers {
        assert_eq!(*order, 0);
        for multiplier in multipliers {
            assert_eq!(multiplier, &brillouin_multiplier);

            let rhs_multiplier = lag.linear_response_rhs(multiplier, None)?;
            assert!(is_expr_type::<MatrixMul>(&rhs_multiplier));

            let rhs = expect_matrix_mul(&rhs_multiplier);
            assert_eq!(rhs.coefficient(), &Number::minus_one());
            assert_eq!(rhs.factors().len(), 1);

            let terms = expect_matrix_add(&rhs.factors()[0]).terms();
            assert_eq!(terms.len(), 2);

            // Equation (50), J. Chem. Phys. 92, 4924-4940 (1990)
            let (exp_ad_map_term, matrix_mul_term) =
                split_two_terms_by_types::<ExpAdjointMap, MatrixMul>(terms);

            let exp_ad_map = expect_exp_adjoint_map(exp_ad_map_term);
            let matrix_mul = expect_matrix_mul(matrix_mul_term);

            assert_eq!(exp_ad_map.generator(), lag.cluster_operator());
            assert_eq!(&build_bch_expansion(exp_ad_map)?, &unperturbed_brillouin_equation);

            assert_matrix_mul_has_factor_pair_unordered(
                matrix_mul,
                lag.cc_lambda_operator(),
                exp_ad_map_term,
            );
        }
    }

    // Test RHS of the first-order Brillouin condition multiplier
    {
        let brillouin_multiplier_x = brillouin_multiplier.differentiate(pert_x.clone())?;
        let rhs_multiplier_x = lag.linear_response_rhs(&brillouin_multiplier_x, None)?;
        assert!(is_expr_type::<MatrixMul>(&rhs_multiplier_x));

        let rhs = expect_matrix_mul(&rhs_multiplier_x);
        assert_eq!(rhs.coefficient(), &Number::minus_one());
        assert_eq!(rhs.factors().len(), 1);

        let terms = expect_matrix_add(&rhs.factors()[0]).terms();
        assert_eq!(terms.len(), 4);

        // Equation (54), J. Chem. Phys. 92, 4924-4940 (1990)
        let exp_ad_map = terms
            .iter()
            .find(|term| is_expr_type::<ExpAdjointMap>(term))
            .map(expect_exp_adjoint_map);
        assert_ne!(exp_ad_map, None);
        assert_eq!(
            exp_ad_map.expect("one term must be ExpAdjointMap").generator(),
            lag.cluster_operator()
        );

        // [E^{-}, kappa^(1), H^(0)]
        let comm_kappa_x_hamiltonian = AdjointMap::new(
            vec![kappa_operator_x.clone(), orb_rot_generator.clone()],
            hamiltonian_operator.clone(),
            Some(true),
            Some(AdjointMode::Symmetrized),
        )?;

        // The second line of equation (54) but with <HF|
        let diff_kappa_transformed_hamiltonian_x = MatrixAdd::new(vec![
            comm_kappa_x_hamiltonian.clone(),
            // [[E^{-}, H^(0)], T^(1)]
            AdjointMap::new(
                vec![cluster_operator_x.clone()],
                unperturbed_brillouin_equation.clone(),
                Some(false),
                Some(AdjointMode::Commutative),
            )?,
            rhs_hartree_fock_x.clone(),
        ])?;
        assert_eq!(
            &build_bch_expansion(exp_ad_map.expect("one term must be ExpAdjointMap"))?,
            &diff_kappa_transformed_hamiltonian_x
        );

        // Term with zeroth-order Brillouin condition multiplier
        let zero_brillouin_multiplier_term =
            terms.iter().find(|term| is_expr_type::<DotProduct>(term)).map(Arc::clone);
        assert_ne!(zero_brillouin_multiplier_term, None);
        assert_eq!(
            &zero_brillouin_multiplier_term.expect("one term must be DotProduct"),
            &DotProduct::new(
                AdjointMap::new(
                    vec![orb_rot_generator.clone()],
                    MatrixAdd::new(vec![
                        rhs_hartree_fock_x.clone(),
                        comm_kappa_x_hamiltonian.clone()
                    ])?,
                    Some(true),
                    Some(AdjointMode::Symmetrized)
                )?,
                false,
                brillouin_multiplier.clone(),
                false,
                Some(false)
            )?
        );

        let cc_multiplier_terms: Vec<_> =
            terms.iter().filter_map(|term| downcast_from_arc::<MatrixMul>(term)).collect();
        assert_eq!(cc_multiplier_terms.len(), 2);

        let mut exist_cc_multiplier_x = false;
        let mut exist_eff_hamiltonian_x = false;

        let cc_lambda_operator_x = lag.cc_lambda_operator().differentiate(pert_x)?;

        for term in cc_multiplier_terms {
            assert_eq!(term.coefficient(), &Number::one());
            assert_eq!(term.factors().len(), 2);

            if &term.factors()[0] == lag.cc_lambda_operator() {
                let oper = expect_exp_adjoint_map(&term.factors()[1]);
                assert_eq!(oper.generator(), lag.cluster_operator());
                assert_eq!(&build_bch_expansion(oper)?, &diff_kappa_transformed_hamiltonian_x);
                exist_eff_hamiltonian_x = true;
            } else if &term.factors()[0] == &cc_lambda_operator_x {
                // The last line of equation (54), but there are two typos in
                // that equation: Lambda should be first order differentiated
                // and Hamiltonian should be undifferentiated.
                let oper = expect_exp_adjoint_map(&term.factors()[1]);
                assert_eq!(oper.generator(), lag.cluster_operator());
                assert_eq!(&build_bch_expansion(oper)?, &unperturbed_brillouin_equation);
                exist_cc_multiplier_x = true;
            } else if &term.factors()[0] == &unperturbed_brillouin_equation {
                assert_eq!(&term.factors()[1], &cc_lambda_operator_x);
                exist_cc_multiplier_x = true;
            } else {
                let oper = expect_exp_adjoint_map(&term.factors()[0]);
                assert_eq!(oper.generator(), lag.cluster_operator());
                assert_eq!(&build_bch_expansion(oper)?, &diff_kappa_transformed_hamiltonian_x);
                assert_eq!(&term.factors()[1], lag.cc_lambda_operator());
                exist_eff_hamiltonian_x = true;
            }
        }

        assert!(exist_cc_multiplier_x);
        assert!(exist_eff_hamiltonian_x);
    }

    Ok(())
}
