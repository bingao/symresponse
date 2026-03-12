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
    let cc_amplitudes = WfnParameter::builder("t").build()?;
    let cc_excitation_operators = OneElecOperator::builder("tau").build()?;
    let cc_multipliers = LagMultiplier::builder("tbar").build()?;
    let orb_rotation_parameters = WfnParameter::builder("kappa").build()?;
    // Orbital rotation generators, $\hat{a}^{\dagger)_{r}\hat{a}_{s}$
    let orb_rotation_generators = OneElecOperator::builder("E-").build()?;
    let brillouin_multipliers = LagMultiplier::builder("kbar").build()?;

    let lag = LagrangianOrbCc::new(
        one_elec_operator,
        two_elec_operator,
        cc_amplitudes,
        cc_excitation_operators,
        cc_multipliers,
        orb_rotation_parameters,
        orb_rotation_generators,
        brillouin_multipliers,
    )?;

    let mut result = lag.get_lagrangian().clone();
    match serde_json::to_string(&result) {
        Ok(json) => println!("Lagrangian = {}", json),
        Err(err) => {
            eprintln!("Serialization of Lagrangian failed: {err}");
        },
    }

    let exten_perturbations = vec![pert_a.clone(), pert_b.clone(), pert_c.clone()];
    let inten_perturbations: Vec<Arc<Perturbation>> = Vec::new();

    // Using `min_wfn_extern = 3` removes all Lagrangian multipliers
    result = lag.response_function(&exten_perturbations, &inten_perturbations, 3, false, None)?;
    match serde_json::to_string(&result) {
        Ok(json) => println!("L^{{abc}} = {}", json),
        Err(err) => {
            eprintln!("Serialization of L^{{abc}} failed: {err}");
        },
    }

    Ok(())
}
