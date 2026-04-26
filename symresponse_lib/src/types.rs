use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tinned::{Expr, PertMultichain, Perturbation, ResidueParameter, TinnedError, generic_error};

// Elimination scheme, internal use
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct EliminationScheme {
    num_perturbations: u32,
    eliminate_wfn: bool,
    min_wfn_order: u32,
    min_multiplier_order: u32,
}

impl EliminationScheme {
    // `requested_min_wfn_order` as 0, means the minimum order for the
    // elimination of wave function parameters will be automatically determined
    // as the next integer of the floor function of the half number of
    // perturbations. For `requested_min_wfn_order` greater than the number of
    // perturbations, it means no elimination of wave function parameters will
    // be performed so that more Lagrangian multipliers can be eliminated.
    pub(crate) fn new(
        num_perturbations: u32,
        requested_min_wfn_order: u32,
    ) -> Result<EliminationScheme, TinnedError> {
        if num_perturbations == 0 {
            return Err(generic_error("Requires at least one perturbation", None));
        }

        // Minimum order for the elimination of wave function parameters is the
        // next integer of the floor function of the half number of
        // perturbations, according to Table IV, J. Chem. Phys. 129, 214103 (2008)
        let auto_min_wfn_order = 1 + (num_perturbations / 2);

        if requested_min_wfn_order > 0 && requested_min_wfn_order < auto_min_wfn_order {
            return Err(generic_error(
                format!(
                    "Requested minimum elimination order {} for wave function parameters is invalid",
                    requested_min_wfn_order
                ),
                None,
            ));
        }

        let resolved_min_wfn_order = if requested_min_wfn_order == 0 {
            auto_min_wfn_order
        } else {
            requested_min_wfn_order
        };

        let eliminate_wfn = requested_min_wfn_order <= num_perturbations;

        // Minimum order for the elimination of Lagrangian multipliers,
        // see Table V, J. Chem. Phys. 129, 214103 (2008)
        let min_multiplier_order = if eliminate_wfn {
            num_perturbations + 1 - resolved_min_wfn_order
        } else {
            0
        };

        Ok(EliminationScheme {
            num_perturbations,
            eliminate_wfn,
            min_wfn_order: resolved_min_wfn_order,
            min_multiplier_order,
        })
    }

    //#[inline]
    //pub(crate) fn num_perturbations(&self) -> u32 {
    //    self.num_perturbations
    //}

    #[inline]
    pub(crate) fn eliminate_wfn(&self) -> bool {
        self.eliminate_wfn
    }

    #[inline]
    pub(crate) fn min_wfn_order(&self) -> u32 {
        self.min_wfn_order
    }

    #[inline]
    pub(crate) fn min_multiplier_order(&self) -> u32 {
        self.min_multiplier_order
    }
}

// Internal use: `diff_params` contains differentiated parameters with respect
// to perturbations involved in a residue computation, while `residue_map`
// contains relationship between each differentiated parameter and its residue
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResidueSetup {
    diff_params: HashSet<Arc<dyn Expr>>,
    residue_map: HashMap<Arc<dyn Expr>, Arc<dyn Expr>>,
}

impl ResidueSetup {
    pub(crate) fn new(
        diff_params: HashSet<Arc<dyn Expr>>,
        residue_map: HashMap<Arc<dyn Expr>, Arc<dyn Expr>>,
    ) -> Self {
        Self {
            diff_params,
            residue_map,
        }
    }

    #[inline]
    pub(crate) fn diff_params(&self) -> &HashSet<Arc<dyn Expr>> {
        &self.diff_params
    }

    #[inline]
    pub(crate) fn residue_map(&self) -> &HashMap<Arc<dyn Expr>, Arc<dyn Expr>> {
        &self.residue_map
    }
}

/// Detailed information about an optimized response function or residue.
///
/// Includes:
/// * `expression` - The resulting expression of response function or residue
/// * `min_wfn_exten_order` - Minimum order of differentiated wave function
///   parameters to eliminate with respect to extensive perturbations
/// * `exten_perturbations` - Extensive perturbations (must contain at least one element)
/// * `inten_perturbations` - Intensive perturbations (may be empty)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResponseDetail {
    pub expression: Arc<dyn Expr>,
    pub min_wfn_exten_order: u32,
    pub exten_perturbations: Vec<Arc<Perturbation>>,
    pub inten_perturbations: Vec<Arc<Perturbation>>,
}

// Internal use: prepare right-hand side (RHS) of linear response equation
pub struct LinearRhsInput<'a> {
    // Equation to be differentiated to get RHS
    pub(crate) equation: &'a Arc<dyn Expr>,
    // Perturbations that the equation will be differentiated with respect to
    pub(crate) derivative: &'a PertMultichain,
    // Differentiated response parameter to be solved and should be removed
    // from the differentiated equation
    pub(crate) diff_parameter: &'a Arc<dyn Expr>,
    // For computation of residue, contains the residue response parameter and
    // unperturbed parameter
    pub(crate) residue_info: Option<(&'a ResidueParameter, Arc<dyn Expr>)>,
}
