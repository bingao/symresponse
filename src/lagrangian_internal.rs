pub(crate) mod sealed {
    use std::collections::HashSet;
    use std::sync::Arc;

    use tinned::{
        Add, Expr, NumberTolerance, Perturbation, TinnedError, generic_error, is_zero_expr,
        sum_pert_frequencies,
    };

    pub trait LagrangianInternal {
        // Checks if the sum of perturbations' frequencies is non zero
        #[inline]
        fn is_non_zero_sum_freqs(
            &self,
            exten_perturbations: &[Arc<Perturbation>],
            inten_perturbations: &[Arc<Perturbation>],
            num_tol: Option<NumberTolerance>,
        ) -> Result<bool, TinnedError> {
            let freq_sum_ext = sum_pert_frequencies(&exten_perturbations).map_err(|e| {
                generic_error(
                    "Sum of extensive perturbations' frequencies failed",
                    Some(Box::new(e)),
                )
            })?;
            let freq_sum_int = sum_pert_frequencies(&inten_perturbations).map_err(|e| {
                generic_error(
                    "Sum of intensive perturbations' frequencies failed",
                    Some(Box::new(e)),
                )
            })?;
            let total_freq = Add::new(vec![freq_sum_ext, freq_sum_int]).map_err(|e| {
                generic_error(
                    "Sum of all perturbations' frequencies failed",
                    Some(Box::new(e)),
                )
            })?;

            Ok(!is_zero_expr(&total_freq, num_tol))
        }

        // Checks if extensive and intensive perturbations have any common one(s)
        #[inline]
        fn has_common_perturbation(
            &self,
            exten_perturbations: &[Arc<Perturbation>],
            inten_perturbations: &[Arc<Perturbation>],
        ) -> bool {
            let exten_set: HashSet<_> = exten_perturbations.iter().collect();
            inten_perturbations.iter().any(|p| exten_set.contains(p))
        }

        // Eliminate differentiated wave function parameter with respect to
        // extensive perturbations `exten_perturbations` from the derivative of
        // quasi-energy Lagrangian. Orders of differentiated wave function
        // parameter to be eliminated are from the minimum one `min_wfn_order` to
        // the maximum one as the size of `exten_perturbations`.
        fn eliminate_wfn_parameter(
            &self,
            lagrangian: &Arc<dyn Expr>,
            exten_perturbations: &[Arc<Perturbation>],
            min_wfn_order: u32,
        ) -> Result<Arc<dyn Expr>, TinnedError>;

        // Eliminate differentiated Lagrangian multipliers with respect to
        // extensive perturbations `exten_perturbations` from the derivative of
        // quasi-energy Lagrangian. Orders of differentiated Lagrangian multipliers
        // to be eliminated are from the minimum one `min_multiplier_order` to the
        // maximum one as the size of `exten_perturbations`.
        fn eliminate_lag_multipliers(
            &self,
            lagrangian: &Arc<dyn Expr>,
            exten_perturbations: &[Arc<Perturbation>],
            min_multiplier_order: u32,
        ) -> Result<Arc<dyn Expr>, TinnedError>;

        // Evaluation at zero perturbation strength
        #[inline]
        fn at_zero_strength(
            &self,
            lagrangian: &Arc<dyn Expr>,
            num_tol: Option<NumberTolerance>,
        ) -> Result<Arc<dyn Expr>, TinnedError> {
            // Remove unperturbed time-differentiated quantities (and unperturbed T
            // matrix for the AO density matrix-based response theory) as well as
            // their perturbed ones but with zero sum of perturbation frequencies.
            // Replace those with non-zero sum of frequencies by corresponding
            // derivatives in the frequency domain multiplied by the sum of
            // frequencies.
            lagrangian.clean_temporum(num_tol)
        }
    }
}
