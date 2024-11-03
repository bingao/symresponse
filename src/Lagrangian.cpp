#include <symengine/add.h>
#include <symengine/constants.h>

#include "SymResponse/Lagrangian.hpp"

namespace SymResponse
{
    bool Lagrangian::validate_perturbation_frequencies(
        const Tinned::PertMultichain& exten_perturbations,
        const Tinned::PertMultichain& inten_perturbations,
        const SymEngine::RCP<const SymEngine::Number>& threshold
    ) const noexcept
    {
        SymEngine::RCP<const SymEngine::Basic> sum_freq = SymEngine::zero;
        for (const auto& p: exten_perturbations)
            sum_freq = SymEngine::add(sum_freq, p->get_frequency());
        for (const auto& p: inten_perturbations)
            sum_freq = SymEngine::add(sum_freq, p->get_frequency());
        if (SymEngine::is_a_Number(*sum_freq)) {
            return Tinned::is_zero_number(
                SymEngine::rcp_dynamic_cast<const SymEngine::Number>(sum_freq),
                threshold
            );
        }
        else {
            // For nonnumerical frequencies, we simply return `true` and users
            // are responsible for the validation
            return true;
        }
    }

    bool Lagrangian::validate_perturbation_disjointness(
        const Tinned::PertMultichain& exten_perturbations,
        const Tinned::PertMultichain& inten_perturbations
    ) const noexcept
    {
        for (const auto& p: inten_perturbations)
            if (exten_perturbations.find(p)!=exten_perturbations.end()) return false;
        return true;
    }

    SymEngine::RCP<const SymEngine::Basic> Lagrangian::at_zero_strength(
        const SymEngine::RCP<const SymEngine::Basic>& L
    )
    {
        // Remove unperturbed time-differentiated quantities (and unperturbed T
        // matrix for the AO density matrix-based response theory) as well as
        // their perturbed ones but with zero sum of perturbation frequencies.
        // Replace those with non-zero sum of frequencies by corresponding
        // derivatives in the frequency domain multiplied by the sum of
        // frequencies.
        return Tinned::clean_temporum(L);
    }
}
