#include <symengine/constants.h>

#include "SymResponse/Lagrangian.hpp"

namespace SymResponse
{
    bool Lagrangian::verify_perturbation_frequencies(
        const Tinned::PerturbationTuple& exten_perturbations,
        const Tinned::PerturbationTuple& inten_perturbations,
        const SymEngine::RCP<const SymEngine::Number>& threshold
    ) const noexcept
    {
        SymEngine::RCP<const SymEngine::Number> sum_freq = SymEngine::zero;
        for (const auto& p: exten_perturbations)
            sum_freq = SymEngine::addnum(sum_freq, p->get_frequency());
        for (const auto& p: inten_perturbations)
            sum_freq = SymEngine::addnum(sum_freq, p->get_frequency());
        return Tinned::is_zero_number(sum_freq, threshold);
    }

    bool Lagrangian::verify_perturbation_disjointedness(
        const Tinned::PerturbationTuple& exten_perturbations,
        const Tinned::PerturbationTuple& inten_perturbations
    ) const noexcept
    {
        for (const auto& p: inten_perturbations)
            if (exten_perturbations.find(p)!=exten_perturbations.end()) return false;
        return true;
    }
}
