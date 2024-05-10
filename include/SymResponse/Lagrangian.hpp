/* SymResponse: a unified framework for response theory
   Copyright 2024 Bin Gao

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.

   This file is the header file of quasi-energy Lagrangian in response theory.

   2024-01-30, Bin Gao:
   * first version
*/

#pragma once

#include <cmath>

#include <symengine/basic.h>
#include <symengine/constants.h>
#include <symengine/dict.h>
#include <symengine/number.h>
#include <symengine/symengine_exception.h>
#include <symengine/symengine_rcp.h>

#include <Tinned.hpp>

namespace SymResponse
{
    // Abstract quasi-energy Lagrangian class, can also be used for
    // quasi-energy without Lagrangian multipliers
    class Lagrangian
    {
        protected:
            // Verify that: (i) at least one extensive perturbation, (ii) sum
            // of all perturbations' frequencies should be zero, and (iii)
            // extensive and intensive perturbations should be disjoint
            inline void verify_perturbations(
                const Tinned::PerturbationTuple& exten_perturbations,
                const Tinned::PerturbationTuple& inten_perturbations
            ) const
            {
                if (exten_perturbations.empty()) throw SymEngine::SymEngineException(
                    "Lagrangian requires at least one extensive perturbation!"
                );
                SymEngine::RCP<const SymEngine::Number> sum_freq = SymEngine::zero;
                for (const auto& p: exten_perturbations)
                    sum_freq = SymEngine::addnum(sum_freq, p->get_frequency());
                if (!inten_perturbations.empty()) {
                    for (const auto& p: inten_perturbations) {
                        sum_freq = SymEngine::addnum(sum_freq, p->get_frequency());
                        if (exten_perturbations.find(p)!=exten_perturbations.end())
                            throw SymEngine::SymEngineException(
                                "Lagrangian gets a same extensive and intensive perturbation!"
                            );
                    }
                }
                if (!sum_freq->is_zero()) throw SymEngine::SymEngineException(
                    "Lagrangian gets perturbations with invalid frequencies!"
                );
            }

            // Differentiate quasi-energy Lagrangian `L`, and then eliminate
            // wave function parameter (`wfn`) and Lagrangian multipliers
            // (`multipliers`) from its derivative according to Table IV and V,
            // J. Chem. Phys. 129, 214103 (2008). Other parameters see the
            // function `get_response_functions()`.
            inline SymEngine::RCP<const SymEngine::Basic> diff_and_eliminate(
                const SymEngine::RCP<const SymEngine::Basic>& L,
                const Tinned::PerturbationTuple& exten_perturbations,
                const Tinned::PerturbationTuple& inten_perturbations,
                const SymEngine::RCP<const SymEngine::Basic>& wfn,
                const SymEngine::set_basic& multipliers,
                const unsigned int min_wfn_extern = 0
            )
            {
                // Differentiate quasi-energy Lagrangian
                auto perturbations = exten_perturbations;
                if (!inten_perturbations.empty()) perturbations.insert(
                    inten_perturbations.begin(), inten_perturbations.end()
                );
                auto result = Tinned::differentiate(L, perturbations);
                // Usually the differentiated quasi-energy Lagrangian cannot be zero
                if (Tinned::is_zero_quantity(*result)) return result;
                // Minimum order for the elimination of wave function
                // parameters, as the next integer of the floor function of the
                // half number of perturbations
                auto min_wfn_order
                    = static_cast<unsigned int>(std::floor(0.5*exten_perturbations.size()))
                    + 1;
                if (min_wfn_extern>0) {
                    if (min_wfn_extern<min_wfn_order) {
                        throw SymEngine::SymEngineException(
                            "Lagrangian::diff_and_eliminate() gets an invalid minimum order "
                            + std::to_string(min_wfn_extern)
                        );
                    }
                    else {
                        min_wfn_order = min_wfn_extern;
                    }
                }
                // Eliminate wave function parameter
                if (min_wfn_extern<=exten_perturbations.size())
                    result = Tinned::eliminate(
                        result, wfn, exten_perturbations, min_wfn_order
                    );
                // Minimum order for the elimination of Lagrangian multipliers
                unsigned int min_multiplier_order
                    = min_wfn_extern<=exten_perturbations.size()
                    ? exten_perturbations.size()-min_wfn_order+1 : 0;
                // For MCSCF, an empty `multipliers` can be used
                for (const auto& multiplier: multipliers)
                    result = Tinned::eliminate(
                        result, multiplier, exten_perturbations, min_multiplier_order
                    );
                // Usually `result` cannot be null after elimination
                if (result.is_null()) return SymEngine::zero;
                return result;
            }

        public:
            explicit Lagrangian() = default;

            // Get response functions
            virtual SymEngine::RCP<const SymEngine::Basic> get_response_functions(
                // Extensive perturbations
                const Tinned::PerturbationTuple& exten_perturbations,
                // Intensive perturbations
                const Tinned::PerturbationTuple& inten_perturbations = {},
                // Minimum order of differentiated wave function parameters with
                // respect to extensive perturbations to be eliminated.
                // Default value is 0 that means it will be automatically
                // determined as the next integer of the floor function of the
                // half number of extensive perturbations. For values greater
                // than the number of extensive perturbations, it means no
                // elimination of wave function parameters so that more
                // Lagrangian multipliers can be eliminated.
                const unsigned int min_wfn_extern = 0
            ) = 0;

            //// Get residues. Other parameters see the function `get_response_functions()`.
            //virtual SymEngine::RCP<const SymEngine::Basic> get_residues(
            //    const Tinned::PerturbationTuple& exten_perturbations,
            //    const Tinned::PerturbationTuple& inten_perturbations = {},
            //    const unsigned int min_wfn_extern = 0
            //) = 0;

            virtual ~Lagrangian() noexcept = default;
    };
}
