/* SymResponse: a unified framework for response theory
   Copyright 2024 Bin Gao

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.

   This file is the header file of quasi-energy Lagrangian in response theory.

   2024-06-02, Bin Gao:
   * rewrite by using template method pattern

   2024-01-30, Bin Gao:
   * first version
*/

#pragma once

#include <cmath>
#include <limits>

#include <symengine/basic.h>
#include <symengine/dict.h>
#include <symengine/number.h>
#include <symengine/real_double.h>
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
            // Validate sum of perturbations' frequencies
            virtual bool validate_perturbation_frequencies(
                const Tinned::PerturbationTuple& exten_perturbations,
                const Tinned::PerturbationTuple& inten_perturbations,
                const SymEngine::RCP<const SymEngine::Number>&
                    threshold = SymEngine::real_double(std::numeric_limits<double>::epsilon())
            ) const noexcept;

            // Validate that extensive and intensive perturbations should be disjoint
            virtual bool validate_perturbation_disjointedness(
                const Tinned::PerturbationTuple& exten_perturbations,
                const Tinned::PerturbationTuple& inten_perturbations
            ) const noexcept;

            // Validate that: (i) at least one extensive perturbation, (ii) sum
            // of all perturbations' frequencies should be zero, and (iii)
            // extensive and intensive perturbations should be disjoint
            inline void validate_perturbations(
                const Tinned::PerturbationTuple& exten_perturbations,
                const Tinned::PerturbationTuple& inten_perturbations,
                const SymEngine::RCP<const SymEngine::Number>&
                    threshold = SymEngine::real_double(std::numeric_limits<double>::epsilon())
            ) const
            {
                if (exten_perturbations.empty()) throw SymEngine::SymEngineException(
                    "Lagrangian requires at least one extensive perturbation!"
                );
                if (!validate_perturbation_frequencies(
                    exten_perturbations, inten_perturbations, threshold
                )) throw SymEngine::SymEngineException(
                    "Lagrangian gets perturbations with non-zero sum frequencies!"
                );
                if (!validate_perturbation_disjointedness(
                    exten_perturbations, inten_perturbations
                )) throw SymEngine::SymEngineException(
                    "Lagrangian gets a same extensive and intensive perturbation!"
                );
            }

            // Get the time-averaged quasi-energy (derivative) Lagrangian
            virtual SymEngine::RCP<const SymEngine::Basic> get_lagrangian() const noexcept = 0;

            // Eliminate differentiated wave function parameter with respect to
            // extensive perturbations `exten_perturbations` from the
            // derivative of quasi-energy Lagrangian `L`. Orders of
            // differentiated wave function parameter to be eliminated are from
            // the minimum one `min_wfn_order` to the maximum one as the size
            // of `exten_perturbations`.
            virtual SymEngine::RCP<const SymEngine::Basic> eliminate_wavefunction_parameter(
                const SymEngine::RCP<const SymEngine::Basic>& L,
                const Tinned::PerturbationTuple& exten_perturbations,
                const unsigned int min_wfn_order
            ) = 0;

            // Eliminate differentiated Lagrangian multipliers with respect to
            // extensive perturbations `exten_perturbations` from the
            // derivative of quasi-energy Lagrangian `L`. Orders of
            // differentiated Lagrangian multipliers to be eliminated are from
            // the minimum one `min_multiplier_order` to the maximum one as the
            // size of `exten_perturbations`.
            virtual SymEngine::RCP<const SymEngine::Basic> eliminate_lagrangian_multipliers(
                const SymEngine::RCP<const SymEngine::Basic>& L,
                const Tinned::PerturbationTuple& exten_perturbations,
                const unsigned int min_multiplier_order
            ) = 0;

        public:
            explicit Lagrangian() = default;

            // Get response functions by using template method pattern
            inline SymEngine::RCP<const SymEngine::Basic> get_response_functions(
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
                const unsigned int min_wfn_exten = 0
            )
            {
                // Validate perturbations
                validate_perturbations(exten_perturbations, inten_perturbations);
                // Differentiate the quasi-energy (derivative) Lagrangian
                auto perturbations = exten_perturbations;
                if (!inten_perturbations.empty()) perturbations.insert(
                    inten_perturbations.begin(), inten_perturbations.end()
                );
                auto result = Tinned::differentiate(get_lagrangian(), perturbations);
                // Usually the differentiated quasi-energy Lagrangian cannot be zero
                if (Tinned::is_zero_quantity(result)) return result;
                // Minimum order for the elimination of wave function
                // parameters is the next integer of the floor function of the
                // half number of perturbations, according to Table IV, J.
                // Chem. Phys. 129, 214103 (2008)
                auto min_wfn_order
                    = static_cast<unsigned int>(std::floor(0.5*exten_perturbations.size()))+1;
                if (min_wfn_exten>0) {
                    if (min_wfn_exten<min_wfn_order) {
                        throw SymEngine::SymEngineException(
                            "Lagrangian::get_response_functions() gets an invalid minimum order "
                            + std::to_string(min_wfn_exten)
                        );
                    }
                    else {
                        min_wfn_order = min_wfn_exten;
                    }
                }
                // Eliminate wave function parameter
                if (min_wfn_exten<=exten_perturbations.size())
                    result = eliminate_wavefunction_parameter(
                        result, exten_perturbations, min_wfn_order
                    );
                // Minimum order for the elimination of Lagrangian multipliers,
                // see Table V, J.  Chem. Phys. 129, 214103 (2008)
                unsigned int min_multiplier_order
                    = min_wfn_exten<=exten_perturbations.size()
                    ? exten_perturbations.size()-min_wfn_order+1 : 0;
                // Eliminate Lagrangian multipliers
                result = eliminate_lagrangian_multipliers(
                    result, exten_perturbations, min_multiplier_order
                );
                // Usually `result` cannot be null after elimination
                if (result.is_null()) return SymEngine::zero;
                // Remove unperturbed time-differentiated quantities (and
                // unperturbed T matrix for the AO density matrix-based
                // response theory) as well as their perturbed ones but with
                // zero sum of perturbation frequencies. Replace those with
                // non-zero sum of frequencies by corresponding derivatives in
                // the frequency domain multiplied by the sum of frequencies.
                return Tinned::clean_temporum(result);
            }

            //// Get residues. Other parameters see the function `get_response_functions()`.
            //virtual SymEngine::RCP<const SymEngine::Basic> get_residues(
            //    const Tinned::PerturbationTuple& exten_perturbations,
            //    const Tinned::PerturbationTuple& inten_perturbations = {},
            //    const unsigned int min_wfn_exten = 0
            //) = 0;

            virtual ~Lagrangian() noexcept = default;
    };
}
