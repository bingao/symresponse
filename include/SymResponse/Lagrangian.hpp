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
#include <functional>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include <symengine/basic.h>
#include <symengine/dict.h>
#include <symengine/number.h>
#include <symengine/real_double.h>
#include <symengine/symengine_exception.h>
#include <symengine/symengine_rcp.h>

#include <Tinned.hpp>

namespace SymResponse
{
    // A map with key being a minimal elimination order of
    // differentiated wave function parameters with respect to
    // extensive perturbations, and value as the corresponding response
    // functions
    typedef std::map<unsigned int, SymEngine::RCP<const SymEngine::Basic>>
        ElimOrderMap;

    // Compare two pairs of perturbation multichains
    inline bool cmp_pert_multichain_pair(
        const std::pair<Tinned::PertMultichain, Tinned::PertMultichain>& lhs,
        const std::pair<Tinned::PertMultichain, Tinned::PertMultichain>& rhs
    )
    {
        auto result = Tinned::compare_pert_multichain(lhs.first, rhs.first);
        switch (result) {
            case -1:
                return true;
            case 1:
                return false;
            default:
                result = Tinned::compare_pert_multichain(lhs.second, rhs.second);
                return result<=0 ? true : false;
        }
    }

    // A map with key being extensive perturbations, and value as a
    // pair of intensive perturbations, and an `ElimOrderMap` object
    typedef std::map<std::pair<Tinned::PertMultichain, Tinned::PertMultichain>,
                     ElimOrderMap, decltype(cmp_pert_multichain_pair)*>
        ElimExtsvMap;

    // Abstract quasi-energy Lagrangian class, can also be used for
    // quasi-energy without Lagrangian multipliers
    class Lagrangian
    {
        protected:
            // Validate sum of perturbations' frequencies
            virtual bool validate_perturbation_frequencies(
                const Tinned::PertMultichain& exten_perturbations,
                const Tinned::PertMultichain& inten_perturbations,
                const SymEngine::RCP<const SymEngine::Number>&
                    threshold = SymEngine::real_double(std::numeric_limits<double>::epsilon())
            ) const noexcept;

            // Validate that extensive and intensive perturbations should be disjoint
            virtual bool validate_perturbation_disjointness(
                const Tinned::PertMultichain& exten_perturbations,
                const Tinned::PertMultichain& inten_perturbations
            ) const noexcept;

            // Validate that: (i) at least one extensive perturbation, (ii) sum
            // of all perturbations' frequencies should be zero, and (iii)
            // extensive and intensive perturbations should be disjoint
            inline void validate_perturbations(
                const Tinned::PertMultichain& exten_perturbations,
                const Tinned::PertMultichain& inten_perturbations,
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
                if (!validate_perturbation_disjointness(
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
                const Tinned::PertMultichain& exten_perturbations,
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
                const Tinned::PertMultichain& exten_perturbations,
                const unsigned int min_multiplier_order
            ) = 0;

            // Evaluation at zero perturbation strength
            virtual SymEngine::RCP<const SymEngine::Basic> at_zero_strength(
                const SymEngine::RCP<const SymEngine::Basic>& L
            );

            // Get unperturbed wave function parameters and Lagrangian
            // multipliers, used to find optimal elimination rules
            virtual SymEngine::vec_basic get_wavefunction_parameter() const noexcept = 0;
            virtual SymEngine::vec_basic get_lagrangian_multipliers() const noexcept = 0;

        public:
            explicit Lagrangian() = default;

            // Get response functions by using template method pattern
            inline SymEngine::RCP<const SymEngine::Basic> get_response_functions(
                // Extensive perturbations
                const Tinned::PertMultichain& exten_perturbations,
                // Intensive perturbations
                const Tinned::PertMultichain& inten_perturbations = {},
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
                // Evaluation at zero perturbation strength
                return at_zero_strength(result);
            }

            // Find optimal elimination rules according to the given weight
            // function on (un)perturbed wave function parameters and
            // Lagrangian multipliers.
            //
            // `excluded` contains symbols that should be excluded from
            // response functions, for example, when a perturbed operator
            // cannot be evaluated.
            //
            // Return a pair of a minimal weight and a map with key being a
            // minimum order of differentiated wave function parameters with
            // respect to extensive perturbations to be eliminated, and value
            // the corresponding response functions.
            //
            // If an empty `ElimOrderMap` is returned, no response functions
            // can be computed with the given conditions.
            inline std::pair<unsigned int, ElimOrderMap>
            get_response_functions(
                const Tinned::PertMultichain& exten_perturbations,
                const Tinned::PertMultichain& inten_perturbations = {},
                const SymEngine::set_basic& excluded = {},
                const std::function<unsigned int(
                    const std::vector<Tinned::FindAllResult>&,
                    const std::vector<Tinned::FindAllResult>&
                )>& get_weight = [](
                    const std::vector<Tinned::FindAllResult>& wfn_parameters,
                    const std::vector<Tinned::FindAllResult>& multipliers
                ) -> unsigned int {
                    unsigned int num_parameters = 0;
                    for (const auto& p: wfn_parameters)
                        for (const auto& t: p) num_parameters += t.second.size();
                    for (const auto& p: multipliers)
                        for (const auto& t: p) num_parameters += t.second.size();
                    return num_parameters;
                }
            )
            {
                unsigned int min_weight = std::numeric_limits<unsigned int>::max();
                ElimOrderMap results;
                // Loop over the order of differentiated wave function
                // parameters with respect to extensive perturbations to be
                // eliminated
                auto min_wfn_order
                    = static_cast<unsigned int>(std::floor(0.5*exten_perturbations.size()))+1;
                auto max_wfn_order = exten_perturbations.size()+1;
                for (unsigned int order=min_wfn_order; order<=max_wfn_order; ++order) {
                    auto rsp_function = get_response_functions(
                        exten_perturbations, inten_perturbations, order
                    );
                    if (!Tinned::exist_any(rsp_function, excluded)) {
                        // Find all (un)perturbed wave function parameters and
                        // Lagrangian multipliers for this response function
                        auto wfn_parameters = std::vector<Tinned::FindAllResult>({});
                        auto multipliers = std::vector<Tinned::FindAllResult>({});
                        for (const auto& wfn_parameter: get_wavefunction_parameter()) {
                            wfn_parameters.push_back(Tinned::find_all(
                                rsp_function, wfn_parameter
                            ));
                        }
                        for (const auto& multiplier: get_lagrangian_multipliers()) {
                            multipliers.push_back(Tinned::find_all(
                                rsp_function, multiplier
                            ));
                        }
                        auto weight = get_weight(wfn_parameters, multipliers);
                        if (weight==min_weight) {
                            results.insert({order, rsp_function});
                        }
                        else if (weight<min_weight) {
                            min_weight = weight;
                            results.clear();
                            results.insert({order, rsp_function});
                        }
                    }
                }
                return std::make_pair(min_weight, results);
            }

            // Find all possible extensive and intensive perturbations, and
            // corresponding optimal elimination rules.
            //
            // `perturbations` contains perturbations that can either be
            // extensive or intensive.
            //
            // Return a pair of a minimal weight and a map with key being
            // extensive perturbations, and value as a pair of (1) intensive
            // perturbations, and (2) a map with key being a minimum order of
            // differentiated wave function parameters with respect to
            // extensive perturbations to be eliminated, and value the
            // corresponding response functions.
            //
            // If an empty `ElimExtsvMap` is returned, no response functions
            // can be computed with the given conditions.
            inline std::pair<unsigned int, ElimExtsvMap>
            get_response_functions(
                const Tinned::PertMultichain& perturbations,
                const Tinned::PertMultichain& exten_perturbations,
                const Tinned::PertMultichain& inten_perturbations = {},
                const SymEngine::set_basic& excluded = {},
                const std::function<unsigned int(
                    const std::vector<Tinned::FindAllResult>&,
                    const std::vector<Tinned::FindAllResult>&
                )>& get_weight = [](
                    const std::vector<Tinned::FindAllResult>& wfn_parameters,
                    const std::vector<Tinned::FindAllResult>& multipliers
                ) -> unsigned int {
                    unsigned int num_parameters = 0;
                    for (const auto& p: wfn_parameters)
                        for (const auto& t: p) num_parameters += t.second.size();
                    for (const auto& p: multipliers)
                        for (const auto& t: p) num_parameters += t.second.size();
                    return num_parameters;
                }
            )
            {
                if (perturbations.empty()) {
                    auto results = get_response_functions(
                        exten_perturbations, inten_perturbations, excluded, get_weight
                    );
                    return std::make_pair(
                        results.first,
                        ElimExtsvMap({
                            {
                                std::make_pair(exten_perturbations, inten_perturbations),
                                results.second
                            }
                        })
                    );
                }
                unsigned int min_weight = std::numeric_limits<unsigned int>::max();
                ElimExtsvMap results;
                // Find unique perturbations and their numbers of occurrences (multiplicities)
                auto pert_multiplicities = Tinned::make_pert_multiplicity(perturbations);
                auto num_perturbations = pert_multiplicities.size();
                unsigned int num_subchains = 1<<num_perturbations;
                // Make sure there is at least one extensive perturbation, so
                // the first subchain 0...0 is discarded for empty extensive
                // perturbations
                unsigned int chain = exten_perturbations.empty() ? 1 : 0;
                // We generate all subsets of the unique perturbations by
                // following the direct approach in Chapter 1, "Next Subset of
                // an n-Set (NEXSUB/LEXSUB)", Combinatorial Algorithms For
                // Computers and Calculators (2nd Edition), Albert Nijenhuis
                // and Herbert S. Wilf. New extensive perturbations are from
                // the subset while new intensive perturbations are from the
                // complement of the subset.
                for (; chain<num_subchains; ++chain) {
                    auto extsv_perturbations = exten_perturbations;
                    auto intsv_perturbations = inten_perturbations;
                    for (unsigned int ipert=0; ipert<num_perturbations; ++ipert) {
                        if (chain & (1<<ipert)) {
                            for (unsigned int imult=0;
                                 imult<pert_multiplicities[ipert].second;
                                 ++imult) extsv_perturbations.insert(
                                    pert_multiplicities[ipert].first
                                );
                        }
                        else {
                            for (unsigned int imult=0;
                                 imult<pert_multiplicities[ipert].second;
                                 ++imult) intsv_perturbations.insert(
                                    pert_multiplicities[ipert].first
                                );
                        }
                    }
                    // Find optimal elimination rules and response functions
                    // for the current extensive and intensive perturbations
                    auto elim_results = get_response_functions(
                        extsv_perturbations, intsv_perturbations, excluded, get_weight
                    );
                    if (elim_results.first==min_weight) {
                        results.insert({
                            std::make_pair(extsv_perturbations, intsv_perturbations),
                            elim_results.second
                        });
                    }
                    else if (elim_results.first<min_weight) {
                        min_weight = elim_results.first;
                        results.clear();
                        results.insert({
                            std::make_pair(extsv_perturbations, intsv_perturbations),
                            elim_results.second
                        });
                    }
                }
                return std::make_pair(min_weight, results);
            }

            //// Get residues. Other parameters see the function `get_response_functions()`.
            //virtual SymEngine::RCP<const SymEngine::Basic> get_residues(
            //    const Tinned::PertMultichain& exten_perturbations,
            //    const Tinned::PertMultichain& inten_perturbations = {},
            //    const unsigned int min_wfn_exten = 0
            //) = 0;

            virtual ~Lagrangian() noexcept = default;
    };
}
