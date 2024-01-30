/* SymResponse: a unified framework for response theory
   Copyright 2024 Bin Gao

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.

   This file is the header file of Lagrangian in response theory.

   2024-01-30, Bin Gao:
   * first version
*/

#pragma once

namespace SymResponse
{
    class Lagrangian
    {
        protected:
            // Truncate perturbed quantities `x`'s in `expression` after `k`-th
            // order when involving the perturbation `a`
            template<typename T>
            inline SymEngine::RCP<const SymEngine::Basic> make_k_truncation(
                const SymEngine::RCP<const SymEngine::Basic>& expression,
                const SymEngine::RCP<T>& x,
                const unsigned int k,
                const SymEngine::RCP<const Tinned::Perturbation>& a
            ) const
            {
                auto all_x = Tinned::find_all<T>(expression, x);
                SymEngine::set_basic higher_x;
                for (const auto& xp: all_x) {
                    auto derivative = xp->get_derivative();
                    if (derivative.find(a) != derivative.end() && derivative.size() > k)
                        higher_x.insert(xp);
                }
                return higher_x.empty()
                    ? expression : Tinned::remove_if(expression, higher_x);
            }

            // Truncate perturbed quantities `x`'s in `expression` after `n`-th
            // order
            template<typename T>
            inline SymEngine::RCP<const SymEngine::Basic> make_n_truncation(
                const SymEngine::RCP<const SymEngine::Basic>& expression,
                const SymEngine::RCP<T>& x,
                const unsigned int n
            ) const
            {
                auto all_x = Tinned::find_all<T>(expression, x);
                SymEngine::set_basic higher_x;
                for (const auto& xp: all_x)
                    if (xp->get_derivative().size() > n) higher_x.insert(xp);
                return higher_x.empty()
                    ? expression : Tinned::remove_if(expression, higher_x);
            }

        public:
            explicit Lagrangian() {}
            // Get response functions
            virtual SymEngine::RCP<const SymEngine::Basic> get_response_functions(
                const SymEngine::multiset_basic& perturbations
            ) noexcept = 0;
            // Get residues
            virtual SymEngine::RCP<const SymEngine::Basic> get_residues(
                const SymEngine::multiset_basic& perturbations
            ) noexcept = 0;
            SymEngine::set_basic get_states() noexcept;
            SymEngine::RCP<const SymEngine::Basic> get_rhs() noexcept;
            virtual ~Lagrangian() noexcept = default;
    };
}
