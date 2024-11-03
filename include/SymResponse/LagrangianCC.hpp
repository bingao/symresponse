/* SymResponse: a unified framework for response theory
   Copyright 2024 Bin Gao

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.

   This file is the header file of Lagrangian in coupled-cluster response
   theory.

   2024-06-03, Bin Gao:
   * first version
*/

#pragma once

#include <symengine/basic.h>
#include <symengine/dict.h>
#include <symengine/constants.h>
#include <symengine/symengine_rcp.h>

#include <Tinned.hpp>

#include "SymResponse/Lagrangian.hpp"

namespace SymResponse
{
    /* Time-averaged quasi-energy Lagrangian */
    class LagrangianCC: virtual public Lagrangian
    {
        protected:
            // Perturbation operators
            SymEngine::vec_basic V_;
            // Coupled-cluster amplitudes (row vector)
            SymEngine::RCP<const Tinned::PerturbedParameter> t_;
            // Lagrangian multipliers (row vector)
            SymEngine::RCP<const Tinned::PerturbedParameter> multiplier_;
            // Similarity-transformed Hamiltonian, or exponential map
            SymEngine::RCP<const SymEngine::Basic> H_cc_;
            // Used to compute the right-hand side of the response equation of
            // Lagrangian multipliers
            SymEngine::RCP<const SymEngine::Basic> rhs_multiplier_;
            // Time-averaged quasi-energy Lagrangian
            SymEngine::RCP<const SymEngine::Basic> L_;

            // Override functions for template method pattern
            virtual SymEngine::RCP<const SymEngine::Basic> get_lagrangian() const noexcept override;

            virtual SymEngine::RCP<const SymEngine::Basic> eliminate_wavefunction_parameter(
                const SymEngine::RCP<const SymEngine::Basic>& L,
                const Tinned::PertMultichain& exten_perturbations,
                const unsigned int min_wfn_order
            ) override;

            virtual SymEngine::RCP<const SymEngine::Basic> eliminate_lagrangian_multipliers(
                const SymEngine::RCP<const SymEngine::Basic>& L,
                const Tinned::PertMultichain& exten_perturbations,
                const unsigned int min_multiplier_order
            ) override;

            virtual SymEngine::RCP<const SymEngine::Basic> at_zero_strength(
                const SymEngine::RCP<const SymEngine::Basic>& L
            ) override;

            virtual SymEngine::vec_basic get_wavefunction_parameter() const noexcept override;
            virtual SymEngine::vec_basic get_lagrangian_multipliers() const noexcept override;

        public:
            explicit LagrangianCC(
                const SymEngine::RCP<const SymEngine::Basic>& H0,
                const SymEngine::vec_basic& V,
                const SymEngine::RCP<const Tinned::PerturbedParameter>& t,
                // Commuting excitation operators (row vector)
                const SymEngine::RCP<const SymEngine::MatrixExpr>& tau,
                const SymEngine::RCP<const Tinned::PerturbedParameter>& multiplier
            );

            // Get right-hand side (RHS) of the (linear) response equation
            inline SymEngine::RCP<const SymEngine::Basic> get_response_rhs(
                const SymEngine::RCP<const SymEngine::Basic>& rsp_parameter,
                const bool is_multiplier = false
            )
            {
                auto op = SymEngine::rcp_dynamic_cast<const Tinned::PerturbedParameter>(
                    rsp_parameter
                );
                auto to_remove = SymEngine::set_basic(V_.begin(), V_.end());
                to_remove.insert(rsp_parameter);
                if (is_multiplier) {
                    return Tinned::remove_if(
                        Tinned::differentiate(rhs_multiplier_, op->get_derivatives()),
                        to_remove
                    );
                }
                else {
                    return Tinned::remove_if(
                        Tinned::differentiate(
                            SymEngine::mul({SymEngine::minus_one, H_cc_}),
                            op->get_derivatives()
                        ),
                        to_remove
                    );
                }
            }

            virtual ~LagrangianCC() noexcept = default;
    };
}
