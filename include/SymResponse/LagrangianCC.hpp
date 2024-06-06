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
#include <symengine/symengine_rcp.h>

#include <Tinned.hpp>

#include "SymResponse/Lagrangian.hpp"

namespace SymResponse
{
    /* Time-averaged quasi-energy Lagrangian */
    class LagrangianCC: virtual public Lagrangian
    {
        protected:
            // Coupled-cluster amplitudes (row vector)
            SymEngine::RCP<const Tinned::PerturbedParameter> amplitudes_;
            // Lagrangian multipliers (row vector)
            SymEngine::RCP<const Tinned::PerturbedParameter> multipliers_;
            // Time-averaged quasi-energy Lagrangian
            SymEngine::RCP<const SymEngine::Basic> L_;

            // Override functions for template method pattern
            virtual SymEngine::RCP<const SymEngine::Basic> get_lagrangian() const noexcept override;

            virtual SymEngine::RCP<const SymEngine::Basic> eliminate_wavefunction_parameter(
                const SymEngine::RCP<const SymEngine::Basic>& L,
                const Tinned::PerturbationTuple& exten_perturbations,
                const unsigned int min_wfn_order
            ) override;

            virtual SymEngine::RCP<const SymEngine::Basic> eliminate_lagrangian_multipliers(
                const SymEngine::RCP<const SymEngine::Basic>& L,
                const Tinned::PerturbationTuple& exten_perturbations,
                const unsigned int min_multiplier_order
            ) override;

        public:
            explicit LagrangianCC(
                const SymEngine::RCP<const SymEngine::Basic>& H,
                const SymEngine::RCP<const Tinned::PerturbedParameter>& amplitudes,
                // Commuting excitation operators (row vector)
                const SymEngine::RCP<const SymEngine::MatrixExpr>& excit_operators,
                const SymEngine::RCP<const Tinned::PerturbedParameter>& multipliers,
                // `ref_state` will be needed for orbital-relaxed response theory
                const SymEngine::RCP<const SymEngine::Basic>&
                    ref_state = SymEngine::RCP<const SymEngine::Basic>()
            );

            virtual ~LagrangianCC() noexcept = default;
    };
}
