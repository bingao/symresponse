/* SymResponse: a unified framework for response theory
   Copyright 2024 Bin Gao

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.

   This file is the header file of Lagrangian in atomic-orbital density matrix
   based response theory.

   2024-01-30, Bin Gao:
   * first version
*/

#pragma once

#include <symengine/basic.h>
#include <symengine/dict.h>
#include <symengine/constants.h>
#include <symengine/symengine_rcp.h>
#include <symengine/matrices/matrix_add.h>
#include <symengine/matrices/matrix_mul.h>

#include <Tinned.hpp>

#include "SymResponse/Lagrangian.hpp"

namespace SymResponse
{
    /* Lagrangian in atomic-orbital density matrix based response theory.

       In the framework of atomic-orbital (AO) density matrix based response
       theory, we need to store:

       . overlap operator,
       . zero-, one- and two-electron operators, and exchange-correlation (XC)
         functional derivative in the Hamiltonian,
       . reference state,
       . excited states, and
       . a linear response solver.

       Because response theory These quantities

       Among them, different operators in the Hamiltonian will be stored in a
       corresponding `std::vector` as one may need several such operators in
       the Hamiltonian.
     */
    class LagrangianDAO: virtual public Lagrangian
    {
        protected:
            // One-electron spin-orbital density matrix
            SymEngine::RCP<const Tinned::ElectronicState> D_;
            // |i\frac{\partial `D_`}{\partial t}>
            SymEngine::RCP<const TemporumOperator> Dt_;
            // Perturbation a
            SymEngine::RCP<const Tinned::Perturbation> a_;
            // Overlap
            SymEngine::RCP<const Tinned::OneElecOperator> S_;
            // |i\frac{\partial `S_`}{\partial t}>
            SymEngine::RCP<const TemporumOperator> St_;

            // Generalized Fock matrix, the sum of one-electron operator(s),
            // two-electron operator(s) and exchange-correlation potential(s)
            SymEngine::RCP<const SymEngine::Basic> F_;

            // Generalized energy derivative with respect to the perturbation a
            // while removing the corresponding perturbed density matrix
            SymEngine::RCP<const SymEngine::Basic> Ea_;

            // Generalized energy-weighted density matrix
            SymEngine::RCP<const SymEngine::Basic> W_;
            // Lagrangian multiplier \lambda
            SymEngine::RCP<const SymEngine::Basic> lambda_;
            // TDSCF equation
            SymEngine::RCP<const SymEngine::Basic> Y_;
            // Lagrangian multiplier \zeta
            SymEngine::RCP<const SymEngine::Basic> zeta_;
            // Idempotency constraint
            SymEngine::RCP<const SymEngine::Basic> Z_;
            // Lagrangian function
            SymEngine::RCP<const SymEngine::Basic> La_;

        public:
            explicit LagrangianDAO(
                const SymEngine::RCP<const Tinned::ElectronicState>& D,
                const SymEngine::RCP<const Tinned::Perturbation>& a,
                const SymEngine::RCP<const SymEngine::Basic>& S = SymEngine::RCP<const SymEngine::Basic>(),
                const SymEngine::RCP<const SymEngine::Basic>& H = SymEngine::RCP<const SymEngine::Basic>(),
                const SymEngine::RCP<const SymEngine::Basic>& G = SymEngine::RCP<const SymEngine::Basic>(),
                const SymEngine::RCP<const SymEngine::Basic>& Exc = SymEngine::RCP<const SymEngine::Basic>(),
                const SymEngine::RCP<const SymEngine::Basic>& Fxc = SymEngine::RCP<const SymEngine::Basic>(),
                const SymEngine::RCP<const SymEngine::Basic>& hnuc = SymEngine::RCP<const SymEngine::Basic>()
            );
            virtual SymEngine::RCP<const SymEngine::Basic> get_response_functions(
                const SymEngine::multiset_basic& perturbations,
                const unsigned int k = 0
            ) noexcept override;
            virtual SymEngine::RCP<const SymEngine::Basic> get_residues(
                const SymEngine::multiset_basic& perturbations,
                const unsigned int k = 0
            ) noexcept override;
            SymEngine::set_basic get_states() noexcept;
            SymEngine::RCP<const SymEngine::Basic> get_rhs() noexcept;
            virtual ~LagrangianDAO() noexcept = default;
    };
}
