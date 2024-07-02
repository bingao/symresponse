/* SymResponse: a unified framework for response theory
   Copyright 2024 Bin Gao

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.

   This file is the header file of (numerical) evaluator for atomic-orbital
   density matrix based response theory.

   2024-06-11, Bin Gao:
   * first version
*/

#pragma once

#include <cstddef>
#include <memory>

#include <symengine/visitor.h>

#include <Tinned.hpp>

#include "SymResponse/LagrangianDAO.hpp"

namespace SymResponse
{
    // CLass template for evaluating different operators
    template<typename OperatorType>
    class OperatorEvaluatorDAO: virtual public Tinned::OperatorEvaluator<OperatorType>
    {
        protected:
            OperatorType D_;
            //FIXME: orthonormal basis
            OperatorType S_;
            OperatorType F_;
            // Cached perturbed density matrices
            std::map<SymEngine::RCP<const Tinned::OneElecDensity>, OperatorType> D_cached_;
            // Particular solution of a perturbed density matrix
            SymEngine::RCP<const SymEngine::Basic> DP_sym_;
            OperatorType DP_;

        public:
            explicit OperatorEvaluatorDAO(
                const OperatorType& D,
                const OperatorType& S,
                const OperatorType& F
            ): D_(D), S_(S), F(F_) {}

            void eval_wavefunction_parameter(
                const SymEngine::RCP<const SymEngine::Basic>& expr
            ) override
            {
                auto D_sym = L_->get_density();
                auto D_all = Tinned::find_all(expr, D_sym);
                for (std::size_t i=0; i<D_all.size(); ++i) {
                    auto Dw = SymEngine::rcp_dynamic_cast<const Tinned::OneElecDensity>(
                        D_all[i]
                    );
                    // Get particular solution of the perturbed density matrix
                    DP_sym_ = L_->get_particular_density(Dw);
                    DP_ = apply(DP_sym_);
                    // Evaluate the right-hand side of the linear response equation
                    auto RHS = apply(L_->get_linear_rhs(Dw, DP_sym_));
                    auto freq_sum = Tinned::get_frequency_sum(D_sym->get_derivatives());
                    auto X = eval_response_parameter(freq_sum, RHS);
                    auto DH = D_*S_*X - X*S_*D_;
                    D[?] = SymEngine::matrix_add({DP_, DH});

                }
            }

            virtual ~OperatorEvaluatorDAO() noexcept = default;
    };

    // Class template for evaluating response functions
    template<typename FunctionType, typename OperatorType>
    class FunctionEvaluatorDAO: virtual public Tinned::FunctionEvaluator<FunctionType>
    {
        protected:
            std::shared_ptr<const LagrangianDAO> L_;

            OperatorEvaluatorDAO<OperatorType> oper_evaluator_;

        public:
            explicit FunctionEvaluatorDAO() = default;

            // Get response functions by using template method pattern
            inline SymEngine::RCP<const SymEngine::Basic> get_response_functions(
                const Tinned::PerturbationTuple& exten_perturbations,
                const Tinned::PerturbationTuple& inten_perturbations = {},
                const unsigned int min_wfn_exten = 0
            )
            {
                auto expr = L_->get_response_function(
                    exten_perturbations, inten_perturbations, min_wfn_exten
                );
                oper_evaluator.eval_wavefunction_parameter(expr);
                return apply(expr);
            }

            virtual ~FunctionEvaluatorDAO() noexcept = default;
    };
}
