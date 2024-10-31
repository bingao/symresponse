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

#include <limits>

#include <symengine/basic.h>
#include <symengine/dict.h>
#include <symengine/number.h>
#include <symengine/real_double.h>
#include <symengine/constants.h>
#include <symengine/matrices/matrix_add.h>
#include <symengine/matrices/matrix_mul.h>
#include <symengine/symengine_rcp.h>

#include <Tinned.hpp>

#include "SymResponse/Lagrangian.hpp"

namespace SymResponse
{
    /* Time-averaged quasi-energy derivative Lagrangian */
    class LagrangianDAO: virtual public Lagrangian
    {
        protected:
            // Whether elimination rules performed in a (almost) symmetric form
            // or not [see Sec. IV G 3, J. Chem. Phys. 129, 214108 (2008)]
            bool sym_elimination_;
            // Perturbation $a$
            SymEngine::RCP<const Tinned::Perturbation> a_;
            // One-electron spin-orbital density matrix
            SymEngine::RCP<const Tinned::OneElecDensity> D_;
            // Overlap
            SymEngine::RCP<const Tinned::OneElecOperator> S_;

            // Generalized energy
            SymEngine::RCP<const SymEngine::Basic> E_;
            // Generalized Fock matrix, the sum of one-electron operator(s),
            // two-electron operator(s) and exchange-correlation potential(s)
            SymEngine::RCP<const SymEngine::Basic> F_;
            // Generalized energy-weighted density matrix
            SymEngine::RCP<const SymEngine::Basic> W_;
            // Lagrangian multiplier $\lambda$
            SymEngine::RCP<const SymEngine::Basic> lambda_;
            // An "artificial" Lagrangian multiplier for elimination
            SymEngine::RCP<const Tinned::PerturbedParameter> tdscf_multiplier_;
            // TDSCF equation
            SymEngine::RCP<const SymEngine::Basic> Y_;
            // Lagrangian multiplier $\zeta$
            SymEngine::RCP<const SymEngine::Basic> zeta_;
            // An "artificial" Lagrangian multiplier for elimination
            SymEngine::RCP<const Tinned::PerturbedParameter> idempotency_multiplier_;
            // Idempotency constraint
            SymEngine::RCP<const SymEngine::Basic> Z_;
            // Time-averaged quasi-energy derivative Lagrangian
            SymEngine::RCP<const SymEngine::Basic> La_;

            // Override functions for template method pattern
            virtual bool validate_perturbation_frequencies(
                const Tinned::PertMultichain& exten_perturbations,
                const Tinned::PertMultichain& inten_perturbations,
                const SymEngine::RCP<const SymEngine::Number>&
                    threshold = SymEngine::real_double(std::numeric_limits<double>::epsilon())
            ) const noexcept override;

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

        public:
            explicit LagrangianDAO(
                const SymEngine::RCP<const Tinned::Perturbation>& a,
                const SymEngine::RCP<const Tinned::OneElecDensity>& D,
                const SymEngine::RCP<const Tinned::OneElecOperator>& S,
                const SymEngine::vec_basic& H = {},
                const SymEngine::RCP<const Tinned::TwoElecOperator>&
                    G = SymEngine::RCP<const Tinned::TwoElecOperator>(),
                const SymEngine::RCP<const Tinned::ExchCorrEnergy>&
                    Exc = SymEngine::RCP<const Tinned::ExchCorrEnergy>(),
                const SymEngine::RCP<const Tinned::ExchCorrPotential>&
                    Fxc = SymEngine::RCP<const Tinned::ExchCorrPotential>(),
                const SymEngine::RCP<const Tinned::NonElecFunction>&
                    hnuc = SymEngine::RCP<const Tinned::NonElecFunction>(),
                const bool sym_elimination = false
            );

            // For orthonormal basis sets
            explicit LagrangianDAO(
                const SymEngine::RCP<const Tinned::Perturbation>& a,
                const SymEngine::RCP<const Tinned::OneElecDensity>& D,
                const SymEngine::vec_basic& H = {},
                const SymEngine::RCP<const Tinned::TwoElecOperator>&
                    G = SymEngine::RCP<const Tinned::TwoElecOperator>(),
                const SymEngine::RCP<const Tinned::ExchCorrEnergy>&
                    Exc = SymEngine::RCP<const Tinned::ExchCorrEnergy>(),
                const SymEngine::RCP<const Tinned::ExchCorrPotential>&
                    Fxc = SymEngine::RCP<const Tinned::ExchCorrPotential>(),
                const SymEngine::RCP<const Tinned::NonElecFunction>&
                    hnuc = SymEngine::RCP<const Tinned::NonElecFunction>(),
                const bool sym_elimination = false
            ): LagrangianDAO(
                a,
                D,
                SymEngine::RCP<const Tinned::OneElecOperator>(),
                H,
                G,
                Exc,
                Fxc,
                hnuc,
                sym_elimination) {}

            //virtual SymEngine::RCP<const SymEngine::Basic> get_residues(
            //    // Extensive perturbations without `a`
            //    const Tinned::PertMultichain& exten_perturbations,
            //    const Tinned::PertMultichain& inten_perturbations = {},
            //    const unsigned int min_wfn_exten = 0
            //) override;

            // Set elimination rules performed either in an asymmetric
            // (`false`) or (almost) symmetric (`true`) form
            inline void set_elimination_form(const bool sym_elimination) noexcept
            {
                sym_elimination_ = sym_elimination;
            }

            // Whether elimination rules performed in an asymmetric (`false`)
            // or (almost) symmetric (`true`) form
            inline bool get_elimination_form() const noexcept
            {
                return sym_elimination_;
            }

            // Get perturbation $a$
            inline SymEngine::RCP<const Tinned::Perturbation> get_perturbation_a() const noexcept
            {
                return a_;
            }

            // Get one-electron spin-orbital density matrix
            inline SymEngine::RCP<const Tinned::OneElecDensity> get_density() const noexcept
            {
                return D_;
            }

            // Get overlap matrix
            inline SymEngine::RCP<const Tinned::OneElecOperator> get_overlap() const noexcept
            {
                return S_;
            }

            // Get generalized energy
            inline SymEngine::RCP<const SymEngine::Basic> get_generalized_energy() const noexcept
            {
                return E_;
            }

            // Get generalized Fock matrix
            inline SymEngine::RCP<const SymEngine::Basic> get_generalized_fock() const noexcept
            {
                return F_;
            }

            // Get generalized energy-weighted density matrix
            inline SymEngine::RCP<const SymEngine::Basic> get_ew_density() const noexcept
            {
                return W_;
            }

            // Get Lagrangian multiplier $\lambda$
            inline SymEngine::RCP<const SymEngine::Basic> get_tdscf_multiplier() const noexcept
            {
                return lambda_;
            }

            // Get TDSCF equation
            inline SymEngine::RCP<const SymEngine::Basic> get_tdscf_equation() const noexcept
            {
                return Y_;
            }

            // Get Lagrangian multiplier $\zeta$
            inline SymEngine::RCP<const SymEngine::Basic> get_idempotency_multiplier() const noexcept
            {
                return zeta_;
            }

            // Get idempotency constraint
            inline SymEngine::RCP<const SymEngine::Basic> get_idempotency_constraint() const noexcept
            {
                return Z_;
            }

//            // Response equation
//            virtual SymEngine::RCP<const SymEngine::Basic> get_response_equation(
//                LHS, RHS
//            ) override;

            // Get particular solution of a perturbed density matrix
            inline SymEngine::RCP<const SymEngine::Basic> get_particular_density(
                const SymEngine::RCP<const SymEngine::Basic>& Dw
            ) const
            {
                auto op = SymEngine::rcp_dynamic_cast<const Tinned::OneElecDensity>(Dw);
                if (S_.is_null()) {
                    auto K = Tinned::remove_if(
                        Tinned::differentiate(
                            SymEngine::matrix_mul({D_, D_}), op->get_derivatives()
                        ),
                        SymEngine::set_basic({Dw})
                    );
                    return SymEngine::matrix_add({
                        SymEngine::matrix_mul({SymEngine::minus_one, K}),
                        SymEngine::matrix_mul({K, D_}),
                        SymEngine::matrix_mul({D_, K}),
                    });
                }
                else {
                    auto K = Tinned::remove_if(
                        Tinned::differentiate(
                            SymEngine::matrix_mul({D_, S_, D_}), op->get_derivatives()
                        ),
                        SymEngine::set_basic({Dw})
                    );
                    return SymEngine::matrix_add({
                        SymEngine::matrix_mul({SymEngine::minus_one, K}),
                        SymEngine::matrix_mul({K, S_, D_}),
                        SymEngine::matrix_mul({D_, S_, K}),
                    });
                }
            }

            // Get right-hand side (RHS) of the (linear) response equation
            inline SymEngine::RCP<const SymEngine::Basic> get_response_rhs(
                const SymEngine::RCP<const SymEngine::Basic>& Dw,
                const SymEngine::RCP<const SymEngine::Basic>& DP
            ) const
            {
                auto op = SymEngine::rcp_dynamic_cast<const Tinned::OneElecDensity>(Dw);
                return Tinned::replace(
                    Tinned::differentiate(Y_, op->get_derivatives()),
                    SymEngine::map_basic_basic({{Dw, DP}})
                );
            }

            virtual ~LagrangianDAO() noexcept = default;
    };
}
