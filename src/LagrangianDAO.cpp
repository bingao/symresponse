/* SymResponse: a unified framework for response theory
   Copyright 2024 Bin Gao

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.

   This file implements the Lagrangian in atomic-orbital density matrix based
   response theory.

   2024-01-30, Bin Gao:
   * first version
*/

#include <symengine/add.h>
#include <symengine/mul.h>

#include "SymResponse/LagrangianDAO.hpp"

namespace SymResponse
{
    LagrangianDAO::LagrangianDAO(
        const SymEngine::RCP<const Tinned::ElectronicState>& D,
        const SymEngine::RCP<const Tinned::Perturbation>& a,
        const SymEngine::RCP<const SymEngine::Basic>& S,
        const SymEngine::RCP<const SymEngine::Basic>& H,
        const SymEngine::RCP<const SymEngine::Basic>& G,
        const SymEngine::RCP<const SymEngine::Basic>& Exc,
        const SymEngine::RCP<const SymEngine::Basic>& Fxc,
        const SymEngine::RCP<const SymEngine::Basic>& hnuc
    ) : D_(D),
        Dt_(Tinned::make_dt_operator(D)),
        a_(a)
    {
        if (!S.is_null()) {
            S_ = SymEngine::rcp_dynamic_cast<const Tinned::OneElecOperator>(S);
            St_ = Tinned::make_dt_operator(S_);
        }
        SymEngine::vec_basic F_terms;
        SymEngine::vec_basic E_terms;
        if (!H.is_null()) {
            F_terms.push_back(H);
            E_terms.push_back(SymEngine::trace(SymEngine::matrix_mul({H, D})));
        }
        if (!G.is_null()) {
            F_terms.push_back(G);
            auto op = SymEngine::rcp_dynamic_cast<const Tinned::TwoElecOperator>(G);
            // We simply use the same name for two-electron operator and energy
            E_terms.push_back(Tinned::make_2el_energy(op->get_name(), op));
        }
        if (!Exc.is_null()) E_terms.push_back(Exc);
        if (!Fxc.is_null()) F_terms.push_back(Fxc);
        if (!hnuc.is_null()) E_terms.push_back(hnuc);

        if (E_terms.empty()) throw SymEngine::SymEngineException(
            "LagrangianDAO got null generalized energy!"
        );
        if (F_terms.empty()) throw SymEngine::SymEngineException(
            "LagrangianDAO got null generalized Fock operator!"
        );

        auto E = SymEngine::add(E_terms);
        F_ = SymEngine::matrix_add(F_terms);

        auto E_a = E->diff(a);
        auto D_a = D->diff(a);
        // The first term in Equation (98), J. Chem. Phys. 129, 214108 (2008)
        auto Ea_ = Tinned::remove_if(E_a, SymEngine::set_basic({D_a}));

        auto one_half = SymEngine::div(SymEngine::one, SymEngine::two);
        auto minus_one_half = SymEngine::div(SymEngine::minus_one, SymEngine::two);
        // Null overlap operator means it is an identity operator
        if (S.is_null()) {
            // \frac{\partial D}{\partial t}D-D\frac{\partial D}{\partial t}
            W_ = SymEngine::matrix_add({
                SymEngine::matrix_mul(SymEngine::vec_basic({D, F_, D})),
                SymEngine::matrix_mul(SymEngine::vec_basic({one_half, Dt_, D})),
                SymEngine::matrix_mul(SymEngine::vec_basic({minus_one_half, D, Dt_}))
            });
            // = D^{a}D-DD^{a}
            lambda_ =

            // Y = FD-DF-i\frac{\partial D}{\partial t}
            Y_ = SymEngine::matrix_add(SymEngine::vec_basic({
                SymEngine::matrix_mul(SymEngine::vec_basic({F_, D})),
                SymEngine::matrix_mul(SymEngine::vec_basic({
                    SymEngine::minus_one, D, F_
                })),
                SymEngine::matrix_mul(SymEngine::vec_basic({
                    SymEngine::minus_one, Dt_
                }))
            }));

            // = F^{a}D+DF^{a}-F^{a}
            zeta_ =

            // Z = D*D-D
            Z_ = SymEngine::matrix_add(SymEngine::vec_basic({
                SymEngine::matrix_mul(SymEngine::vec_basic({D, D})),
                SymEngine::matrix_mul(SymEngine::vec_basic({SymEngine::minus_one, D}))
            }));
        }
        else {
            // Equation (95), J. Chem. Phys. 129, 214108 (2008)
            W_ = SymEngine::matrix_add({
                SymEngine::matrix_mul(SymEngine::vec_basic({D, F_, D})),
                SymEngine::matrix_mul(SymEngine::vec_basic({one_half, Dt_, S, D})),
                SymEngine::matrix_mul(SymEngine::vec_basic({minus_one_half, D, S, Dt_}))
            });
            // Equation (220), J. Chem. Phys. 129, 214108 (2008)
            lambda_ = SymEngine::matrix_add(SymEngine::vec_basic({
                SymEngine::matrix_mul(SymEngine::vec_basic({D_a, S, D})),
                SymEngine::matrix_mul(SymEngine::vec_basic({
                    SymEngine::minus_one, D, S, D_a
                }))
            }));
            // Equation (229), J. Chem. Phys. 129, 214108 (2008)
            Y_ = SymEngine::matrix_add(SymEngine::vec_basic({
                SymEngine::matrix_mul(SymEngine::vec_basic({F_, D, S})),
                SymEngine::matrix_mul(SymEngine::vec_basic({
                    SymEngine::minus_one, S, D, F_
                })),
                SymEngine::matrix_mul(SymEngine::vec_basic({
                    SymEngine::minus_one, S, Dt_, S
                })),
                SymEngine::matrix_mul(SymEngine::vec_basic({
                    minus_one_half, St_, D, S
                })),
                SymEngine::matrix_mul(SymEngine::vec_basic({
                    minus_one_half, S, D, St_
                }))
            }));
            // Equation (224), J. Chem. Phys. 129, 214108 (2008)
            auto F_a = F_->diff(a);
            auto S_a = S->diff(a);
            zeta_ = SymEngine::matrix_add(SymEngine::vec_basic({
                SymEngine::matrix_mul(SymEngine::vec_basic({F_a, D, S})),
                SymEngine::matrix_mul(SymEngine::vec_basic({
                    SymEngine::minus_one, F, D, S_a
                })),
                SymEngine::matrix_mul(SymEngine::vec_basic({one_half, St_, D, S_a})),
                SymEngine::matrix_mul(SymEngine::vec_basic({S, Dt_, S_a})),
                SymEngine::matrix_mul(SymEngine::vec_basic({S, D, F_a})),
                SymEngine::matrix_mul(SymEngine::vec_basic({
                    SymEngine::minus_one, S_a, D, F_
                })),
                SymEngine::matrix_mul(SymEngine::vec_basic({
                    minus_one_half, S_a, D, St_
                })),
                SymEngine::matrix_mul(SymEngine::vec_basic({
                    SymEngine::minus_one, S_a, Dt_, S
                })),
                SymEngine::matrix_mul(SymEngine::vec_basic({SymEngine::minus_one, F_a}))
            }));
            // Equation (230), J. Chem. Phys. 129, 214108 (2008)
            Z_ = SymEngine::matrix_add(SymEngine::vec_basic({
                SymEngine::matrix_mul(SymEngine::vec_basic({D, S, D})),
                SymEngine::matrix_mul(SymEngine::vec_basic({SymEngine::minus_one, D}))
            }));
            // Equation (216), J. Chem. Phys. 129, 214108 (2008)
            La_ = ;
        }
    }

    SymEngine::RCP<const SymEngine::Basic> LagrangianDAO::get_response_functions(
        const SymEngine::multiset_basic& perturbations,
        const unsigned int k
    ) noexcept
    {
        auto Ea = Ea_;
        auto W = SymEngine::matrix_symbol(std::string("W"));
        auto SW = SymEngine::matrix_mul(SymEngine::vec_basic({S_, W}));
        auto SaW = SymEngine::matrix_mul();
        auto lambda = SymEngine::matrix_symbol(std::string("lambda"));
        auto zeta = SymEngine::matrix_symbol(std::string("zeta"));
        for (const auto& p: perturbations) {
            Ea = Ea->diff(p);
        }
        if (k > 0) {
            // `perturbations.size()` == k+n+1
            unsigned int kmax = perturbations.size()%2 == 0
                ? perturbations.size()/2 : (perturbations.size()-1)/2;
            if (k > kmax)
            unsigned int n = perturbations.size()-k-1;
        }
    }

    //TODO: double residue problem happens for Hessian of excited states
}
