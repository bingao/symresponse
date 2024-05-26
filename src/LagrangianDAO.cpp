#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/number.h>
#include <symengine/constants.h>
#include <symengine/matrices/matrix_add.h>
#include <symengine/matrices/matrix_mul.h>
#include <symengine/matrices/trace.h>
#include <symengine/symengine_exception.h>

#include "SymResponse/LagrangianDAO.hpp"

namespace SymResponse
{
    LagrangianDAO::LagrangianDAO(
        const SymEngine::RCP<const Tinned::Perturbation>& a,
        const SymEngine::RCP<const Tinned::OneElecDensity>& D,
        const SymEngine::RCP<const Tinned::OneElecOperator>& S,
        const SymEngine::vec_basic& H,
        const SymEngine::RCP<const Tinned::TwoElecOperator>& G,
        const SymEngine::RCP<const Tinned::ExchCorrEnergy>& Exc,
        const SymEngine::RCP<const Tinned::ExchCorrPotential>& Fxc,
        const SymEngine::RCP<const Tinned::NonElecFunction>& hnuc,
        const bool sym_elimination
    ) : sym_elimination_(sym_elimination), a_(a), D_(D), S_(S)
    {
        SymEngine::vec_basic E_terms;
        SymEngine::vec_basic F_terms;
        if (!H.empty()) {
            //E_terms.push_back(SymEngine::trace(
            //    SymEngine::matrix_mul({SymEngine::matrix_add(H), D})
            //));
            for (const auto& op: H)
                E_terms.push_back(SymEngine::trace(SymEngine::matrix_mul({op, D})));
            F_terms = H;
        }
        if (!G.is_null()) {
            E_terms.push_back(Tinned::make_2el_energy(G));
            F_terms.push_back(G);
        }
        if (!Exc.is_null()) E_terms.push_back(Exc);
        if (!Fxc.is_null()) F_terms.push_back(Fxc);
        if (!hnuc.is_null()) E_terms.push_back(hnuc);

        if (E_terms.empty()) throw SymEngine::SymEngineException(
            "LagrangianDAO gets nothing for generalized energy!"
        );
        if (F_terms.empty()) throw SymEngine::SymEngineException(
            "LagrangianDAO gets nothing for generalized Fock operator!"
        );

        E_ = SymEngine::add(E_terms);
        F_ = SymEngine::matrix_add(F_terms);

        // |i\frac{\partial `D_`}{\partial t}>
        auto Dt = Tinned::make_dt_operator(D);
        auto D_a = D->diff(a);
        // `Tinned::differentiate` can remove zero quantities after differentiation
        auto F_a = Tinned::differentiate(F_, Tinned::PerturbationTuple({a}));

        auto one_half = SymEngine::div(SymEngine::one, SymEngine::two);
        auto minus_one_half = SymEngine::div(SymEngine::minus_one, SymEngine::two);
        // Null overlap operator means it is an identity operator
        if (S.is_null()) {
            throw SymEngine::SymEngineException(
                "LagrangianDAO has not implemented for orthonormal basis!"
            );
            //// = D^{a}D-DD^{a}
            //lambda_ =
            //// Y = FD-DF-i\frac{\partial D}{\partial t}
            //Y_ = SymEngine::matrix_add({
            //    SymEngine::matrix_mul({F_, D}),
            //    SymEngine::matrix_mul({SymEngine::minus_one, D, F_}),
            //    SymEngine::matrix_mul({SymEngine::minus_one, Dt})
            //});
            //// = F^{a}D+DF^{a}-F^{a}
            //zeta_ =
            //// Z = D*D-D
            //Z_ = SymEngine::matrix_add({
            //    SymEngine::matrix_mul({D, D}),
            //    SymEngine::matrix_mul({SymEngine::minus_one, D})
            //});
        }
        else {
            // |i\frac{\partial `S`}{\partial t}>
            auto St = Tinned::make_dt_operator(S);
            // Equation (95), J. Chem. Phys. 129, 214108 (2008)
            W_ = SymEngine::matrix_add({
                SymEngine::matrix_mul({D, F_, D}),
                SymEngine::matrix_mul({one_half, Dt, S, D}),
                SymEngine::matrix_mul({minus_one_half, D, S, Dt})
            });
            // Equation (220), J. Chem. Phys. 129, 214108 (2008)
            lambda_ = SymEngine::matrix_add({
                SymEngine::matrix_mul({D_a, S, D}),
                SymEngine::matrix_mul({SymEngine::minus_one, D, S, D_a})
            });
            // Equation (229), J. Chem. Phys. 129, 214108 (2008)
            Y_ = SymEngine::matrix_add({
                SymEngine::matrix_mul({F_, D, S}),
                SymEngine::matrix_mul({SymEngine::minus_one, S, D, F_}),
                SymEngine::matrix_mul({SymEngine::minus_one, S, Dt, S}),
                SymEngine::matrix_mul({minus_one_half, St, D, S}),
                SymEngine::matrix_mul({minus_one_half, S, D, St})
            });
            auto S_a = S->diff(a);
            // Equation (224), J. Chem. Phys. 129, 214108 (2008)
            zeta_ = Tinned::is_zero_quantity(S_a)
                  ? SymEngine::matrix_add({
                        SymEngine::matrix_mul({F_a, D, S}),
                        SymEngine::matrix_mul({S, D, F_a}),
                        SymEngine::matrix_mul({SymEngine::minus_one, F_a})
                    })
                  : SymEngine::matrix_add({
                        SymEngine::matrix_mul({F_a, D, S}),
                        SymEngine::matrix_mul({SymEngine::minus_one, F_, D, S_a}),
                        SymEngine::matrix_mul({one_half, St, D, S_a}),
                        SymEngine::matrix_mul({S, Dt, S_a}),
                        SymEngine::matrix_mul({S, D, F_a}),
                        SymEngine::matrix_mul({SymEngine::minus_one, S_a, D, F_}),
                        SymEngine::matrix_mul({minus_one_half, S_a, D, St}),
                        SymEngine::matrix_mul({SymEngine::minus_one, S_a, Dt, S}),
                        SymEngine::matrix_mul({SymEngine::minus_one, F_a})
                    });
            // Equation (230), J. Chem. Phys. 129, 214108 (2008)
            Z_ = SymEngine::matrix_add({
                SymEngine::matrix_mul({D, S, D}),
                SymEngine::matrix_mul({SymEngine::minus_one, D})
            });
        }
    }

    SymEngine::RCP<const SymEngine::Basic> LagrangianDAO::get_response_functions(
        const Tinned::PerturbationTuple& exten_perturbations,
        const Tinned::PerturbationTuple& inten_perturbations,
        const unsigned int min_wfn_extern
    )
    {
        // Verify all perturbations and we put `a` into extensive perturbations
        auto all_exten_perturbations = exten_perturbations;
        all_exten_perturbations.insert(a_);
        verify_perturbations(all_exten_perturbations, inten_perturbations);
        // Construct the response functions according to elimination rules
        if (sym_elimination_) {
            //FIXME: to implement for both cases: with/without intensive perturbations
            if (!inten_perturbations.empty()) throw SymEngine::SymEngineException(
                "LagrangianDAO::get_response_functions has not implemented for intensive perturbations!"
            );
        }
        else {
            if (S_.is_null()) throw SymEngine::SymEngineException(
                "LagrangianDAO::get_response_functions has not implemented for orthonormal basis!"
            );
            SymEngine::vec_basic constraints;
            auto S_a = S_->diff(a_);
            // Pulay term
            if (!Tinned::is_zero_quantity(S_a)) constraints.push_back(
                SymEngine::trace(SymEngine::matrix_mul({S_a, W_}))
            );
            // Make artificial multipliers for elimination
            auto lambda = Tinned::make_lagrangian_multiplier(
                std::string("tdscf-multiplier")
            );
            auto zeta = Tinned::make_lagrangian_multiplier(
                std::string("idempotency-multiplier")
            );
            // TDSCF equation and idempotency constraint
            constraints.push_back(
                SymEngine::trace(SymEngine::matrix_mul({lambda, Y_}))
            );
            constraints.push_back(
                SymEngine::trace(SymEngine::matrix_mul({zeta, Z_}))
            );
            // Time-averaged quasi-energy derivative Lagrangian
            auto La = SymEngine::add({
                // The first term in Equation (98), J. Chem. Phys. 129, 214108 (2008)
                Tinned::remove_if(
                    Tinned::differentiate(E_, Tinned::PerturbationTuple({a_})),
                    SymEngine::set_basic({D_->diff(a_)})
                ),
                SymEngine::mul(SymEngine::minus_one, SymEngine::add(constraints))
            });
            // Differentiate time-averaged quasi-energy derivative Lagrangian
            // and eliminate peturbed density matrices and multipliers
            auto result = diff_and_eliminate(
                La,
                exten_perturbations,
                inten_perturbations,
                D_,
                SymEngine::set_basic({lambda, zeta}),
                min_wfn_extern
            );
            // Replace artificial multipliers with real differentiated ones
            result = Tinned::replace_all<Tinned::LagMultiplier>(
                result,
                Tinned::TinnedBasicMap<Tinned::LagMultiplier>({
                    {lambda, lambda_}, {zeta, zeta_}
                })
            );
            // Remove unperturbed time-differentiated density and overlap
            // matrices, and replace their perturbed ones with corresponding
            // perturbed density and overlap matrices multiplied by sums of
            // perturbation frequencies
            return Tinned::clean_temporum(result);
        }
    }

    //TODO: double residue problem happens for Hessian of excited states
}
