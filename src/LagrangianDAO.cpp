#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/number.h>
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
        //FIXME: to implement for both cases: with/without intensive perturbations
        if (sym_elimination_) throw SymEngine::SymEngineException(
            "LagrangianDAO has not implemented for almost symmetric form of elimination!"
        );

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
        auto F_a = Tinned::differentiate(F_, Tinned::PertMultichain({a}));

        // Terms of differentiated quasi-energy derivative Lagrangian with
        // respect to the perturbation `a`
        SymEngine::vec_basic La_terms;

        auto one_half = SymEngine::div(SymEngine::one, SymEngine::two);
        auto minus_one_half = SymEngine::div(SymEngine::minus_one, SymEngine::two);
        // Null overlap operator means it is an identity operator
        if (S.is_null()) {
            // = D^{a}D-DD^{a}
            lambda_ = SymEngine::matrix_add({
                SymEngine::matrix_mul({D_a, D}),
                SymEngine::matrix_mul({SymEngine::minus_one, D, D_a})
            });
            // Y = FD-DF-i\frac{\partial D}{\partial t}
            Y_ = SymEngine::matrix_add({
                SymEngine::matrix_mul({F_, D}),
                SymEngine::matrix_mul({SymEngine::minus_one, D, F_}),
                SymEngine::matrix_mul({SymEngine::minus_one, Dt})
            });
            // = F^{a}D+DF^{a}-F^{a}
            zeta_ = SymEngine::matrix_add({
                SymEngine::matrix_mul({F_a, D}),
                SymEngine::matrix_mul({D, F_a}),
                SymEngine::matrix_mul({SymEngine::minus_one, F_a})
            });
            // Z = D*D-D
            Z_ = SymEngine::matrix_add({
                SymEngine::matrix_mul({D, D}),
                SymEngine::matrix_mul({SymEngine::minus_one, D})
            });
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
            // Pulay term
            auto S_a = S->diff(a);
            if (!Tinned::is_zero_quantity(S_a)) La_terms.push_back(
                SymEngine::trace(SymEngine::matrix_mul({S_a, W_}))
            );
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
        // Make an "artificial" Lagrangian multiplier for elimination
        tdscf_multiplier_ = Tinned::make_perturbed_parameter(
            std::string("tdscf-multiplier")
        );
        La_terms.push_back(
            SymEngine::trace(SymEngine::matrix_mul({tdscf_multiplier_, Y_}))
        );
        // Make an "artificial" Lagrangian multiplier for elimination
        idempotency_multiplier_ = Tinned::make_perturbed_parameter(
            std::string("idempotency-multiplier")
        );
        La_terms.push_back(
            SymEngine::trace(SymEngine::matrix_mul({idempotency_multiplier_, Z_}))
        );
        // Construct the time-averaged quasi-energy derivative Lagrangian
        La_ = SymEngine::add({
            // The first term in Equation (98), J. Chem. Phys. 129, 214108 (2008)
            Tinned::remove_if(
                Tinned::differentiate(E_, Tinned::PertMultichain({a_})),
                SymEngine::set_basic({D_->diff(a_)})
            ),
            SymEngine::mul(SymEngine::minus_one, SymEngine::add(La_terms))
        });
    }

    bool LagrangianDAO::validate_perturbation_frequencies(
        const Tinned::PertMultichain& exten_perturbations,
        const Tinned::PertMultichain& inten_perturbations,
        const SymEngine::RCP<const SymEngine::Number>& threshold
    ) const noexcept
    {
        // Here we need to inlcude the frequency of the perturbation `a_`,
        // which can either be extensive or intensive
        SymEngine::RCP<const SymEngine::Basic> sum_freq = a_->get_frequency();
        for (const auto& p: exten_perturbations)
            sum_freq = SymEngine::add(sum_freq, p->get_frequency());
        for (const auto& p: inten_perturbations)
            sum_freq = SymEngine::add(sum_freq, p->get_frequency());
        if (SymEngine::is_a_Number(*sum_freq)) {
            return Tinned::is_zero_number(
                SymEngine::rcp_dynamic_cast<const SymEngine::Number>(sum_freq),
                threshold
            );
        }
        else {
            // For nonnumerical frequencies, we simply return `true` and users
            // are responsible for the validation
            return true;
        }
    }

    SymEngine::RCP<const SymEngine::Basic> LagrangianDAO::get_lagrangian() const noexcept
    {
        return La_;
    }

    SymEngine::RCP<const SymEngine::Basic> LagrangianDAO::eliminate_wavefunction_parameter(
        const SymEngine::RCP<const SymEngine::Basic>& L,
        const Tinned::PertMultichain& exten_perturbations,
        const unsigned int min_wfn_order
    )
    {
        return Tinned::eliminate(L, D_, exten_perturbations, min_wfn_order);
    }

    SymEngine::RCP<const SymEngine::Basic> LagrangianDAO::eliminate_lagrangian_multipliers(
        const SymEngine::RCP<const SymEngine::Basic>& L,
        const Tinned::PertMultichain& exten_perturbations,
        const unsigned int min_multiplier_order
    )
    {
        auto result = Tinned::eliminate(
            L, tdscf_multiplier_, exten_perturbations, min_multiplier_order
        );
        result = Tinned::eliminate(
            result, idempotency_multiplier_, exten_perturbations, min_multiplier_order
        );
        // Replace "artificial" multipliers with real differentiated ones
        return Tinned::replace_all<Tinned::PerturbedParameter>(
            result,
            Tinned::TinnedBasicMap<Tinned::PerturbedParameter>({
                {tdscf_multiplier_, lambda_}, {idempotency_multiplier_, zeta_}
            })
        );
    }

    //TODO: double residue problem happens for Hessian of excited states
}
