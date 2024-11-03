#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/matrices/matrix_mul.h>
#include <symengine/matrices/transpose.h>
#include <symengine/symengine_exception.h>

#include "SymResponse/LagrangianCC.hpp"

namespace SymResponse
{
    LagrangianCC::LagrangianCC(
        const SymEngine::RCP<const SymEngine::Basic>& H0,
        const SymEngine::vec_basic& V,
        const SymEngine::RCP<const Tinned::PerturbedParameter>& t,
        const SymEngine::RCP<const SymEngine::MatrixExpr>& tau,
        const SymEngine::RCP<const Tinned::PerturbedParameter>& multiplier
    ): V_(V), t_(t), multiplier_(multiplier)
    {
        auto T = SymEngine::matrix_mul({t, SymEngine::transpose(tau)});
        auto H_cc_terms = SymEngine::vec_basic({Tinned::make_cc_hamiltonian(T, H0)});
        auto multiplier_terms = SymEngine::vec_basic({
            Tinned::make_cc_hamiltonian(
                T, Tinned::make_adjoint_map(SymEngine::vec_basic({tau}), H0)
            )
        });
        for (const auto& op: V) {
            H_cc_terms.push_back(Tinned::make_cc_hamiltonian(T, op));
            multiplier_terms.push_back(
                Tinned::make_cc_hamiltonian(
                    T, Tinned::make_adjoint_map(SymEngine::vec_basic({tau}), op)
                )
            );
        }
        H_cc_ = SymEngine::add(H_cc_terms);
        auto terms = multiplier_terms;
        for (const auto& op: multiplier_terms)
            // Should be inner product, or transpose `multiplier`
            terms.push_back(SymEngine::matrix_mul({
                multiplier,
                SymEngine::matrix_mul({Tinned::make_conjugate_transpose(tau), op})
            }));
        rhs_multiplier_ = SymEngine::add(terms);
        terms = H_cc_terms;
        for (const auto& op: H_cc_terms)
            terms.push_back(SymEngine::matrix_mul({
                multiplier,
                SymEngine::matrix_mul({Tinned::make_conjugate_transpose(tau), op})
            }));
        terms.push_back(SymEngine::mul(
            SymEngine::minus_one,
            // Should be inner product, or transpose `multiplier`
            SymEngine::matrix_mul({
                multiplier, SymEngine::transpose(Tinned::make_dt_operator(t))
            })
        ));
        L_ = SymEngine::add(terms);
    }

    SymEngine::RCP<const SymEngine::Basic> LagrangianCC::get_lagrangian() const noexcept
    {
        return L_;
    }

    SymEngine::RCP<const SymEngine::Basic> LagrangianCC::eliminate_wavefunction_parameter(
        const SymEngine::RCP<const SymEngine::Basic>& L,
        const Tinned::PertMultichain& exten_perturbations,
        const unsigned int min_wfn_order
    )
    {
        return Tinned::eliminate(L, t_, exten_perturbations, min_wfn_order);
    }

    SymEngine::RCP<const SymEngine::Basic> LagrangianCC::eliminate_lagrangian_multipliers(
        const SymEngine::RCP<const SymEngine::Basic>& L,
        const Tinned::PertMultichain& exten_perturbations,
        const unsigned int min_multiplier_order
    )
    {
        return Tinned::eliminate(
            L, multiplier_, exten_perturbations, min_multiplier_order
        );
    }

    SymEngine::RCP<const SymEngine::Basic> LagrangianCC::at_zero_strength(
        const SymEngine::RCP<const SymEngine::Basic>& L
    )
    {
        // Remove undifferentiated perturbation operators and unperturbed
        // time-differentiated quantities
        return Tinned::clean_temporum(Tinned::remove_if(
            L, SymEngine::set_basic(V_.begin(), V_.end())
        ));
    }

    SymEngine::vec_basic LagrangianCC::get_wavefunction_parameter() const noexcept
    {
        return SymEngine::vec_basic({t_});
    }

    SymEngine::vec_basic LagrangianCC::get_lagrangian_multipliers() const noexcept
    {
        return SymEngine::vec_basic({multiplier_});
    }
}
