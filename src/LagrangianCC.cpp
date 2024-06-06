#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/constants.h>
#include <symengine/matrices/matrix_mul.h>
#include <symengine/matrices/transpose.h>
#include <symengine/symengine_exception.h>

#include "SymResponse/LagrangianCC.hpp"

namespace SymResponse
{
    LagrangianCC::LagrangianCC(
        const SymEngine::RCP<const SymEngine::Basic>& H,
        const SymEngine::RCP<const Tinned::PerturbedParameter>& amplitudes,
        const SymEngine::RCP<const SymEngine::MatrixExpr>& excit_operators,
        const SymEngine::RCP<const Tinned::PerturbedParameter>& multipliers,
        const SymEngine::RCP<const SymEngine::Basic>& ref_state
    ): amplitudes_(amplitudes), multipliers_(multipliers)
    {
        if (!ref_state.is_null()) throw SymEngine::SymEngineException(
            "LagrangianCC has not implemented for reference state!"
        );
        auto T = SymEngine::matrix_mul({
            amplitudes, SymEngine::transpose(excit_operators)
        });
        auto cc_hamiltonian = Tinned::make_cc_hamiltonian(T, H);
        L_ = SymEngine::add({
            cc_hamiltonian,
            SymEngine::matrix_mul({
                multipliers,
                SymEngine::matrix_mul({
                    Tinned::make_conjugate_transpose(excit_operators), cc_hamiltonian
                })
            }),
            SymEngine::mul(
                SymEngine::minus_one,
                SymEngine::matrix_mul({
                    multipliers,
                    SymEngine::transpose(Tinned::make_dt_operator(amplitudes))
                })
            )
        });
    }

    SymEngine::RCP<const SymEngine::Basic> LagrangianCC::get_lagrangian() const noexcept
    {
        return L_;
    }

    SymEngine::RCP<const SymEngine::Basic> LagrangianCC::eliminate_wavefunction_parameter(
        const SymEngine::RCP<const SymEngine::Basic>& L,
        const Tinned::PerturbationTuple& exten_perturbations,
        const unsigned int min_wfn_order
    )
    {
        return Tinned::eliminate(L, amplitudes_, exten_perturbations, min_wfn_order);
    }

    SymEngine::RCP<const SymEngine::Basic> LagrangianCC::eliminate_lagrangian_multipliers(
        const SymEngine::RCP<const SymEngine::Basic>& L,
        const Tinned::PerturbationTuple& exten_perturbations,
        const unsigned int min_multiplier_order
    )
    {
        return Tinned::eliminate(
            L, multipliers_, exten_perturbations, min_multiplier_order
        );
    }
}
