#include "SymResponse/LagrangianCC.hpp"

namespace SymResponse
{
    LagrangianCC::LagrangianCC(
        const SymEngine::RCP<const SymEngine::Basic>& H,
        const SymEngine::RCP<const Tinned::PerturbedParameter>& amplitudes,
        const SymEngine::RCP<const SymEngine::Basic>& excit_operators,
        const SymEngine::RCP<const Tinned::PerturbedParameter>& multipliers,
        const SymEngine::RCP<const SymEngine::Basic>& ref_state
    ):
    {
        auto excit_operators = Tinned::make_1el_operator(std::string("excit-operators"));
        auto amplitudes = Tinned::make_perturbed_parameter(std::string("amplitudes"));
        auto multipliers = Tinned::make_perturbed_parameter(std::string("multipliers"));

        T_ = SymEngine::matrix_mul({
            amplitudes,
            SymEngine::transpose(excit_operators)
        });

        //auto eadj_H = make_eadj_hamiltonian(std::string("eadj(H)"), T, H);
        eadj_H_ = make_eadj_hamiltonian(T, H);

        L_ = SymEngine::add({
            eadj_H_,
            SymEngine::matrix_mul({
                multipliers,
                SymEngine::matrix_mul({
                    // We transpose `excit_operators` and then take
                    // conjugate transpose of each element, see Equation
                    // (446), Chem. Rev. 2012, 112, 543-631
                    //
                    // Conjugate transpose
                    // Size(m, n*Z) [A_{m,n}, B_{m,n}, ..., Z_{m,n}]
                    // Size(n*Z, m) column vector and element A_{m,n}^{H}
                    Tinned::conjugate_transpose(excit_operators),
                    eadj_H_
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
}
