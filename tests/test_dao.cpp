#define CATCH_CONFIG_MAIN

#include <string>
#include <utility>

#include <catch2/catch.hpp>

#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/constants.h>
#include <symengine/dict.h>
#include <symengine/matrices/matrix_add.h>
#include <symengine/matrices/matrix_mul.h>
#include <symengine/matrices/trace.h>

#include <Tinned.hpp>

#include "SymResponse.hpp"

//using namespace Tinned;
using namespace SymResponse;

TEST_CASE("Test density matrix-based response theory", "[LagrangianDAO]")
{
    auto a = Tinned::make_perturbation(std::string("a"));
    auto b = Tinned::make_perturbation(std::string("b"));
    auto c = Tinned::make_perturbation(std::string("c"));
    auto dependencies = Tinned::PertDependency({
        std::make_pair(a, 99), std::make_pair(b, 99), std::make_pair(c, 99)
    });
    auto D = Tinned::make_1el_density(std::string("D"));
    auto S = Tinned::make_1el_operator(std::string("S"), dependencies);
    auto h = Tinned::make_1el_operator(std::string("h"), dependencies);
    auto V = Tinned::make_1el_operator(std::string("V"), dependencies);
    auto T = Tinned::make_t_matrix(dependencies);
    auto G = Tinned::make_2el_operator(std::string("G"), D, dependencies);
    auto weight = Tinned::make_nonel_function(std::string("weight"));
    auto Omega = Tinned::make_1el_operator(std::string("Omega"), dependencies);
    auto Exc = Tinned::make_xc_energy(std::string("Exc"), D, Omega, weight);
    auto Fxc = Tinned::make_xc_potential(std::string("Fxc"), D, Omega, weight);
    auto hnuc = Tinned::make_nonel_function(std::string("hnuc"), dependencies);

    auto lagrangian = LagrangianDAO(
        a, D, S, SymEngine::vec_basic({h, V, T}), G, Exc, Fxc, hnuc
    );

    // Compute Equation (235), J. Chem. Phys. 129, 214108 (2008)
    auto result = lagrangian.get_response_functions(
        Tinned::PerturbationTuple({b, c}), {}, 2
    );

    auto E = lagrangian.get_generalized_energy();
    auto W = lagrangian.get_ew_density();
    auto lambda = lagrangian.get_tdscf_multiplier();
    auto Y = lagrangian.get_tdscf_equation();
    auto zeta = lagrangian.get_idempotency_multiplier();
    auto Z = lagrangian.get_idempotency_constraint();
    auto D_a = D->diff(a);
    auto D_bc = (D->diff(b))->diff(c);
    auto E_a = E->diff(a);
    // The first term in Equation (98), J. Chem. Phys. 129, 214108 (2008)
    auto E_0a = Tinned::remove_if(E_a, SymEngine::set_basic({D_a}));
    auto E_0a_bc = (E_0a->diff(b))->diff(c);
    auto S_a = S->diff(a);
    auto S_ab = S_a->diff(b);
    auto S_ac = S_a->diff(c);
    auto S_abc = S_ab->diff(c);
    auto W_b = W->diff(b);
    auto W_c = W->diff(c);
    auto W_bc_1 = Tinned::remove_if(W_b->diff(c), SymEngine::set_basic({D_bc}));
    auto Y_bc_1 = Tinned::remove_if((Y->diff(b))->diff(c), SymEngine::set_basic({D_bc}));
    auto Z_bc_1 = Tinned::remove_if((Z->diff(b))->diff(c), SymEngine::set_basic({D_bc}));

    // Equation (235), J. Chem. Phys. 129, 214108 (2008)
    auto ref = SymEngine::add(SymEngine::vec_basic({
        Tinned::remove_if(E_0a_bc, SymEngine::set_basic({D_bc})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_abc, W})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ac, W_b})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ab, W_c})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_a, W_bc_1})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, lambda, Y_bc_1})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, zeta, Z_bc_1}))
    }));

    REQUIRE(SymEngine::eq(*result, *ref));
}
