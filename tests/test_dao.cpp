#define CATCH_CONFIG_MAIN

#include <iostream>

#include <string>
#include <utility>

#include <catch2/catch.hpp>

#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/real_double.h>
#include <symengine/constants.h>
#include <symengine/dict.h>
#include <symengine/matrices/matrix_add.h>
#include <symengine/matrices/matrix_mul.h>
#include <symengine/matrices/trace.h>

#include <Tinned.hpp>

#include "SymResponse.hpp"

//using namespace Tinned;
using namespace SymResponse;

TEST_CASE("Test third-order quasienergy derivative Lagrangian with S^{a} = 0", "[LagrangianDAO]")
{
    // Set perturbations
    auto a = Tinned::make_perturbation(std::string("a"), SymEngine::real_double(0.5));
    auto b = Tinned::make_perturbation(std::string("b"), SymEngine::real_double(-0.2));
    auto c = Tinned::make_perturbation(std::string("c"), SymEngine::real_double(-0.3));
    auto dependencies = Tinned::PertDependency({
        std::make_pair(a, 99),
        std::make_pair(b, 99),
        std::make_pair(c, 99)
    });

    // Set different operators and note that `S` and `T` do not depend `a`
    auto D = Tinned::make_1el_density(std::string("D"));
    auto S = Tinned::make_1el_operator(
        std::string("S"),
        Tinned::PertDependency({std::make_pair(b, 99), std::make_pair(c, 99)})
    );
    auto h = Tinned::make_1el_operator(std::string("h"), dependencies);
    auto V = Tinned::make_1el_operator(std::string("V"), dependencies);
    auto T = Tinned::make_t_matrix(
        Tinned::PertDependency({std::make_pair(b, 99), std::make_pair(c, 99)})
    );
    auto G = Tinned::make_2el_operator(std::string("G"), D, dependencies);
    auto weight = Tinned::make_nonel_function(std::string("weight"));
    auto Omega = Tinned::make_1el_operator(std::string("Omega"), dependencies);
    auto Exc = Tinned::make_xc_energy(std::string("Exc"), D, Omega, weight);
    auto Fxc = Tinned::make_xc_potential(std::string("Fxc"), D, Omega, weight);
    auto hnuc = Tinned::make_nonel_function(std::string("hnuc"), dependencies);

    // Create quasi-energy derivative Lagrangian
    auto lagrangian = LagrangianDAO(
        a, D, S, SymEngine::vec_basic({h, V, T}), G, Exc, Fxc, hnuc
    );

    // Get generalized energy, Lagrangian multipliers, TDSCF equation and
    // idempotency constraint, used for verifying response functions
    auto E = lagrangian.get_generalized_energy();
    auto lambda = lagrangian.get_tdscf_multiplier();
    auto zeta = lagrangian.get_idempotency_multiplier();
    auto Y = lagrangian.get_tdscf_equation();
    auto Z = lagrangian.get_idempotency_constraint();

    auto F = SymEngine::matrix_add(SymEngine::vec_basic({h, V, T, G, Fxc}));
    auto F_a = Tinned::differentiate(F, Tinned::PerturbationTuple({a}));
    REQUIRE(SymEngine::eq(
        *F_a,
        *SymEngine::matrix_add(SymEngine::vec_basic({
            h->diff(a), V->diff(a), G->diff(a), Fxc->diff(a)
        }))
    ));

    REQUIRE(SymEngine::eq(
        *zeta,
        *SymEngine::matrix_add({
            SymEngine::matrix_mul({F_a, D, S}),
            SymEngine::matrix_mul({S, D, F_a}),
            SymEngine::matrix_mul({SymEngine::minus_one, F_a})
        })
    ));

    // Response function in Equation (237), J. Chem. Phys. 129, 214108 (2008)
    auto L_abc_0_2 = lagrangian.get_response_functions(
        Tinned::PerturbationTuple({b, c}), {}, 3
    );
}

TEST_CASE("Test third-order quasienergy derivative Lagrangian", "[LagrangianDAO]")
{
    // Set perturbations
    auto a = Tinned::make_perturbation(std::string("a"), SymEngine::real_double(0.5));
    auto b = Tinned::make_perturbation(std::string("b"), SymEngine::real_double(-0.2));
    auto c = Tinned::make_perturbation(std::string("c"), SymEngine::real_double(-0.3));
    auto dependencies = Tinned::PertDependency({
        std::make_pair(a, 99),
        std::make_pair(b, 99),
        std::make_pair(c, 99)
    });

    // Set different operators
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

    // Create quasi-energy derivative Lagrangian
    auto lagrangian = LagrangianDAO(
        a, D, S, SymEngine::vec_basic({h, V, T}), G, Exc, Fxc, hnuc
    );

    // Get generalized energy, generalized energy-weighted density matrix,
    // Lagrangian multipliers, TDSCF equation and idempotency constraint, used
    // for verifying response functions
    auto E = lagrangian.get_generalized_energy();
    auto W = lagrangian.get_ew_density();
    auto lambda = lagrangian.get_tdscf_multiplier();
    auto zeta = lagrangian.get_idempotency_multiplier();
    auto Y = lagrangian.get_tdscf_equation();
    auto Z = lagrangian.get_idempotency_constraint();

    // Response function in Equation (237), J. Chem. Phys. 129, 214108 (2008)
    auto L_abc_0_2 = lagrangian.get_response_functions(
        Tinned::PerturbationTuple({b, c}), {}, 3
    );

    // Compute each term of Equation (237), J. Chem. Phys. 129, 214108 (2008)
    auto D_a = D->diff(a);
    auto E_a = Tinned::differentiate(E, Tinned::PerturbationTuple({a}));
    // The first term in Equation (98), J. Chem. Phys. 129, 214108 (2008)
    auto E_0a = Tinned::remove_if(E_a, SymEngine::set_basic({D_a}));
    auto E_0a_bc = Tinned::differentiate(E_0a, Tinned::PerturbationTuple({b, c}));
    auto S_a = S->diff(a);
    auto S_ab = S_a->diff(b);
    auto S_ac = S_a->diff(c);
    auto S_abc = S_ab->diff(c);
    auto W_b = Tinned::differentiate(W, Tinned::PerturbationTuple({b}));
    auto W_c = Tinned::differentiate(W, Tinned::PerturbationTuple({c}));
    auto W_bc = Tinned::differentiate(W_b, Tinned::PerturbationTuple({c}));

    // `ref` is computed by following Equation (237), J. Chem. Phys. 129, 214108 (2008)
    auto ref = SymEngine::add({
        E_0a_bc,
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_abc, W})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ac, W_b})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ab, W_c})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_a, W_bc}))
    });

    REQUIRE(SymEngine::eq(*L_abc_0_2, *Tinned::clean_temporum(ref)));

    // Response function in Equation (235), J. Chem. Phys. 129, 214108 (2008)
    auto L_abc_1_1 = lagrangian.get_response_functions(
        Tinned::PerturbationTuple({b, c}), {}, 2
    );

    // Compute each term of Equation (235), J. Chem. Phys. 129, 214108 (2008)
    auto D_bc = (D->diff(b))->diff(c);
    auto W_bc_1 = Tinned::remove_if(W_bc, SymEngine::set_basic({D_bc}));
    auto Y_bc_1 = Tinned::remove_if(
        Tinned::differentiate(Y, Tinned::PerturbationTuple({b, c})),
        SymEngine::set_basic({D_bc})
    );
    auto Z_bc_1 = Tinned::remove_if(
        Tinned::differentiate(Z, Tinned::PerturbationTuple({b, c})),
        SymEngine::set_basic({D_bc})
    );

    // `ref` is computed by following Equation (235), J. Chem. Phys. 129, 214108 (2008)
    ref = SymEngine::add({
        Tinned::remove_if(E_0a_bc, SymEngine::set_basic({D_bc})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_abc, W})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ac, W_b})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ab, W_c})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_a, W_bc_1})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, lambda, Y_bc_1})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, zeta, Z_bc_1}))
    });

    REQUIRE(SymEngine::eq(*L_abc_1_1, *Tinned::clean_temporum(ref)));
}

TEST_CASE("Test fouth-order quasienergy derivative Lagrangian", "[LagrangianDAO]")
{
    auto a = Tinned::make_perturbation(std::string("a"), SymEngine::real_double(0.6));
    auto b = Tinned::make_perturbation(std::string("b"), SymEngine::real_double(-0.1));
    auto c = Tinned::make_perturbation(std::string("c"), SymEngine::real_double(-0.2));
    auto d = Tinned::make_perturbation(std::string("d"), SymEngine::real_double(-0.3));
    auto dependencies = Tinned::PertDependency({
        std::make_pair(a, 99),
        std::make_pair(b, 99),
        std::make_pair(c, 99),
        std::make_pair(d, 99)
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

    auto E = lagrangian.get_generalized_energy();
    auto W = lagrangian.get_ew_density();
    auto lambda = lagrangian.get_tdscf_multiplier();
    auto zeta = lagrangian.get_idempotency_multiplier();
    auto Y = lagrangian.get_tdscf_equation();
    auto Z = lagrangian.get_idempotency_constraint();

    // Response function in Equation (241), J. Chem. Phys. 129, 214108 (2008)
    auto L_abcd_0_3 = lagrangian.get_response_functions(
        Tinned::PerturbationTuple({b, c, d}), {}, 4
    );

    auto D_a = D->diff(a);
    auto E_a = Tinned::differentiate(E, Tinned::PerturbationTuple({a}));
    auto E_0a = Tinned::remove_if(E_a, SymEngine::set_basic({D_a}));
    auto E_0a_bcd = Tinned::differentiate(E_0a, Tinned::PerturbationTuple({b, c, d}));
    auto S_a = S->diff(a);
    auto S_ab = S_a->diff(b);
    auto S_ac = S_a->diff(c);
    auto S_ad = S_a->diff(d);
    auto S_abc = S_ab->diff(c);
    auto S_abd = S_ab->diff(d);
    auto S_acd = S_ac->diff(d);
    auto S_abcd = S_abc->diff(d);
    auto W_b = Tinned::differentiate(W, Tinned::PerturbationTuple({b}));
    auto W_c = Tinned::differentiate(W, Tinned::PerturbationTuple({c}));
    auto W_d = Tinned::differentiate(W, Tinned::PerturbationTuple({d}));
    auto W_bc = Tinned::differentiate(W_b, Tinned::PerturbationTuple({c}));
    auto W_bd = Tinned::differentiate(W_b, Tinned::PerturbationTuple({d}));
    auto W_cd = Tinned::differentiate(W_c, Tinned::PerturbationTuple({d}));
    auto W_bcd = Tinned::differentiate(W_bc, Tinned::PerturbationTuple({d}));

    // `ref` is computed by following Equation (241), J. Chem. Phys. 129, 214108 (2008)
    auto ref = SymEngine::add({
        E_0a_bcd,
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_abcd, W})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_acd, W_b})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_abd, W_c})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_abc, W_d})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ad, W_bc})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ac, W_bd})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ab, W_cd})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_a, W_bcd}))
    });

    REQUIRE(SymEngine::eq(*L_abcd_0_3, *Tinned::clean_temporum(ref)));

    // Response function in Equation (239), J. Chem. Phys. 129, 214108 (2008)
    auto L_abcd_2_1 = lagrangian.get_response_functions(
        Tinned::PerturbationTuple({b, c, d}), {}, 2
    );

    auto D_bc = (D->diff(b))->diff(c);
    auto D_bd = (D->diff(b))->diff(d);
    auto D_cd = (D->diff(c))->diff(d);
    auto D_bcd = D_bc->diff(d);
    auto W_bc_1 = Tinned::remove_if(W_bc, SymEngine::set_basic({D_bc}));
    auto W_bd_1 = Tinned::remove_if(W_bd, SymEngine::set_basic({D_bd}));
    auto W_cd_1 = Tinned::remove_if(W_cd, SymEngine::set_basic({D_cd}));
    auto W_bcd_1 = Tinned::remove_if(W_bcd, SymEngine::set_basic({D_bc, D_bd, D_cd, D_bcd}));
    auto Y_bc = Tinned::differentiate(Y, Tinned::PerturbationTuple({b, c}));
    auto Y_bc_1 = Tinned::remove_if(Y_bc, SymEngine::set_basic({D_bc}));
    auto Y_bd_1 = Tinned::remove_if(
        Tinned::differentiate(Y, Tinned::PerturbationTuple({b, d})),
        SymEngine::set_basic({D_bd})
    );
    auto Y_cd_1 = Tinned::remove_if(
        Tinned::differentiate(Y, Tinned::PerturbationTuple({c, d})),
        SymEngine::set_basic({D_cd})
    );
    auto Y_bcd = Tinned::differentiate(Y_bc, Tinned::PerturbationTuple({d}));
    auto Y_bcd_1 = Tinned::remove_if(Y_bcd, SymEngine::set_basic({D_bc, D_bd, D_cd, D_bcd}));
    auto Z_bc = Tinned::differentiate(Z, Tinned::PerturbationTuple({b, c}));
    auto Z_bc_1 = Tinned::remove_if(Z_bc, SymEngine::set_basic({D_bc}));
    auto Z_bd_1 = Tinned::remove_if(
        Tinned::differentiate(Z, Tinned::PerturbationTuple({b, d})),
        SymEngine::set_basic({D_bd})
    );
    auto Z_cd_1 = Tinned::remove_if(
        Tinned::differentiate(Z, Tinned::PerturbationTuple({c, d})),
        SymEngine::set_basic({D_cd})
    );
    auto Z_bcd = Tinned::differentiate(Z_bc, Tinned::PerturbationTuple({d}));
    auto Z_bcd_1 = Tinned::remove_if(Z_bcd, SymEngine::set_basic({D_bc, D_bd, D_cd, D_bcd}));

    // `ref` is computed by following Equation (239), J. Chem. Phys. 129, 214108 (2008)
    ref = SymEngine::add({
        Tinned::remove_if(E_0a_bcd, SymEngine::set_basic({D_bc, D_bd, D_cd, D_bcd})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_abcd, W})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_acd, W_b})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_abd, W_c})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_abc, W_d})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ad, W_bc_1})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ac, W_bd_1})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ab, W_cd_1})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_a, W_bcd_1})),
        SymEngine::trace(SymEngine::matrix_mul({
            SymEngine::minus_one,
            Tinned::differentiate(lambda, Tinned::PerturbationTuple({d})),
            Y_bc_1
        })),
        SymEngine::trace(SymEngine::matrix_mul({
            SymEngine::minus_one,
            Tinned::differentiate(lambda, Tinned::PerturbationTuple({c})),
            Y_bd_1
        })),
        SymEngine::trace(SymEngine::matrix_mul({
            SymEngine::minus_one,
            Tinned::differentiate(lambda, Tinned::PerturbationTuple({b})),
            Y_cd_1
        })),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, lambda, Y_bcd_1})),
        SymEngine::trace(SymEngine::matrix_mul({
            SymEngine::minus_one,
            Tinned::differentiate(zeta, Tinned::PerturbationTuple({d})),
            Z_bc_1
        })),
        SymEngine::trace(SymEngine::matrix_mul({
            SymEngine::minus_one,
            Tinned::differentiate(zeta, Tinned::PerturbationTuple({c})),
            Z_bd_1
        })),
        SymEngine::trace(SymEngine::matrix_mul({
            SymEngine::minus_one,
            Tinned::differentiate(zeta, Tinned::PerturbationTuple({b})),
            Z_cd_1
        })),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, zeta, Z_bcd_1}))
    });

    REQUIRE(SymEngine::eq(*L_abcd_2_1, *Tinned::clean_temporum(ref)));

    // Response function in Equation (240), J. Chem. Phys. 129, 214108 (2008)
    auto L_abcd_1_2 = lagrangian.get_response_functions(
        Tinned::PerturbationTuple({b, c, d}), {}, 3
    );

    auto W_bcd_2 = Tinned::remove_if(W_bcd, SymEngine::set_basic({D_bcd}));
    auto Y_bcd_2 = Tinned::remove_if(Y_bcd, SymEngine::set_basic({D_bcd}));
    auto Z_bcd_2 = Tinned::remove_if(Z_bcd, SymEngine::set_basic({D_bcd}));

    // `ref` is computed by following Equation (240), J. Chem. Phys. 129, 214108 (2008)
    ref = SymEngine::add({
        Tinned::remove_if(E_0a_bcd, SymEngine::set_basic({D_bcd})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_abcd, W})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_acd, W_b})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_abd, W_c})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_abc, W_d})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ad, W_bc})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ac, W_bd})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ab, W_cd})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_a, W_bcd_2})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, lambda, Y_bcd_2})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, zeta, Z_bcd_2}))
    });

    REQUIRE(SymEngine::eq(*L_abcd_1_2, *Tinned::clean_temporum(ref)));
}

//FIXME: perturbation with zero frequency
//FIXME: some operators do not depend on all perturbations
// speical case, L^{fggg}_{2,1}
// non-empty intensive perturbations

//std::cout << Tinned::latexify(result, 16) << "\n\n";
