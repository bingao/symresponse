#define CATCH_CONFIG_MAIN
#include <iostream>

#include <string>
#include <utility>

#include <catch2/catch.hpp>

#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/real_double.h>
#include <symengine/complex_double.h>
#include <symengine/constants.h>
#include <symengine/dict.h>
#include <symengine/symbol.h>
#include <symengine/matrices/matrix_add.h>
#include <symengine/matrices/matrix_mul.h>
#include <symengine/matrices/trace.h>

#include <Tinned.hpp>

#include "SymResponse.hpp"

//using namespace Tinned;
using namespace SymResponse;

TEST_CASE("Test L^{abc} with S^{a} = 0", "[LagrangianDAO]")
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
    auto bas_dependencies = Tinned::PertDependency({
        std::make_pair(b, 99), std::make_pair(c, 99)
    });

    // Set different operators and note that `S` and `T` do not depend `a`
    auto D = Tinned::make_1el_density(std::string("D"));
    auto S = Tinned::make_1el_operator(std::string("S"), bas_dependencies);
    auto h = Tinned::make_1el_operator(std::string("h"), dependencies);
    auto V = Tinned::make_1el_operator(std::string("V"), dependencies);
    auto T = Tinned::make_t_matrix(bas_dependencies);
    auto G = Tinned::make_2el_operator(std::string("G"), D, dependencies);
    auto weight = Tinned::make_nonel_function(std::string("weight"), dependencies);
    auto Omega = Tinned::make_1el_operator(std::string("Omega"), bas_dependencies);
    auto Exc = Tinned::make_xc_energy(std::string("Exc"), D, Omega, weight);
    auto Fxc = Tinned::make_xc_potential(std::string("Fxc"), D, Omega, weight);
    auto hnuc = Tinned::make_nonel_function(std::string("hnuc"), dependencies);

    // Create quasi-energy derivative Lagrangian
    auto La = LagrangianDAO(
        a, D, S, SymEngine::vec_basic({h, V, T}), G, Exc, Fxc, hnuc
    );

    // Get generalized energy, Lagrangian multipliers, TDSCF equation and
    // idempotency constraint, used for verifying response functions
    auto E = La.get_generalized_energy();
    auto lambda = La.get_tdscf_multiplier();
    auto zeta = La.get_idempotency_multiplier();
    auto Y = La.get_tdscf_equation();
    auto Z = La.get_idempotency_constraint();

    auto h_a = h->diff(a);
    auto V_a = V->diff(a);
    auto F = SymEngine::matrix_add(SymEngine::vec_basic({h, V, T, G, Fxc}));
    auto F_a = Tinned::differentiate(F, Tinned::PertMultichain({a}));
    REQUIRE(SymEngine::eq(
        *F_a,
        *SymEngine::matrix_add({h_a, V_a, G->diff(a), Fxc->diff(a)})
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
    auto L_abc_0_2 = La.get_response_functions(
        Tinned::PertMultichain({b, c}), {}, 3
    );

    // Compute only the first term of Equation (237), J. Chem. Phys. 129, 214108 (2008)
    auto D_a = D->diff(a);
    auto E_a = Tinned::differentiate(E, Tinned::PertMultichain({a}));
    // The first term in Equation (98), J. Chem. Phys. 129, 214108 (2008)
    auto E_0a = Tinned::remove_if(E_a, SymEngine::set_basic({D_a}));
    REQUIRE(SymEngine::eq(
        *E_0a,
        *SymEngine::add({
            SymEngine::trace(SymEngine::matrix_mul({h_a, D})),
            SymEngine::trace(SymEngine::matrix_mul({V_a, D})),
            Tinned::make_2el_energy(
                SymEngine::make_rcp<const Tinned::TwoElecOperator>(
                    G->get_name(), D, dependencies, SymEngine::multiset_basic({a})
                )
            ),
            Tinned::remove_if(
                Tinned::differentiate(Exc, Tinned::PertMultichain({a})),
                SymEngine::set_basic({D_a})
            ),
            hnuc->diff(a)
        })
    ));
    auto E_0a_bc = Tinned::differentiate(E_0a, Tinned::PertMultichain({b, c}));

    REQUIRE(SymEngine::eq(*L_abc_0_2, *E_0a_bc));

    // Response function in Equation (235), J. Chem. Phys. 129, 214108 (2008)
    auto L_abc_1_1 = La.get_response_functions(
        Tinned::PertMultichain({b, c}), {}, 2
    );

    // Compute the first and last two terms of Equation (235), J. Chem. Phys. 129, 214108 (2008)
    auto D_bc = (D->diff(b))->diff(c);
    auto Y_bc_1 = Tinned::remove_if(
        Tinned::differentiate(Y, Tinned::PertMultichain({b, c})),
        SymEngine::set_basic({D_bc})
    );
    auto Z_bc_1 = Tinned::remove_if(
        Tinned::differentiate(Z, Tinned::PertMultichain({b, c})),
        SymEngine::set_basic({D_bc})
    );

    // `ref` is computed by following Equation (235), J. Chem. Phys. 129, 214108 (2008)
    auto ref = SymEngine::add({
        Tinned::remove_if(E_0a_bc, SymEngine::set_basic({D_bc})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, lambda, Y_bc_1})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, zeta, Z_bc_1}))
    });

    REQUIRE(SymEngine::eq(*L_abc_1_1, *Tinned::clean_temporum(ref)));
}

TEST_CASE("Test L^{abc} with different perturbation dependencies", "[LagrangianDAO]")
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
    auto bas_dependencies = Tinned::PertDependency({
        std::make_pair(a, 99), std::make_pair(b, 99)
    });

    // Set different operators
    auto D = Tinned::make_1el_density(std::string("D"));
    auto S = Tinned::make_1el_operator(std::string("S"), bas_dependencies);
    auto h = Tinned::make_1el_operator(
        std::string("h"),
        Tinned::PertDependency({std::make_pair(a, 99), std::make_pair(c, 99)})
    );
    auto V = Tinned::make_1el_operator(std::string("V"), dependencies);
    auto T = Tinned::make_t_matrix(bas_dependencies);
    auto G = Tinned::make_2el_operator(
        std::string("G"),
        D,
        Tinned::PertDependency({std::make_pair(b, 99), std::make_pair(c, 99)})
    );
    auto weight = Tinned::make_nonel_function(
        std::string("weight"), Tinned::PertDependency({std::make_pair(b, 99)})
    );
    auto Omega = Tinned::make_1el_operator(std::string("Omega"), bas_dependencies);
    auto Exc = Tinned::make_xc_energy(std::string("Exc"), D, Omega, weight);
    auto Fxc = Tinned::make_xc_potential(std::string("Fxc"), D, Omega, weight);
    auto hnuc = Tinned::make_nonel_function(
        std::string("hnuc"),
        Tinned::PertDependency({std::make_pair(b, 99), std::make_pair(c, 99)})
    );

    // Create quasi-energy derivative Lagrangian
    auto La = LagrangianDAO(
        a, D, S, SymEngine::vec_basic({h, V, T}), G, Exc, Fxc, hnuc
    );

    // Response function in Equation (237), J. Chem. Phys. 129, 214108 (2008)
    auto L_abc_0_2 = La.get_response_functions(
        Tinned::PertMultichain({b, c}), {}, 3
    );

    // We check simply (un)perturbed quantities in response functions
    auto S_a = S->diff(a);
    auto S_b = S->diff(b);
    auto S_ab = S_a->diff(b);
    REQUIRE(SymEngine::unified_eq(
        find_all(L_abc_0_2, S), SymEngine::vec_basic({S, S_a, S_b, S_ab})
    ));
    auto h_a = h->diff(a);
    auto h_c = h->diff(c);
    auto h_ac = h_a->diff(c);
    REQUIRE(SymEngine::unified_eq(
        find_all(L_abc_0_2, h), SymEngine::vec_basic({h, h_c, h_a, h_ac})
    ));
    auto T_a = T->diff(a);
    auto T_b = T->diff(b);
    auto T_ab = T_a->diff(b);
    REQUIRE(SymEngine::unified_eq(
        find_all(L_abc_0_2, T), SymEngine::vec_basic({T_b, T_a, T_ab})
    ));
    auto D_b = SymEngine::rcp_dynamic_cast<const Tinned::ElectronicState>(D->diff(b));
    auto D_c = SymEngine::rcp_dynamic_cast<const Tinned::ElectronicState>(D->diff(c));
    auto D_bc = SymEngine::rcp_dynamic_cast<const Tinned::ElectronicState>(D_b->diff(c));
    auto Gb_D = SymEngine::make_rcp<const Tinned::TwoElecOperator>(
        G->get_name(), D, G->get_dependencies(), SymEngine::multiset_basic({b})
    );
    auto G_Db = SymEngine::make_rcp<const Tinned::TwoElecOperator>(
        G->get_name(), D_b, G->get_dependencies()
    );
    auto Gc_D = SymEngine::make_rcp<const Tinned::TwoElecOperator>(
        G->get_name(), D, G->get_dependencies(), SymEngine::multiset_basic({c})
    );
    auto G_Dc = SymEngine::make_rcp<const Tinned::TwoElecOperator>(
        G->get_name(), D_c, G->get_dependencies()
    );
    auto Gbc_D = SymEngine::make_rcp<const Tinned::TwoElecOperator>(
        G->get_name(), D, G->get_dependencies(), SymEngine::multiset_basic({b, c})
    );
    auto Gb_Dc = SymEngine::make_rcp<const Tinned::TwoElecOperator>(
        G->get_name(), D_c, G->get_dependencies(), SymEngine::multiset_basic({b})
    );
    auto Gc_Db = SymEngine::make_rcp<const Tinned::TwoElecOperator>(
        G->get_name(), D_b, G->get_dependencies(), SymEngine::multiset_basic({c})
    );
    auto G_Dbc = SymEngine::make_rcp<const Tinned::TwoElecOperator>(
        G->get_name(), D_bc, G->get_dependencies()
    );
    REQUIRE(SymEngine::unified_eq(
        find_all(L_abc_0_2, G),
        SymEngine::vec_basic({G, G_Dbc, G_Dc, G_Db, Gc_Db, Gb_D, Gc_D, Gb_Dc, Gbc_D})
    ));
    auto weight_b = weight->diff(b);
    REQUIRE(SymEngine::unified_eq(
        find_all(L_abc_0_2, weight), SymEngine::vec_basic({weight, weight_b})
    ));
    auto Omega_a = Omega->diff(a);
    auto Omega_b = Omega->diff(b);
    auto Omega_ab = Omega_a->diff(b);
    REQUIRE(SymEngine::unified_eq(
        find_all(L_abc_0_2, Omega),
        SymEngine::vec_basic({Omega, Omega_a, Omega_b, Omega_ab})
    ));
    REQUIRE(find_all(L_abc_0_2, hnuc).empty());

    // Response function in Equation (235), J. Chem. Phys. 129, 214108 (2008)
    auto L_abc_1_1 = La.get_response_functions(
        Tinned::PertMultichain({b, c}), {}, 2
    );

    REQUIRE(SymEngine::unified_eq(
        find_all(L_abc_1_1, S), SymEngine::vec_basic({S, S_a, S_b, S_ab})
    ));
    REQUIRE(SymEngine::unified_eq(
        find_all(L_abc_1_1, h), SymEngine::vec_basic({h, h_c, h_a, h_ac})
    ));
    REQUIRE(SymEngine::unified_eq(
        find_all(L_abc_1_1, T), SymEngine::vec_basic({T_b, T_a, T_ab})
    ));
    // `G_Da` exists in the Lagrangian multiplier of idempotency constraint
    auto G_Da = SymEngine::make_rcp<const Tinned::TwoElecOperator>(
        G->get_name(),
        SymEngine::rcp_dynamic_cast<const Tinned::ElectronicState>(D->diff(a)),
        G->get_dependencies()
    );
    REQUIRE(SymEngine::unified_eq(
        find_all(L_abc_1_1, G),
        SymEngine::vec_basic({G, G_Dc, G_Db, G_Da, Gc_Db, Gb_D, Gc_D, Gb_Dc, Gbc_D})
    ));
    REQUIRE(SymEngine::unified_eq(
        find_all(L_abc_1_1, weight), SymEngine::vec_basic({weight, weight_b})
    ));
    REQUIRE(SymEngine::unified_eq(
        find_all(L_abc_1_1, Omega),
        SymEngine::vec_basic({Omega, Omega_a, Omega_b, Omega_ab})
    ));
    REQUIRE(find_all(L_abc_1_1, hnuc).empty());
}

TEST_CASE("Test L^{abc} with complex frequencies", "[LagrangianDAO]")
{
    // Set perturbations
    auto a = Tinned::make_perturbation(std::string("a"), SymEngine::complex_double(0.5, -0.1));
    auto b = Tinned::make_perturbation(std::string("b"), SymEngine::complex_double(-0.2, -0.2));
    auto c = Tinned::make_perturbation(std::string("c"), SymEngine::complex_double(-0.3, 0.3));
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
    auto weight = Tinned::make_nonel_function(std::string("weight"), dependencies);
    auto Omega = Tinned::make_1el_operator(std::string("Omega"), dependencies);
    auto Exc = Tinned::make_xc_energy(std::string("Exc"), D, Omega, weight);
    auto Fxc = Tinned::make_xc_potential(std::string("Fxc"), D, Omega, weight);
    auto hnuc = Tinned::make_nonel_function(std::string("hnuc"), dependencies);

    // Create quasi-energy derivative Lagrangian
    auto La = LagrangianDAO(
        a, D, S, SymEngine::vec_basic({h, V, T}), G, Exc, Fxc, hnuc
    );

    // Get generalized energy, generalized energy-weighted density matrix,
    // Lagrangian multipliers, TDSCF equation and idempotency constraint, used
    // for verifying response functions
    auto E = La.get_generalized_energy();
    auto W = La.get_ew_density();
    auto lambda = La.get_tdscf_multiplier();
    auto zeta = La.get_idempotency_multiplier();
    auto Y = La.get_tdscf_equation();
    auto Z = La.get_idempotency_constraint();

    // Response function in Equation (237), J. Chem. Phys. 129, 214108 (2008)
    auto L_abc_0_2 = La.get_response_functions(
        Tinned::PertMultichain({b, c}), {}, 3
    );

    // Compute each term of Equation (237), J. Chem. Phys. 129, 214108 (2008)
    auto D_a = D->diff(a);
    auto E_a = Tinned::differentiate(E, Tinned::PertMultichain({a}));
    // The first term in Equation (98), J. Chem. Phys. 129, 214108 (2008)
    auto E_0a = Tinned::remove_if(E_a, SymEngine::set_basic({D_a}));
    auto E_0a_bc = Tinned::differentiate(E_0a, Tinned::PertMultichain({b, c}));
    auto S_a = S->diff(a);
    auto S_ab = S_a->diff(b);
    auto S_ac = S_a->diff(c);
    auto S_abc = S_ab->diff(c);
    auto W_b = Tinned::differentiate(W, Tinned::PertMultichain({b}));
    auto W_c = Tinned::differentiate(W, Tinned::PertMultichain({c}));
    auto W_bc = Tinned::differentiate(W_b, Tinned::PertMultichain({c}));

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
    auto L_abc_1_1 = La.get_response_functions(
        Tinned::PertMultichain({b, c}), {}, 2
    );

    // Compute each term of Equation (235), J. Chem. Phys. 129, 214108 (2008)
    auto D_bc = (D->diff(b))->diff(c);
    auto W_bc_1 = Tinned::remove_if(W_bc, SymEngine::set_basic({D_bc}));
    auto Y_bc_1 = Tinned::remove_if(
        Tinned::differentiate(Y, Tinned::PertMultichain({b, c})),
        SymEngine::set_basic({D_bc})
    );
    auto Z_bc_1 = Tinned::remove_if(
        Tinned::differentiate(Z, Tinned::PertMultichain({b, c})),
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

TEST_CASE("Test L^{abc} with zero frequency perturbations", "[LagrangianDAO]")
{
    auto a = Tinned::make_perturbation(std::string("a"), SymEngine::real_double(0.0));
    auto b = Tinned::make_perturbation(std::string("b"), SymEngine::real_double(0.0));
    auto c = Tinned::make_perturbation(std::string("c"), SymEngine::real_double(0.0));
    auto dependencies = Tinned::PertDependency({
        std::make_pair(a, 99),
        std::make_pair(b, 99),
        std::make_pair(c, 99)
    });

    auto D = Tinned::make_1el_density(std::string("D"));
    auto S = Tinned::make_1el_operator(std::string("S"), dependencies);
    auto h = Tinned::make_1el_operator(std::string("h"), dependencies);
    auto V = Tinned::make_1el_operator(std::string("V"), dependencies);
    // We provide `T` matrix, but it will never survive due to zero frequency perturbations
    auto T = Tinned::make_t_matrix(dependencies);
    auto G = Tinned::make_2el_operator(std::string("G"), D, dependencies);
    auto weight = Tinned::make_nonel_function(std::string("weight"), dependencies);
    auto Omega = Tinned::make_1el_operator(std::string("Omega"), dependencies);
    auto Exc = Tinned::make_xc_energy(std::string("Exc"), D, Omega, weight);
    auto Fxc = Tinned::make_xc_potential(std::string("Fxc"), D, Omega, weight);
    auto hnuc = Tinned::make_nonel_function(std::string("hnuc"), dependencies);

    // Create quasi-energy derivative Lagrangian
    auto La = LagrangianDAO(
        a, D, S, SymEngine::vec_basic({h, V, T}), G, Exc, Fxc, hnuc
    );

    // Fock matrix without `T` matrix
    auto F = SymEngine::matrix_add(SymEngine::vec_basic({h, V, G, Fxc}));

    auto S_a = S->diff(a);
    auto F_a = Tinned::differentiate(F, Tinned::PertMultichain({a}));

    // Generalized energy, generalized energy-weighted density matrix,
    // Lagrangian multipliers and TDSCF equation and idempotency constraint,
    // without `T` matrix and time-differentiated quantities
    auto E = Tinned::remove_if(
        La.get_generalized_energy(), SymEngine::set_basic({T})
    );
    auto W = SymEngine::matrix_mul({D, F, D});
    auto lambda = La.get_tdscf_multiplier();
    auto zeta = SymEngine::matrix_add({
        SymEngine::matrix_mul({F_a, D, S}),
        SymEngine::matrix_mul({SymEngine::minus_one, F, D, S_a}),
        SymEngine::matrix_mul({S, D, F_a}),
        SymEngine::matrix_mul({SymEngine::minus_one, S_a, D, F}),
        SymEngine::matrix_mul({SymEngine::minus_one, F_a})
    });
    auto Y = SymEngine::matrix_add({
        SymEngine::matrix_mul({F, D, S}),
        SymEngine::matrix_mul({SymEngine::minus_one, S, D, F})
    });
    auto Z = La.get_idempotency_constraint();

    // Response function in Equation (237), J. Chem. Phys. 129, 214108 (2008)
    auto L_abc_0_2 = La.get_response_functions(
        Tinned::PertMultichain({b, c}), {}, 3
    );

    // Compute each term of Equation (237), J. Chem. Phys. 129, 214108 (2008)
    auto D_a = D->diff(a);
    auto E_a = Tinned::differentiate(E, Tinned::PertMultichain({a}));
    // The first term in Equation (98), J. Chem. Phys. 129, 214108 (2008)
    auto E_0a = Tinned::remove_if(E_a, SymEngine::set_basic({D_a}));
    auto E_0a_bc = Tinned::differentiate(E_0a, Tinned::PertMultichain({b, c}));
    auto S_ab = S_a->diff(b);
    auto S_ac = S_a->diff(c);
    auto S_abc = S_ab->diff(c);
    auto W_b = Tinned::differentiate(W, Tinned::PertMultichain({b}));
    auto W_c = Tinned::differentiate(W, Tinned::PertMultichain({c}));
    auto W_bc = Tinned::differentiate(W_b, Tinned::PertMultichain({c}));

    // `ref` is computed by following Equation (237), J. Chem. Phys. 129, 214108 (2008)
    auto ref = SymEngine::add({
        E_0a_bc,
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_abc, W})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ac, W_b})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_ab, W_c})),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, S_a, W_bc}))
    });

    REQUIRE(SymEngine::eq(*L_abc_0_2, *ref));

    // Response function in Equation (235), J. Chem. Phys. 129, 214108 (2008)
    auto L_abc_1_1 = La.get_response_functions(
        Tinned::PertMultichain({b, c}), {}, 2
    );

    // Compute each term of Equation (235), J. Chem. Phys. 129, 214108 (2008)
    auto D_bc = (D->diff(b))->diff(c);
    auto W_bc_1 = Tinned::remove_if(W_bc, SymEngine::set_basic({D_bc}));
    auto Y_bc_1 = Tinned::remove_if(
        Tinned::differentiate(Y, Tinned::PertMultichain({b, c})),
        SymEngine::set_basic({D_bc})
    );
    auto Z_bc_1 = Tinned::remove_if(
        Tinned::differentiate(Z, Tinned::PertMultichain({b, c})),
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

    REQUIRE(SymEngine::eq(*L_abc_1_1, *ref));
}

TEST_CASE("Test L^{abcd} with complex/zero/imaginary/real frequencies", "[LagrangianDAO]")
{
    auto a = Tinned::make_perturbation(std::string("a"), SymEngine::complex_double(0.5, -0.1));
    auto b = Tinned::make_perturbation(std::string("b"), SymEngine::complex_double(0.0, 0.0));
    auto c = Tinned::make_perturbation(std::string("c"), SymEngine::complex_double(0.0, 0.1));
    auto d = Tinned::make_perturbation(std::string("d"), SymEngine::complex_double(-0.5, 0.0));
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
    auto weight = Tinned::make_nonel_function(std::string("weight"), dependencies);
    auto Omega = Tinned::make_1el_operator(std::string("Omega"), dependencies);
    auto Exc = Tinned::make_xc_energy(std::string("Exc"), D, Omega, weight);
    auto Fxc = Tinned::make_xc_potential(std::string("Fxc"), D, Omega, weight);
    auto hnuc = Tinned::make_nonel_function(std::string("hnuc"), dependencies);

    auto La = LagrangianDAO(
        a, D, S, SymEngine::vec_basic({h, V, T}), G, Exc, Fxc, hnuc
    );

    auto E = La.get_generalized_energy();
    auto W = La.get_ew_density();
    auto lambda = La.get_tdscf_multiplier();
    auto zeta = La.get_idempotency_multiplier();
    auto Y = La.get_tdscf_equation();
    auto Z = La.get_idempotency_constraint();

    // Response function in Equation (241), J. Chem. Phys. 129, 214108 (2008)
    auto L_abcd_0_3 = La.get_response_functions(
        Tinned::PertMultichain({b, c, d}), {}, 4
    );

    auto D_a = D->diff(a);
    auto E_a = Tinned::differentiate(E, Tinned::PertMultichain({a}));
    auto E_0a = Tinned::remove_if(E_a, SymEngine::set_basic({D_a}));
    auto E_0a_bcd = Tinned::differentiate(E_0a, Tinned::PertMultichain({b, c, d}));
    auto S_a = S->diff(a);
    auto S_ab = S_a->diff(b);
    auto S_ac = S_a->diff(c);
    auto S_ad = S_a->diff(d);
    auto S_abc = S_ab->diff(c);
    auto S_abd = S_ab->diff(d);
    auto S_acd = S_ac->diff(d);
    auto S_abcd = S_abc->diff(d);
    auto W_b = Tinned::differentiate(W, Tinned::PertMultichain({b}));
    auto W_c = Tinned::differentiate(W, Tinned::PertMultichain({c}));
    auto W_d = Tinned::differentiate(W, Tinned::PertMultichain({d}));
    auto W_bc = Tinned::differentiate(W_b, Tinned::PertMultichain({c}));
    auto W_bd = Tinned::differentiate(W_b, Tinned::PertMultichain({d}));
    auto W_cd = Tinned::differentiate(W_c, Tinned::PertMultichain({d}));
    auto W_bcd = Tinned::differentiate(W_bc, Tinned::PertMultichain({d}));

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
    auto L_abcd_2_1 = La.get_response_functions(
        Tinned::PertMultichain({b, c, d}), {}, 2
    );

    auto D_bc = (D->diff(b))->diff(c);
    auto D_bd = (D->diff(b))->diff(d);
    auto D_cd = (D->diff(c))->diff(d);
    auto D_bcd = D_bc->diff(d);
    auto W_bc_1 = Tinned::remove_if(W_bc, SymEngine::set_basic({D_bc}));
    auto W_bd_1 = Tinned::remove_if(W_bd, SymEngine::set_basic({D_bd}));
    auto W_cd_1 = Tinned::remove_if(W_cd, SymEngine::set_basic({D_cd}));
    auto W_bcd_1 = Tinned::remove_if(W_bcd, SymEngine::set_basic({D_bc, D_bd, D_cd, D_bcd}));
    auto Y_bc = Tinned::differentiate(Y, Tinned::PertMultichain({b, c}));
    auto Y_bc_1 = Tinned::remove_if(Y_bc, SymEngine::set_basic({D_bc}));
    auto Y_bd_1 = Tinned::remove_if(
        Tinned::differentiate(Y, Tinned::PertMultichain({b, d})),
        SymEngine::set_basic({D_bd})
    );
    auto Y_cd_1 = Tinned::remove_if(
        Tinned::differentiate(Y, Tinned::PertMultichain({c, d})),
        SymEngine::set_basic({D_cd})
    );
    auto Y_bcd = Tinned::differentiate(Y_bc, Tinned::PertMultichain({d}));
    auto Y_bcd_1 = Tinned::remove_if(Y_bcd, SymEngine::set_basic({D_bc, D_bd, D_cd, D_bcd}));
    auto Z_bc = Tinned::differentiate(Z, Tinned::PertMultichain({b, c}));
    auto Z_bc_1 = Tinned::remove_if(Z_bc, SymEngine::set_basic({D_bc}));
    auto Z_bd_1 = Tinned::remove_if(
        Tinned::differentiate(Z, Tinned::PertMultichain({b, d})),
        SymEngine::set_basic({D_bd})
    );
    auto Z_cd_1 = Tinned::remove_if(
        Tinned::differentiate(Z, Tinned::PertMultichain({c, d})),
        SymEngine::set_basic({D_cd})
    );
    auto Z_bcd = Tinned::differentiate(Z_bc, Tinned::PertMultichain({d}));
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
            Tinned::differentiate(lambda, Tinned::PertMultichain({d})),
            Y_bc_1
        })),
        SymEngine::trace(SymEngine::matrix_mul({
            SymEngine::minus_one,
            Tinned::differentiate(lambda, Tinned::PertMultichain({c})),
            Y_bd_1
        })),
        SymEngine::trace(SymEngine::matrix_mul({
            SymEngine::minus_one,
            Tinned::differentiate(lambda, Tinned::PertMultichain({b})),
            Y_cd_1
        })),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, lambda, Y_bcd_1})),
        SymEngine::trace(SymEngine::matrix_mul({
            SymEngine::minus_one,
            Tinned::differentiate(zeta, Tinned::PertMultichain({d})),
            Z_bc_1
        })),
        SymEngine::trace(SymEngine::matrix_mul({
            SymEngine::minus_one,
            Tinned::differentiate(zeta, Tinned::PertMultichain({c})),
            Z_bd_1
        })),
        SymEngine::trace(SymEngine::matrix_mul({
            SymEngine::minus_one,
            Tinned::differentiate(zeta, Tinned::PertMultichain({b})),
            Z_cd_1
        })),
        SymEngine::trace(SymEngine::matrix_mul({SymEngine::minus_one, zeta, Z_bcd_1}))
    });

    REQUIRE(SymEngine::eq(*L_abcd_2_1, *Tinned::clean_temporum(ref)));

    // Response function in Equation (240), J. Chem. Phys. 129, 214108 (2008)
    auto L_abcd_1_2 = La.get_response_functions(
        Tinned::PertMultichain({b, c, d}), {}, 3
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

//FIXME: will non-empty intensive perturbations give different results?
TEST_CASE("Test L^{abcd} with intensive perturbations", "[LagrangianDAO]")
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
    auto weight = Tinned::make_nonel_function(std::string("weight"), dependencies);
    auto Omega = Tinned::make_1el_operator(std::string("Omega"), dependencies);
    auto Exc = Tinned::make_xc_energy(std::string("Exc"), D, Omega, weight);
    auto Fxc = Tinned::make_xc_potential(std::string("Fxc"), D, Omega, weight);
    auto hnuc = Tinned::make_nonel_function(std::string("hnuc"), dependencies);

    auto La = LagrangianDAO(
        //a, D, S, SymEngine::vec_basic({h, V, T}), G, Exc, Fxc, hnuc
        a, D, S, SymEngine::vec_basic({h}), G
    );

    // Response function in Equations (239), (240) and (241), J. Chem. Phys. 129, 214108 (2008)
    auto L_abcd_2_1 = La.get_response_functions(
        Tinned::PertMultichain({b, c, d}), {}, 2
    );
    auto L_abcd_1_2 = La.get_response_functions(
        Tinned::PertMultichain({b, c, d}), {}, 3
    );
    auto L_abcd_0_3 = La.get_response_functions(
        Tinned::PertMultichain({b, c, d}), {}, 4
    );

    // Extensive perturbations a, b and c, intensive perturbations d
    auto La_bc_d_3 = La.get_response_functions(
        Tinned::PertMultichain({b, c}), Tinned::PertMultichain({d}), 3
    );
    auto La_bc_d_2 = La.get_response_functions(
        Tinned::PertMultichain({b, c}), Tinned::PertMultichain({d}), 2
    );

    REQUIRE(SymEngine::neq(*La_bc_d_3, *L_abcd_1_2));
    REQUIRE(SymEngine::neq(*La_bc_d_3, *L_abcd_2_1));
    REQUIRE(SymEngine::eq(*La_bc_d_3, *L_abcd_0_3));

    REQUIRE(SymEngine::neq(*La_bc_d_2, *L_abcd_2_1));
    REQUIRE(SymEngine::neq(*La_bc_d_2, *L_abcd_1_2));
    REQUIRE(SymEngine::neq(*La_bc_d_2, *L_abcd_0_3));

std::cout << "L_abcd_2_1\n" << Tinned::latexify(L_abcd_2_1, 20) << "\n\n";
std::cout << "La_bc_d_2\n" << Tinned::latexify(La_bc_d_2, 20) << "\n\n";
}

//FIXME: speical case, L^{fggg}_{2,1}

//std::cout << Tinned::latexify(result, 16) << "\n\n";

inline unsigned int
get_weight(
    const SymEngine::vec_basic& wfn_parameters,
    const SymEngine::vec_basic& multipliers
)
{
    return wfn_parameters.size()+multipliers.size();
}

TEST_CASE("Test the search for optimal elimination rules", "[LagrangianDAO]")
{
    auto omega_a = SymEngine::symbol("\\omega_\\alpha");
    auto omega_b = SymEngine::symbol("\\omega_\\beta");
    auto omega_c = SymEngine::symbol("\\omega_\\gamma");
    auto omega_d = SymEngine::symbol("\\omega_\\delta");
    auto a = Tinned::make_perturbation(std::string("a"), omega_a);
    auto b = Tinned::make_perturbation(std::string("b"), omega_b);
    auto c = Tinned::make_perturbation(std::string("c"), omega_c);
    auto d = Tinned::make_perturbation(std::string("d"), omega_d);

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
    auto weight = Tinned::make_nonel_function(std::string("weight"), dependencies);
    auto Omega = Tinned::make_1el_operator(std::string("Omega"), dependencies);
    auto Exc = Tinned::make_xc_energy(std::string("Exc"), D, Omega, weight);
    auto Fxc = Tinned::make_xc_potential(std::string("Fxc"), D, Omega, weight);
    auto hnuc = Tinned::make_nonel_function(std::string("hnuc"), dependencies);

    auto La = LagrangianDAO(
        a, D, S, SymEngine::vec_basic({h, V, T}), G, Exc, Fxc, hnuc
    );

    auto results = La.get_response_functions(
        Tinned::PertMultichain({b, c, d}), {}, {}, get_weight
    );
std::cout << "\n\nweight = " << results.first << "\n";
for (const auto& r: results.second) {
    std::cout << "elimination = " << r.first << "\n"
              << "response function = " << Tinned::latexify(r.second) << "\n";
}
std::cout << "\n\n";
}
