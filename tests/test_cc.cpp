#define CATCH_CONFIG_MAIN
#include <iostream>

#include <string>
#include <utility>

#include <catch2/catch.hpp>

#include <symengine/dict.h>
#include <symengine/symbol.h>

#include <Tinned.hpp>

#include "SymResponse.hpp"

using namespace SymResponse;

TEST_CASE("Test the second and third order response functions", "[LagrangianCC]")
{
    // Set perturbations
    auto omega_a = SymEngine::symbol("omega_A");
    auto omega_b = SymEngine::symbol("omega_B");
    auto omega_c = SymEngine::symbol("omega_C");
    auto a = Tinned::make_perturbation(std::string("a"), omega_a);
    auto b = Tinned::make_perturbation(std::string("b"), omega_b);
    auto c = Tinned::make_perturbation(std::string("c"), omega_c);

    // Unperturbed Hamiltonian, perturbation operators, amplitudes, excitation
    // operators and multipliers
    auto H0 = Tinned::make_1el_operator(std::string("H_0"));
    auto Va = Tinned::make_1el_operator(
        std::string("V_A"), Tinned::PertDependency({std::make_pair(a, 1)})
    );
    auto Vb = Tinned::make_1el_operator(
        std::string("V_B"), Tinned::PertDependency({std::make_pair(b, 1)})
    );
    auto Vc = Tinned::make_1el_operator(
        std::string("V_C"), Tinned::PertDependency({std::make_pair(c, 1)})
    );
    auto t = Tinned::make_perturbed_parameter(std::string("t"));
    auto tau = Tinned::make_1el_operator(std::string("\\tau"));
    auto multiplier = Tinned::make_perturbed_parameter(std::string("\\lambda"));

    // Create quasi-energy Lagrangian
    auto L = LagrangianCC(H0, SymEngine::vec_basic({Va, Vb, Vc}), t, tau, multiplier);

    // Response function <<A;B>>, no intensive perturbations
    auto L_ab = L.get_response_functions(Tinned::PertMultichain({a, b}), {}, 0);
    std::cout << "\\langle\\langle A;B\\rangle\\rangle_{\\omega_{B}} = "
              << Tinned::latexify(L_ab) << "\n\n";

    auto t_a = t->diff(a);
    auto t_b = t->diff(b);

    auto rhs_t_a = L.get_response_rhs(t_a);
    std::cout << "-\\boldsymbol{\\xi}^{a}_{\\omega} = "
              << Tinned::latexify(rhs_t_a) << "\n\n";

    // Response function <<A;B,C>>, no intensive perturbations
    auto L_abc = L.get_response_functions(Tinned::PertMultichain({a, b, c}), {}, 0);
    std::cout << "\\langle\\langle A;B,C\\rangle\\rangle_{\\omega_{B},\\omega_{C}} = "
              << Tinned::latexify(L_abc) << "\n\n";

    auto t_c = t->diff(c);
    auto multiplier_a = multiplier->diff(a);
    auto multiplier_b = multiplier->diff(b);
    auto multiplier_c = multiplier->diff(c);

    auto rhs_multiplier_a = L.get_response_rhs(multiplier_a, true);
    std::cout << "-\\boldsymbol{\\zeta}^{a}_{\\omega} = "
              << Tinned::latexify(rhs_multiplier_a) << "\n\n";
}
