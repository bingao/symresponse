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
#include <symengine/matrices/matrix_add.h>
#include <symengine/matrices/matrix_mul.h>

#include <Tinned.hpp>

#include "SymResponse.hpp"

//using namespace Tinned;
using namespace SymResponse;

TEST_CASE("Test ...", "[LagrangianCC]")
{
    auto a = Tinned::make_perturbation(std::string("a"));
    auto b = Tinned::make_perturbation(std::string("b"));
    auto c = Tinned::make_perturbation(std::string("c"));
    auto dependencies = Tinned::PertDependency({
        std::make_pair(a, 99), std::make_pair(b, 99), std::make_pair(c, 99)
    });
    auto H = Tinned::make_1el_operator(std::string("H"), dependencies);
    auto amplitudes = Tinned::make_perturbed_parameter(std::string("amplitudes"));
    auto excit_operators = Tinned::make_1el_operator(std::string("excit-operators"));
    auto multipliers = Tinned::make_perturbed_parameter(std::string("multipliers"));

    // Create quasi-energy Lagrangian
    auto lagrangian = LagrangianCC(H, amplitudes, excit_operators, multipliers);

    // Response function, n+1 rule, no intensive perturbations
    auto L_abc_4 = lagrangian.get_response_functions(
        Tinned::PerturbationTuple({a, b, c}), {}, 4
    );
std::cout << "\n\nL_abc_4 = " << Tinned::stringify(L_abc_4) << "\n";

    auto all_amplitudes = Tinned::find_all(L_abc_4, amplitudes);
for (const auto& t: all_amplitudes) std::cout << "t = " << Tinned::stringify(t) << "\n";
std::cout << "\n";

    // Response function, ?? rule, no intensive perturbations
    auto L_abc_3 = lagrangian.get_response_functions(
        Tinned::PerturbationTuple({a, b, c}), {}, 3
    );
std::cout << "\n\nL_abc_3 = " << Tinned::stringify(L_abc_3) << "\n";

    all_amplitudes = Tinned::find_all(L_abc_3, amplitudes);
for (const auto& t: all_amplitudes) std::cout << "t = " << Tinned::stringify(t) << "\n";
std::cout << "\n";
}

