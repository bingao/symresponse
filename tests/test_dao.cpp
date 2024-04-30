#define CATCH_CONFIG_MAIN

#include <iostream>

#include <string>
#include <utility>

#include <catch2/catch.hpp>

#include <symengine/dict.h>
#include <symengine/matrices/matrix_add.h>

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
        //a, D, S, SymEngine::matrix_add(SymEngine::vec_basic({h, V, T})), G, Exc, Fxc, hnuc
        a, D, S, h, G
    );

    auto result = lagrangian.get_response_functions(Tinned::PerturbationTuple({b, c}));
std::cout << "result = " << Tinned::stringify(result) << "\n";
}
