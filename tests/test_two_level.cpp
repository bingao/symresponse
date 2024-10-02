#define CATCH_CONFIG_MAIN
#include <iostream>

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include <symengine/dict.h>
#include <symengine/number.h>
#include <symengine/constants.h>
#include <symengine/rational.h>
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/functions.h>
#include <symengine/symbol.h>
#include <symengine/pow.h>
#include <symengine/matrices/diagonal_matrix.h>
#include <symengine/matrices/immutable_dense_matrix.h>
#include <symengine/symengine_rcp.h>
#include <symengine/subs.h>

#include <Tinned.hpp>
#include <Tinned/TwoLevelAtom.hpp>

#include "SymResponse.hpp"

//using namespace Tinned;
using namespace SymResponse;

// This test adopts a two-state model system, from section 5.10.3, "Principles
// and Practices of Molecular Properties: Theory, Modeling and Simulations",
// Patrick Norman, Kenneth Ruud and Trond Saue.

// Density matrix for a pure state
// [[rho, exp(-i*phi)*sqrt(rho*(1-rho))], [exp(i*phi)*sqrt(rho*(1-rho)), 1-rho]]
// See ...
inline SymEngine::RCP<const SymEngine::MatrixExpr> pure_state_density(
    const SymEngine::RCP<const SymEngine::Basic>& rho,
    const SymEngine::RCP<const SymEngine::Basic>& phi
)
{
    auto rho_22 = SymEngine::sub(SymEngine::one, rho);
    if (SymEngine::eq(*rho, *SymEngine::zero) ||
        SymEngine::eq(*rho_22, *SymEngine::zero)) {
        return SymEngine::diagonal_matrix({rho, rho_22});
    }
    else {
        auto rho_12 = SymEngine::sqrt(SymEngine::mul(rho, rho_22));
        if (SymEngine::eq(*phi, *SymEngine::zero)) {
            return SymEngine::immutable_dense_matrix(
                2, 2, {rho, rho_12, rho_12, rho_22}
            );
        }
        else {
            return SymEngine::immutable_dense_matrix(
                2, 2,
                {
                    rho,
                    SymEngine::mul(
                        SymEngine::sub(
                            SymEngine::cos(phi),
                            SymEngine::mul(SymEngine::I, SymEngine::sin(phi))
                        ),
                        rho_12
                    ),
                    SymEngine::mul(
                        SymEngine::add(
                            SymEngine::cos(phi),
                            SymEngine::mul(SymEngine::I, SymEngine::sin(phi))
                        ),
                        rho_12
                    ),
                    rho_22
                }
            );
        }
    }
}

// Unperturbed Hamiltonian [[E0, 0], [0, E1]]
inline SymEngine::RCP<const SymEngine::MatrixExpr> make_unperturbed_hamiltonian(
    const SymEngine::RCP<const SymEngine::Basic>& E0,
    const SymEngine::RCP<const SymEngine::Basic>& E1
)
{
    return SymEngine::diagonal_matrix({E0, E1});
}

// Field operator
inline SymEngine::RCP<const SymEngine::MatrixExpr> make_field_operator(
    const SymEngine::RCP<const SymEngine::Basic>& V_00,
    const SymEngine::RCP<const SymEngine::Basic>& V_01,
    const SymEngine::RCP<const SymEngine::Basic>& V_10,
    const SymEngine::RCP<const SymEngine::Basic>& V_11
)
{
    return SymEngine::immutable_dense_matrix(
        2, 2, {V_00, V_01, V_10, V_11}
    );
}

// Pair of `SymEngine::RCP<const SymEngine::Basic>`
typedef std::pair<SymEngine::RCP<const SymEngine::Basic>,
                  SymEngine::RCP<const SymEngine::Basic>> pair_basic;

// Factors of an expression
typedef std::vector<std::pair<SymEngine::RCP<const SymEngine::Basic>,
                              signed long int>> factor_basic;

// Convert a given expression into a vector of factors
inline factor_basic convert_to_factors(const SymEngine::RCP<const SymEngine::Basic>& expr)
{
    if (SymEngine::is_a_sub<const SymEngine::Mul>(*expr)) {
        factor_basic factors;
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Mul>(expr)->get_args();
        for (const auto& arg: args) {
            auto result = convert_to_factors(arg);
            factors.insert(factors.end(), result.begin(), result.end());
        }
        return factors;
    }
    else if (SymEngine::is_a_sub<const SymEngine::Pow>(*expr)) {
        auto p = SymEngine::rcp_dynamic_cast<const SymEngine::Pow>(expr);
        REQUIRE(SymEngine::is_a_sub<const SymEngine::Integer>(*(p->get_exp())));
        auto exp = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(p->get_exp());
        return factor_basic({std::make_pair(p->get_base(), exp->as_int())});
    }
    else {
        return factor_basic({std::make_pair(expr, 1)});
    }
}

// Conver a given vector of factors into an expression
inline SymEngine::RCP<const SymEngine::Basic> convert_from_factors(
    const factor_basic& factors
)
{
    SymEngine::vec_basic args;
    for (const auto& f: factors) {
        if (f.second==1) {
            args.push_back(f.first);
        }
        else {
            args.push_back(SymEngine::pow(f.first, SymEngine::integer(f.second)));
        }
    }
    return SymEngine::mul(args);
}

// Addition of two fractions, where the numerator is `first` and the
// denominator `second`
pair_basic add_fractions(const pair_basic& frac1, const pair_basic& frac2)
{
    auto factors1 = convert_to_factors(frac1.second);
    auto factors2 = convert_to_factors(frac2.second);
    // Try to find the least common multiple of these two denominators
    int sign_numerator2 = 1;
    factor_basic lcm_factors;
    for (auto iter1=factors1.begin(); iter1!=factors1.end();) {
        auto not_found = true;
        for (auto iter2=factors2.begin(); iter2!=factors2.end(); ++iter2) {
            // Check if the factors of the first and the second fractions have
            // the same base or with opposite sign
            auto eq_base = SymEngine::eq(*(iter1->first), *(iter2->first));
            auto eq_minus_base = eq_base ? false : SymEngine::eq(
                *(SymEngine::expand(iter1->first)),
                *(SymEngine::expand(SymEngine::mul(SymEngine::minus_one, iter2->first)))
            );
            if (eq_minus_base) if (iter2->second%2!=0) sign_numerator2 *= -1;
            if (eq_base || eq_minus_base) {
                // Both the first and second fractions have the factor and save
                // it for LCM
                if (iter1->second==iter2->second) {
                    lcm_factors.push_back(*iter1);
                    iter1 = factors1.erase(iter1);
                    factors2.erase(iter2);
                }
                else if (iter1->second>iter2->second) {
                    // Save the factor (a power) from the first fraction for
                    // LCM, and remove that (with less exponent) from
                    // `factors2`
                    lcm_factors.push_back(*iter1);
                    // `factors1` will contain factors that the second fraction
                    // will multiply by
                    iter1->second -= iter2->second;
                    ++iter1;
                    factors2.erase(iter2);
                }
                else {
                    if (eq_minus_base) iter2->first = iter1->first;
                    // Save the factor (a power) from the second fraction for
                    // LCM
                    lcm_factors.push_back(*iter2);
                    // `factors2` will contain factors that the first fraction
                    // will multiply by
                    iter2->second -= iter1->second;
                    iter1 = factors1.erase(iter1);
                }
                not_found = false;
                break;
            }
        }
        // The factor of the first fraction does not exist in the second
        // fraction, so we save it for LCM
        if (not_found) {
            lcm_factors.push_back(*iter1);
            ++iter1;
        }
    }
    // Save left factors in the second fraction for LCM
    lcm_factors.insert(lcm_factors.end(), factors2.begin(), factors2.end());
    // `frac1.first`*`SymEngine::mul(factor2)` +/- `frac2.first`*`SymEngine::mul(factor1)`
    if (sign_numerator2==-1) factors1.push_back(std::make_pair(SymEngine::minus_one, 1));
    auto numerator_factors = convert_to_factors(SymEngine::expand(SymEngine::add(
        SymEngine::mul(frac1.first, convert_from_factors(factors2)),
        SymEngine::mul(frac2.first, convert_from_factors(factors1))
    )));
    // Remove common factors from the numerator and denominator (LCM)
    int sign_numerator = 1;
    for (auto iter1=numerator_factors.begin(); iter1!=numerator_factors.end();) {
        auto not_found = true;
        for (auto iter2=lcm_factors.begin(); iter2!=lcm_factors.end(); ++iter2) {
            auto eq_base = SymEngine::eq(*(iter1->first), *(iter2->first));
            auto eq_minus_base = eq_base ? false : SymEngine::eq(
                *(SymEngine::expand(iter1->first)),
                *(SymEngine::expand(SymEngine::mul(SymEngine::minus_one, iter2->first)))
            );
            if (eq_minus_base) if (iter1->second%2!=0) sign_numerator *= -1;
            if (eq_base || eq_minus_base) {
                if (iter1->second==iter2->second) {
                    iter1 = numerator_factors.erase(iter1);
                    lcm_factors.erase(iter2);
                }
                else if (iter1->second>iter2->second) {
                    if (eq_minus_base) iter1->first = iter2->first;
                    iter1->second -= iter2->second;
                    ++iter1;
                    lcm_factors.erase(iter2);
                }
                else {
                    iter2->second -= iter1->second;
                    iter1 = numerator_factors.erase(iter1);
                }
                not_found = false;
                break;
            }
        }
        if (not_found) ++iter1;
    }
    if (sign_numerator==-1)
        numerator_factors.push_back(std::make_pair(SymEngine::minus_one, 1));
    return std::make_pair(
        convert_from_factors(numerator_factors), convert_from_factors(lcm_factors)
    );
}

// Convert response functions into a map with key the product of matrix
// elements of field operators, and value the fraction of frequency differences
// where the numerator and denominator are stored separately into a pair
std::map<SymEngine::RCP<const SymEngine::Basic>, pair_basic, SymEngine::RCPBasicKeyLess>
rsp_functions_to_map(const SymEngine::RCP<const SymEngine::Basic>& expr)
{
    auto sum = SymEngine::expand(expr);
    REQUIRE(SymEngine::is_a_sub<const SymEngine::Add>(*sum));
    std::map<SymEngine::RCP<const SymEngine::Basic>,
             pair_basic, SymEngine::RCPBasicKeyLess> result;
    auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Add>(sum)->get_args();
    for (const auto& arg: args) {
        REQUIRE(SymEngine::is_a_sub<const SymEngine::Mul>(*arg));
        // Each term of response functions is expressed as a product of
        // V...V/(omega...omega)/.../(omega...omega)
        auto product = SymEngine::rcp_dynamic_cast<const SymEngine::Mul>(arg);
        SymEngine::vec_basic field_oper_args;
        SymEngine::vec_basic numerator_args;
        SymEngine::vec_basic denominator_args;
        for (const auto& f: product->get_args()) {
            if (SymEngine::is_a_sub<const SymEngine::Pow>(*f)) {
                auto p = SymEngine::rcp_dynamic_cast<const SymEngine::Pow>(f);
                REQUIRE(SymEngine::is_a_Number(*p->get_exp()));
                auto exp = SymEngine::rcp_dynamic_cast<const SymEngine::Number>(
                    p->get_exp()
                );
                if (exp->is_negative()) {
                    if (exp->is_minus_one()) {
                        denominator_args.push_back(p->get_base());
                    }
                    else {
                        denominator_args.push_back(SymEngine::pow(
                            p->get_base(), SymEngine::mul(SymEngine::minus_one, exp)
                        ));
                    }
                }
                else {
                    field_oper_args.push_back(f);
                }
            }
            else if (SymEngine::is_a_Number(*f)) {
                numerator_args.push_back(f);
            }
            else {
                field_oper_args.push_back(f);
            }
        }
        auto field_oper = SymEngine::expand(SymEngine::mul(field_oper_args));
        SymEngine::RCP<const SymEngine::Basic> numerator;
        if (numerator_args.empty()) {
            numerator = SymEngine::one;
        }
        else {
            numerator = SymEngine::expand(SymEngine::mul(numerator_args));
        }
        auto denominator = SymEngine::mul(denominator_args);
        auto iter = result.find(field_oper);
        if (iter==result.end()) {
            // `field_oper` may already exist with an opposite sign
            iter = result.find(
                SymEngine::expand(SymEngine::mul(SymEngine::minus_one, field_oper))
            );
            if (iter==result.end()) {
                result[field_oper] = std::make_pair(numerator, denominator);
            }
            else {
                iter->second = add_fractions(
                    iter->second,
                    std::make_pair(
                        numerator, SymEngine::mul(SymEngine::minus_one, denominator)
                    )
                );
            }
        }
        else {
            iter->second = add_fractions(
                iter->second, std::make_pair(numerator, denominator)
            );
        }
    }
    return result;
}

// Compare the equality of two response functions
inline void compare_rsp_functions(
    const SymEngine::RCP<const SymEngine::Basic>& expr1,
    const SymEngine::RCP<const SymEngine::Basic>& expr2
)
{
    auto terms1 = rsp_functions_to_map(expr1);
    auto terms2 = rsp_functions_to_map(expr2);
    REQUIRE(terms1.size()==terms2.size());
    for (const auto& term1: terms1) {
        auto numerator1 = SymEngine::expand(term1.second.first);
        auto term2 = terms2[term1.first];
        auto numerator2 = SymEngine::expand(term2.first);
std::cout << "key = " << Tinned::stringify(term1.first) << "\n";
std::cout << "numerator1 = " << Tinned::stringify(term1.second.first, false) << "\n";
std::cout << "numerator2 = " << Tinned::stringify(term2.first, false) << "\n";
std::cout << "denominator1 = " << Tinned::stringify(term1.second.second, false) << "\n";
std::cout << "denominator2 = " << Tinned::stringify(term2.second, false) << "\n";
auto diff = add_fractions(
    term1.second, std::make_pair(SymEngine::mul(SymEngine::minus_one, term2.first), term2.second)
);
std::cout << "diff = " << Tinned::stringify(diff.first) << "/" << Tinned::stringify(diff.second) << "\n\n";
        if (SymEngine::eq(*numerator1, *numerator2)) {
            auto denominator1 = SymEngine::expand(term1.second.second);
            auto denominator2 = SymEngine::expand(term2.second);
            REQUIRE(SymEngine::eq(*denominator1, *denominator2));
        }
        else {
            auto minus_numerator2 = SymEngine::expand(
                SymEngine::mul(SymEngine::minus_one, term2.first)
            );
            // Check if the second numerator has opposite sign
            REQUIRE(SymEngine::eq(*numerator1, *minus_numerator2));
            auto denominator1 = SymEngine::expand(term1.second.second);
            auto minus_denominator2 = SymEngine::expand(
                SymEngine::mul(SymEngine::minus_one, term2.second)
            );
            REQUIRE(SymEngine::eq(*denominator1, *minus_denominator2));
        }
    }
}

TEST_CASE("Test two-level atom", "[LagrangianDAO, TwoLevelFunction, TwoLevelOperator]")
{
    // Set perturbations
    auto omega_a = SymEngine::symbol("omega_a");
    auto omega_b = SymEngine::symbol("omega_b");
    auto omega_c = SymEngine::symbol("omega_c");
    auto omega_d = SymEngine::symbol("omega_d");
    auto omega_e = SymEngine::symbol("omega_e");
    auto omega_f = SymEngine::symbol("omega_f");
    auto a = Tinned::make_perturbation(std::string("a"), omega_a);
    auto b = Tinned::make_perturbation(std::string("b"), omega_b);
    auto c = Tinned::make_perturbation(std::string("c"), omega_c);
    auto d = Tinned::make_perturbation(std::string("d"), omega_d);
    auto e = Tinned::make_perturbation(std::string("e"), omega_e);
    auto f = Tinned::make_perturbation(std::string("f"), omega_f);

    // Different operators
    auto D = Tinned::make_1el_density(std::string("\\rho"));
    auto H0 = Tinned::make_1el_operator(std::string("H_0"));
    auto Va = Tinned::make_1el_operator(
        std::string("V_\\alpha"), Tinned::PertDependency({std::make_pair(a, 1)})
    );
    auto Vb = make_1el_operator(
        std::string("V_\\beta"), Tinned::PertDependency({std::make_pair(b, 1)})
    );
    auto Vc = make_1el_operator(
        std::string("V_\\gamma"), Tinned::PertDependency({std::make_pair(c, 1)})
    );
    auto Vd = make_1el_operator(
        std::string("V_\\delta"), Tinned::PertDependency({std::make_pair(d, 1)})
    );
    auto Ve = make_1el_operator(
        std::string("V_\\varepsilon"), Tinned::PertDependency({std::make_pair(e, 1)})
    );
    auto Vf = make_1el_operator(
        std::string("V_\\zeta"), Tinned::PertDependency({std::make_pair(f, 1)})
    );

    // Create quasi-energy derivative Lagrangian
    auto lagrangian = LagrangianDAO(
        a, D, SymEngine::vec_basic({H0, Va, Vb, Vc, Vd, Ve, Vf})
    );

    // Evaluator for response functions
    auto E0 = SymEngine::symbol("E_0");
    auto E1 = SymEngine::symbol("E_1");
    //auto E0 = SymEngine::rational(-1, 5);
    //auto E1 = SymEngine::rational(1, 10);
    auto val_H0 = make_unperturbed_hamiltonian(E0, E1);
    auto Va_00 = SymEngine::symbol("V_{\\alpha,00}");
    auto Va_01 = SymEngine::symbol("V_{\\alpha,01}");
    auto Va_10 = SymEngine::symbol("V_{\\alpha,10}");
    auto Va_11 = SymEngine::symbol("V_{\\alpha,11}");
    auto val_Ba = make_field_operator(Va_00, Va_01, Va_10, Va_11);
    auto Vb_00 = SymEngine::symbol("V_{\\beta,00}");
    auto Vb_01 = SymEngine::symbol("V_{\\beta,01}");
    auto Vb_10 = SymEngine::symbol("V_{\\beta,10}");
    auto Vb_11 = SymEngine::symbol("V_{\\beta,11}");
    auto val_Bb = make_field_operator(Vb_00, Vb_01, Vb_10, Vb_11);
    auto Vc_00 = SymEngine::symbol("V_{\\gamma,00}");
    auto Vc_01 = SymEngine::symbol("V_{\\gamma,01}");
    auto Vc_10 = SymEngine::symbol("V_{\\gamma,10}");
    auto Vc_11 = SymEngine::symbol("V_{\\gamma,11}");
    auto val_Bc = make_field_operator(Vc_00, Vc_01, Vc_10, Vc_11);
    auto Vd_00 = SymEngine::symbol("V_{\\delta,00}");
    auto Vd_01 = SymEngine::symbol("V_{\\delta,01}");
    auto Vd_10 = SymEngine::symbol("V_{\\delta,10}");
    auto Vd_11 = SymEngine::symbol("V_{\\delta,11}");
    auto val_Bd = make_field_operator(Vd_00, Vd_01, Vd_10, Vd_11);
    auto Ve_00 = SymEngine::symbol("V_{\\varepsilon,00}");
    auto Ve_01 = SymEngine::symbol("V_{\\varepsilon,01}");
    auto Ve_10 = SymEngine::symbol("V_{\\varepsilon,10}");
    auto Ve_11 = SymEngine::symbol("V_{\\varepsilon,11}");
    auto val_Be = make_field_operator(Ve_00, Ve_01, Ve_10, Ve_11);
    auto Vf_00 = SymEngine::symbol("V_{\\zeta,00}");
    auto Vf_01 = SymEngine::symbol("V_{\\zeta,01}");
    auto Vf_10 = SymEngine::symbol("V_{\\zeta,10}");
    auto Vf_11 = SymEngine::symbol("V_{\\zeta,11}");
    auto val_Bf = make_field_operator(Vf_00, Vf_01, Vf_10, Vf_11);
    // We use the ground state [[1, 0], [0, 0]]
    auto val_D = pure_state_density(SymEngine::one, SymEngine::zero);

    // Create the evaluator for response functions
    auto fun_eval = Tinned::TwoLevelFunction(
        std::make_pair(H0, val_H0),
        std::map<SymEngine::RCP<const Tinned::OneElecOperator>,
                 SymEngine::RCP<const SymEngine::MatrixExpr>,
                 SymEngine::RCPBasicKeyLess>({
            {Va, val_Ba}, {Vb, val_Bb}, {Vc, val_Bc}, {Vd, val_Bd}, {Ve, val_Be}, {Vf, val_Bf}
        }),
        std::make_pair(D, val_D)
    );

    auto L_ab_0_1 = lagrangian.get_response_functions(
        Tinned::PertTuple({b}), {}, 2
    );
    auto val_L_ab_0_1 = fun_eval.apply(L_ab_0_1);
    auto val_ref = SymEngine::add(
        SymEngine::div(
            SymEngine::mul({SymEngine::minus_one, Vb_01, Va_10}),
            SymEngine::sub(omega_b, SymEngine::sub(E0, E1))
        ),
        SymEngine::div(
            SymEngine::mul(Vb_10, Va_01),
            SymEngine::sub(omega_b, SymEngine::sub(E1, E0))
        )
    );
    REQUIRE(SymEngine::eq(*val_L_ab_0_1, *val_ref));
    //std::cout << "L_{ab,0,1} = " << Tinned::latexify(L_ab_0_1) << "\n";
    //std::cout << "L_{ab,0,1} = " << Tinned::latexify(val_L_ab_0_1) << "\n\n";

    // Response function in Equation (237), J. Chem. Phys. 129, 214108 (2008)
    auto L_abc_0_2 = lagrangian.get_response_functions(
        Tinned::PertTuple({b, c}), {}, 3
    );
    auto val_L_abc_0_2 = fun_eval.apply(L_abc_0_2);
    //std::cout << "L_{abc,0,2} = " << Tinned::latexify(L_abc_0_2) << "\n";
    //std::cout << "L_{abc,0,2} = " << Tinned::latexify(val_L_abc_0_2) << "\n\n";

    // Response function in Equation (235), J. Chem. Phys. 129, 214108 (2008)
    auto L_abc_1_1 = lagrangian.get_response_functions(
        Tinned::PertTuple({b, c}), {}, 2
    );
    auto val_L_abc_1_1 = fun_eval.apply(L_abc_1_1);
    //std::cout << "L_{abc,1,1} = " << Tinned::latexify(L_abc_1_1) << "\n";
    //std::cout << "L_{abc,1,1} = " << Tinned::latexify(val_L_abc_1_1) << "\n\n";

    // Check the equality of `val_L_abc_0_2` and `val_L_abc_1_1`
    compare_rsp_functions(
        val_L_abc_0_2,
        SymEngine::subs(
            val_L_abc_1_1,
            {
                {
                    omega_a,
                    SymEngine::mul(SymEngine::minus_one, SymEngine::add(omega_b, omega_c))
                }
            }
        )
    );

    auto L_abcd_0_3 = lagrangian.get_response_functions(
        Tinned::PertTuple({b, c, d}), {}, 4
    );
    auto val_L_abcd_0_3 = fun_eval.apply(L_abcd_0_3);

    auto L_abcd_1_2 = lagrangian.get_response_functions(
        Tinned::PertTuple({b, c, d}), {}, 3
    );
    auto val_L_abcd_1_2 = fun_eval.apply(L_abcd_1_2);
    compare_rsp_functions(
        val_L_abcd_0_3,
        SymEngine::subs(
            val_L_abcd_1_2,
            {
                {
                    omega_a,
                    SymEngine::mul(
                        SymEngine::minus_one, SymEngine::add({omega_b, omega_c, omega_d})
                    )
                }
            }
        )
    );

    auto L_abcd_2_1 = lagrangian.get_response_functions(
        Tinned::PertTuple({b, c, d}), {}, 2
    );
    auto val_L_abcd_2_1 = fun_eval.apply(L_abcd_2_1);

//    auto L_abcde_0_4 = lagrangian.get_response_functions(
//        Tinned::PertTuple({b, c, d, e}), {}, 5
//    );
//    auto val_L_abcde_0_4 = SymEngine::simplify(fun_eval.apply(L_abcde_0_4));
//
//    auto L_abcdef_0_5 = lagrangian.get_response_functions(
//        Tinned::PertTuple({b, c, d, e, f}), {}, 6
//    );
//    auto val_L_abcdef_0_5 = SymEngine::simplify(fun_eval.apply(L_abcdef_0_5));
//
//    auto L_abcdef_1_4 = lagrangian.get_response_functions(
//        Tinned::PertTuple({b, c, d, e, f}), {}, 5
//    );
//    auto val_L_abcdef_1_4 = SymEngine::simplify(fun_eval.apply(L_abcdef_1_4));
//
//    auto L_abcdef_2_3 = lagrangian.get_response_functions(
//        Tinned::PertTuple({b, c, d, e, f}), {}, 4
//    );
//    auto val_L_abcdef_2_3 = SymEngine::simplify(fun_eval.apply(L_abcdef_2_3));
//
//    auto L_abcdef_3_2 = lagrangian.get_response_functions(
//        Tinned::PertTuple({b, c, d, e, f}), {}, 3
//    );
//    auto val_L_abcdef_3_2 = SymEngine::simplify(fun_eval.apply(L_abcdef_3_2));
}
