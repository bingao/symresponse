#include <stdio.h>
#include <string.h>

#include "symresponse.h"
#include "symresponse_cleanup.h"

void cc_response(void) {
    TinnedErrorHandle_t* err = NULL;

    // Create perturbation a
    ExprHandle_t* freq_a = tinned_symbol_new("omega_a", &err);
    if (!freq_a) {
        fprintf(
            stderr,
            "Failed to create frequency a, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }
    PerturbationHandle_t* pert_a = tinned_perturbation_new("a", freq_a, &err);
    if (!pert_a) {
        fprintf(
            stderr,
            "Failed to create perturbation a, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }
    TINNED_SAFE_FREE_EXPR(freq_a);

    // Create perturbation b
    ExprHandle_t* freq_b = tinned_symbol_new("omega_b", &err);
    if (!freq_b) {
        fprintf(
            stderr,
            "Failed to create frequency b, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }
    PerturbationHandle_t* pert_b = tinned_perturbation_new("b", freq_b, &err);
    if (!pert_b) {
        fprintf(
            stderr,
            "Failed to create perturbation b, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }
    TINNED_SAFE_FREE_EXPR(freq_b);

    // Unperturbed Hamiltonian
    ExprHandle_t* H0 = tinned_one_elec_operator_new("H0", NULL, &err);
    if (!H0) {
        fprintf(
            stderr,
            "Failed to create unperturbed Hamiltonian, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    // Perturbation operators
    PertMultichainHandle_t* deps_Va = tinned_pert_multichain_new();
    bool ok = tinned_pert_multichain_insert(deps_Va, pert_a, &err);
    if (!ok) {
        fprintf(
            stderr,
            "Failed to create dependencies of Va, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }
    ExprHandle_t* Va = tinned_one_elec_operator_new("Va", deps_Va, &err);
    if (!Va) {
        fprintf(
            stderr,
            "Failed to create perturbation operator Va, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }
    TINNED_SAFE_FREE_PERT_MULTICHAIN(deps_Va);

    PertMultichainHandle_t* deps_Vb = tinned_pert_multichain_new();
    ok = tinned_pert_multichain_insert(deps_Vb, pert_b, &err);
    if (!ok) {
        fprintf(
            stderr,
            "Failed to create dependencies of Vb, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }
    ExprHandle_t* Vb = tinned_one_elec_operator_new("Vb", deps_Vb, &err);
    if (!Vb) {
        fprintf(
            stderr,
            "Failed to create perturbation operator Vb, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }
    TINNED_SAFE_FREE_PERT_MULTICHAIN(deps_Vb);

    // Amplitudes
    ExprHandle_t* cc_amplitude = tinned_wfn_parameter_new("t", &err);
    if (!cc_amplitude) {
        fprintf(
            stderr,
            "Failed to create amplitudes, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    // Excitation operators
    ExprHandle_t* excitation_operators = tinned_one_elec_operator_new("tau", NULL, &err);
    if (!excitation_operators) {
        fprintf(
            stderr,
            "Failed to create excitation operators, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    // Lagrangian multipliers
    ExprHandle_t* multipliers = tinned_lag_multiplier_new("lambda", &err);
    if (!multipliers) {
        fprintf(
            stderr,
            "Failed to create Lagrangian multipliers, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    // Create quasi-energy Lagrangian
    ExprHandle_t const * const perturbing_operators[3] = {Va, Vb};
    ExprSlice_t perturbation_oper_slice = {
        .ptr = perturbing_operators,
        .len = 2,
    };

    LagrangianHandle_t* L = symresponse_lagrangian_cc_new(
        H0, &perturbation_oper_slice, cc_amplitude, excitation_operators, multipliers, &err
    );
    if (!L) {
        fprintf(
            stderr,
            "Failed to create quasi-energy Lagrangian, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    TINNED_SAFE_FREE_EXPR(H0);
    TINNED_SAFE_FREE_EXPR(Va);
    TINNED_SAFE_FREE_EXPR(Vb);
    TINNED_SAFE_FREE_EXPR(cc_amplitude);
    TINNED_SAFE_FREE_EXPR(excitation_operators);
    TINNED_SAFE_FREE_EXPR(multipliers);

    // Response function <<A; B>>, no intensive perturbations
    PerturbationHandle_t const * const exten_perturbations[2] = {pert_a, pert_b};
    PerturbationSlice_t exten_slice = {
        .ptr = exten_perturbations,
        .len = 2,
    };
    // Here, `false` disables the validation of sum of perturbation frequencies
    ExprHandle_t* L_ab = symresponse_response_function (
        L, &exten_slice, NULL, 0, false, NULL, &err
    );
    if (!L_ab) {
        fprintf(
            stderr,
            "Failed to compute <<A; B>>, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    // Serialize <<A; B>>
    char* str_L_ab = tinned_expr_serialize_json(L_ab, &err);
    if (!str_L_ab) {
        fprintf(
            stderr,
            "Failed to serialize <<A; B>>, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }
    fprintf(stdout, "<<A; B>> = %s\n\n", str_L_ab);
    TINNED_SAFE_FREE_STR(str_L_ab);

    // Cleanup
    TINNED_SAFE_FREE_PERTURBATION(pert_a);
    TINNED_SAFE_FREE_PERTURBATION(pert_b);
    TINNED_SAFE_FREE_EXPR(L_ab);
    SYMRESPONSE_SAFE_FREE_LAG(L);
    if (err) TINNED_SAFE_FREE_ERR(err);
}