#include <stdio.h>
#include <string.h>

#include "symresponse.h"
#include "symresponse_cleanup.h"

void dao_response(void) {
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

    // Create dependencies with respect to perturbations
    PerturbationEntry_t pert_entries[2];
    pert_entries[0] = tinned_perturbation_entry_new(pert_a, 99);
    pert_entries[1] = tinned_perturbation_entry_new(pert_b, 99);
    PerturbationEntrySlice_t pert_slice = {
        .ptr = pert_entries,
        .len = 2,
    };

    PertMultichainHandle_t* dependencies = tinned_pert_multichain_from_entries(pert_slice, &err);
    if (!dependencies) {
        fprintf(
            stderr,
            "Failed to create dependencies, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    // Set different operators
    ExprHandle_t* D = tinned_wfn_parameter_new("D", false, &err);
    if (!D) {
        fprintf(
            stderr,
            "Failed to create density matrix, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    PerturbationHandle_t const * const independent_perturbations[2] = {pert_a, pert_b};
    PerturbationSlice_t independent_slice = {
        .ptr = independent_perturbations,
        .len = 2,
    };

    ExprHandle_t* S = tinned_one_elec_matrix_new(
        "S",
        false,
        dependencies,
        &independent_slice,
        &err
    );
    if (!S) {
        fprintf(
            stderr,
            "Failed to create overlap matrix, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    ExprHandle_t* h = tinned_one_elec_matrix_new(
        "h",
        false,
        dependencies,
        &independent_slice,
        &err
    );
    if (!h) {
        fprintf(
            stderr,
            "Failed to create one-electron Hamiltonian, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    ExprHandle_t* V = tinned_one_elec_matrix_new(
        "V",
        true,
        dependencies,
        &independent_slice,
        &err
    );
    if (!V) {
        fprintf(
            stderr,
            "Failed to create external perturbation operator, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    ExprHandle_t* T = tinned_basis_time_evolution_new(dependencies, &err);
    if (!T) {
        fprintf(
            stderr,
            "Failed to create T matrix, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    ExprHandle_t* G = tinned_ao_two_elec_matrix_new("G", D, dependencies, NULL, &err);
    if (!G) {
        fprintf(
            stderr,
            "Failed to create two-electron operator, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    // Make grid weight
    ExprHandle_t* weight = tinned_non_elec_function_new(
        "weight",
        false,
        dependencies,
        &independent_slice,
        &err
    );
    if (!weight) {
        fprintf(
            stderr,
            "Failed to create grid weight, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    // Make generalized overlap distribution
    ExprHandle_t* Omega = tinned_one_elec_matrix_new(
        "Omega",
        false,
        dependencies,
        &independent_slice,
        &err
    );
    if (!Omega) {
        fprintf(
            stderr,
            "Failed to create generalized overlap distribution, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    // Make exchange-correlation energy functional and potential operator
    ExprHandle_t* Exc = tinned_exch_corr_energy_new("Exc", weight, D, Omega, &err);
    if (!Exc) {
        fprintf(
            stderr,
            "Failed to create exchange-correlation energy functional, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    ExprHandle_t* Vxc = tinned_exch_corr_potential_new("Vxc", weight, D, Omega, &err);
    if (!Vxc) {
        fprintf(
            stderr,
            "Failed to create exchange-correlation potential operator, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    TINNED_SAFE_FREE_EXPR(weight);
    TINNED_SAFE_FREE_EXPR(Omega);

    ExprHandle_t* hnuc = tinned_non_elec_function_new(
        "hnuc",
        false,
        dependencies,
        &independent_slice,
        &err
    );
    if (!hnuc) {
        fprintf(
            stderr,
            "Failed to create nuclear contributions, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    TINNED_SAFE_FREE_PERT_MULTICHAIN(dependencies);

    // Create quasi-energy derivative Lagrangian
    ExprHandle_t const * const one_elec_operators[3] = {h, V, T};
    ExprSlice_t one_elec_slice = {
        .ptr = one_elec_operators,
        .len = 3,
    };

    SymmetrizeModeReprC_t mode = SYMMETRIZE_MODE_REPR_C_AUTO;

    LagrangianHandle_t* La = symresponse_lagrangian_dao_new(
        pert_a, D, S, &one_elec_slice, G, Exc, Vxc, hnuc, &mode, NULL, &err
    );
    if (!La) {
        fprintf(
            stderr,
            "Failed to create quasi-energy derivative Lagrangian, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    TINNED_SAFE_FREE_EXPR(S);
    TINNED_SAFE_FREE_EXPR(h);
    TINNED_SAFE_FREE_EXPR(V);
    TINNED_SAFE_FREE_EXPR(T);
    TINNED_SAFE_FREE_EXPR(G);
    TINNED_SAFE_FREE_EXPR(Exc);
    TINNED_SAFE_FREE_EXPR(Vxc);
    TINNED_SAFE_FREE_EXPR(hnuc);

    // Get and serialize the quasi-energy derivative Lagrangian
    ExprHandle_t* expr_La = symresponse_get_lagrangian (La, &err);
    if (!expr_La) {
        fprintf(
            stderr,
            "Failed to get the quasi-energy derivative Lagrangian, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    char* str_La = tinned_expr_serialize_json(expr_La, &err);
    if (!str_La) {
        fprintf(
            stderr,
            "Failed to serialize the quasi-energy derivative Lagrangian, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }
    fprintf(stdout, "L^a = %s\n\n", str_La);
    TINNED_SAFE_FREE_STR(str_La);
    TINNED_SAFE_FREE_EXPR(expr_La);

    // Compute <<A; B>>
    PerturbationHandle_t const * const exten_perturbations[1] = {pert_b};
    PerturbationSlice_t exten_slice = {
        .ptr = exten_perturbations,
        .len = 1,
    };
    // Here, `false` disables the validation of sum of perturbation frequencies
    ExprHandle_t* La_b = symresponse_get_response_function (
        La, &exten_slice, NULL, 0, false, NULL, &err
    );
    if (!La_b) {
        fprintf(
            stderr,
            "Failed to compute <<A; B>>, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }

    // Serialize <<A; B>>
    char* str_La_b = tinned_expr_serialize_json(La_b, &err);
    if (!str_La_b) {
        fprintf(
            stderr,
            "Failed to serialize <<A; B>>, with error message: %s\n",
            tinned_error_display(err)
        );
        return;
    }
    fprintf(stdout, "<<A; B>> = %s\n\n", str_La_b);
    TINNED_SAFE_FREE_STR(str_La_b);

    // Cleanup
    TINNED_SAFE_FREE_PERTURBATION(pert_a);
    TINNED_SAFE_FREE_PERTURBATION(pert_b);
    TINNED_SAFE_FREE_EXPR(D);
    TINNED_SAFE_FREE_EXPR(La_b);
    SYMRESPONSE_SAFE_FREE_LAG(La);
    if (err) TINNED_SAFE_FREE_ERR(err);
}