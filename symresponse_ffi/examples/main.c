#include <stdio.h>

typedef void (*example_fn)(void);

typedef struct {
    const char *name;
    example_fn fn;
} example_entry;

/* Prototypes from the list */
#define X(name) void name(void);
#include "all_examples.def"
#undef X

/* Registry from the same list */
static const example_entry EXAMPLES[] = {
#define X(name) { #name, name },
#include "all_examples.def"
#undef X
};

int main(void) {
    size_t n = sizeof(EXAMPLES) / sizeof(EXAMPLES[0]);

    printf("Running %zu C FFI examples:\n\n", n);

    for (size_t i = 0; i < n; ++i) {
        printf("=== Example %zu: %s ===\n", i + 1, EXAMPLES[i].name);
        EXAMPLES[i].fn();
        printf("\n");
    }

    puts("All examples have finished.");
    return 0;
}
