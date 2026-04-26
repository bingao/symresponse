# SymResponse

A unified framework for response theory at different levels of
electronic-structure theory.

The theoretical background can be found in:

\[1\] Bin Gao and Magnus Ringholm,
    *Unified Framework for Molecular Response Functions of Different Electronic-Structure Models*,
    *J. Phys. Chem. A* **2025**, *129*, 3709-3721.

Reference \[1\] describes the computation of response functions by using
- Atomic orbital density matrix-based response theory
- Coupled-cluster response theory

\[2\] Bin Gao, Magnus Ringholm amd Kenneth Ruud,
    *A unified framework for the application of response theory using different electronic-structure models*,
    *in manuscript* **2026**.

Reference \[2\] describes the computation of residues by using
- Atomic orbital density matrix-based response theory
- Coupled-cluster response theory
- Multi-configurational self-consistent field response theory

---

## License

SymResponse is licensed under the Mozilla Public License Version 2.0. See the
`LICENSE` file for more information.

---

## Installation

SymResponse depends on the symbolic computation crate
[Tinned](https://github.com/bingao/tinned), and is a reimplementation of \[1\]
using the Rust programming language.

### Build Rust library

```bash
cargo build -p symresponse_lib --release
```

or

```bash
cargo build-lib
```

---

## Documentation

The full API documentation can be generated with:

```bash
cargo doc -p symresponse_lib
```

To open it in a browser:

```bash
cargo doc -p symresponse_lib --open
```

---

## Examples

Examples of using the SymResponse crate can be found in the `tests` folder:

- `cc_response.rs` and `cc_residue.rs` demonstrate the computation of response
  functions and residues at the level of coupled-cluster response theory.
- `dao_response.rs` and `dao_residue.rs` demonstrate response function and
  residue computations within atomic orbital density matrix–based response
  theory.
- `mcscf_response.rs` and `mcscf_residue.rs` demonstrate response function and
  residue computations within multi-configurational self-consistent field
  (MCSCF) response theory.
- `orb_cc_response.rs` demonstrates response function computations within
  orbital-relaxed coupled-cluster response theory.

---

## C FFI

This project also provides a C-compatible interface in the `symresponse_ffi`
crate, allowing the library to be used from C and other languages with FFI
support.

See the `symresponse_ffi` README for usage details and examples.
