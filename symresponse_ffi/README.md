# SymResponse C FFI

C interface for the SymResponse library.

---

## License

SymResponse C FFI is licensed under the Mozilla Public License, Version 2.0.
See the `LICENSE` file for details.

---

## Installation

SymResponse C FFI depends on the SymResponse Rust library, as well as the
symbolic computation crate [Tinned](https://github.com/bingao/tinned) and its
C FFI.

### Build the C FFI library

The C library can be built using:

```bash
cargo build -p symresponse_ffi --profile release-ffi
```

or

```bash
cargo build-ffi
```

### Generate C headers (optional)

Whenever the source code in `symresponse_ffi/src` is updated, the C header
file should be regenerated using:

```bash
cargo test -p symresponse_ffi --features c-headers --profile release -- --exact header_gen::generate_c_header --nocapture
```

or

```bash
cargo gen-headers
```

---

## Usage

- The API is documented in the generated C header file: `include/symresponse.h`
- Example programs can be found in the `examples` directory

---

## Memory Management

(Describe ownership rules here, e.g., who allocates and frees objects, and which
functions must be used to release resources.)

---

## Safety Notes

(Describe thread-safety guarantees, undefined behavior risks, and any required
usage constraints for the C API.)
