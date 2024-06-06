# SymResponse

A unified framework for response theory at different levels of
electronic-structure theory. Theoretical background can be found in:

* Reference [[1]](#1) describes the implementation of atomic orbital (AO)
  density matrix-based and coupled-cluster response theories, and numerical
  evaluation of the AO density matrix-based response theory at Hartree-Fock
  level.

## License

SymResponse is licensed under Mozilla Public License Version 2.0, see the
[LICENSE](LICENSE) for more information.

## Installation

SymResponse depends on libraries SymEngine and Tinned. Both of them require
CMake and C++ compiler which supports C++11 standard.

Clone and build [forked SymEngine library](https://github.com/bingao/symengine)
first, which has implmented derivatives for different matrix expressions.

Then clone [Tinned library](https://github.com/bingao/tinned) and build it by
setting `SymEngine_DIR` to the SymEngine installation or build directory.

Last, build SymResponse by setting both `SymEngine_DIR` and `Tinned_DIR`, where
the latter is the Tinned installation or build directory.

## SymResponse APIs

## Examples

## References

<a id="1">[1]</a>
Bin Gao and Magnus Ringholm "Unified framework for simulating molecular
response functions of different electronic-structure models", in manuscript.
