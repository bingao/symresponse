# SymResponse

A unified framework for response theory at different levels of
electronic-structure theory.

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

