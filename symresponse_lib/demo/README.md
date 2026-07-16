# SymResponse demo

Before running this demo, you may need to set up `JLRS_JULIA_DIR` and
`DYLD_LIBRARY_PATH`.

## For a normal official Julia installation

One may use
```bash
export JLRS_JULIA_DIR="$(
    cd "$(dirname "$(realpath "$(which julia)")")/.." &&
    pwd
)"

export DYLD_LIBRARY_PATH="$JLRS_JULIA_DIR/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Check it as
```bash
echo "$JLRS_JULIA_DIR"
test -f "$JLRS_JULIA_DIR/include/julia/julia_version.h" &&
    echo "Julia header found"
```

## If Julia was installed with `juliaup`

One can use
```bash
export JLRS_JULIA_DIR="$(
    julia -e 'print(normpath(joinpath(Sys.BINDIR, "..")))'
)"

export DYLD_LIBRARY_PATH="$JLRS_JULIA_DIR/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Then, confirm
```bash
echo "$JLRS_JULIA_DIR"
ls "$JLRS_JULIA_DIR/include/julia/julia_version.h"
```

## Run this demo

Inside the demo folder, simply type

```bash
cargo run
```

If one needs to use this demo outside the crate, the Git respository of
SymResponse may be used. One needs to change the line `symresponse = ...` in
`Cargo.toml` to

```
symresponse = { git = "https://github.com/bingao/symresponse", branch = "main", package = "symresponse_lib" }
```
