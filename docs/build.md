# Building

Use a build directory outside the source tree. Q-LEAF uses C++17, OpenMP, and `-O3 -march=native -Wall -Wextra`. Release search, entry generation, and real-tag construction use `-DNDEBUG`; the synthetic builder and loader/utils use GNU C++17 with `-UNDEBUG` and debug information. NSG and NN-Descent use C++11, `-O3 -march=native -Wall -DINFO`; NN-Descent assertions stay enabled. The synthetic builder uses `--as-needed` and section collection. CMake emits a warning for compilers other than GCC 15.2.0.

## External FAISS

Use FAISS commit `2135e5a28d0f2a932dd02b64b566ac1a52266540` (CMake project version 1.12.0) from [FAISS](https://github.com/facebookresearch/faiss). Build its CPU library:

```bash
git -C "$FAISS_SRC" checkout 2135e5a28d0f2a932dd02b64b566ac1a52266540
cmake -S "$FAISS_SRC" -B "$FAISS_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_EXTRAS=OFF -DFAISS_ENABLE_C_API=OFF \
  -DFAISS_ENABLE_MKL=OFF -DFAISS_OPT_LEVEL=avx2 -DFAISS_USE_LTO=OFF \
  -DBLA_VENDOR=OpenBLAS -DBLAS_LIBRARIES="$OPENBLAS_LIBRARY" \
  -DLAPACK_LIBRARIES="$OPENBLAS_LIBRARY" -DBUILD_TESTING=OFF
cmake --build "$FAISS_BUILD" --target faiss --parallel 2
```

The graph and entry-point tools link the generic `libfaiss.a`, not `libfaiss_avx2.a`, and OpenBLAS 0.3.25.

Set `FAISS_ROOT` to the FAISS source/build tree or installation prefix. For separate source and build directories, pass `-DFAISS_INCLUDE_DIR="$FAISS_SRC" -DFAISS_LIBRARY="$FAISS_BUILD/faiss/libfaiss.a"`.

In the experiments of the paper, FAISS and the NN-Descent library were compiled with GCC 9.4.0, and the other components with GCC 15.2.0.

## Q-LEAF

```bash
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DFAISS_ROOT="$FAISS_ROOT" -DQLEAF_OPENBLAS="$OPENBLAS_LIBRARY" \
  -DQLEAF_GFORTRAN="$GFORTRAN_LIBRARY" -DBUILD_TESTING=ON
cmake --build "$BUILD_DIR" --parallel 2
```

Targets: `qleaf_bench`, `build_nsg_subgraphs`, `build_nsg_subgraphs_ml`, `compute_ep_nsg`, and the three existing C++ tests. The root build selects the vendored library sources directly so that the two upstream projects can retain their original build files and target names. Upstream test programs are not part of the root build. `-DQLEAF_BUILD_TOOLS=OFF` configures search and its C++ tests without the external FAISS/BLAS/Boost dependencies.

`environment.yml` lists compiler and Python preparation packages. The skewed-label generator requires Python FAISS for ground truth; install bindings from the pinned FAISS commit in a compatible Python environment. Label-only generation uses NumPy and tqdm and does not require Python FAISS.
