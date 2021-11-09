# MOTION2NX -- A Framework for Generic Hybrid Two-Party Computation and Private Inference with Neural Networks

This software is an extension of the [MOTION framework for multi-party
computation](https://github.com/encryptogroup/MOTION).
We additionally implemented five 2PC protocols with passive security together
with all 20 possible conversions among each other to enable private evaluation
of hybrid circuits:

- Yao's Garbled Circuits with FreeXOR and Half-Gates
- Arithmetic and Boolean variants of Goldreich-Micali-Wigderson
- Arithmetic and Boolean variants of the secret-sharing-based protocols from [ABY2.0 (Patra et al., USENIX Security '21)](https://eprint.iacr.org/2020/1225)

Moreover, we support private inference with neural networks by providing secure
tensor data types and specialized building blocks for common tensor operations.
With support of the [Open Neural Network Exchange (ONNX)](https://onnx.ai) file
format, this makes our framework interoperable with industry-standard deep
learning frameworks such as TensorFlow and PyTorch.

Compared to the original MOTION codebase, we made architectural improvements
to increase flexibility and performance of the framework.
Although the interfaces of this work are currently not compatible with the
original framework due to the concurrent development of both branches, it is
planned to integrate the MOTION2NX features into MOTION itself.


More information about this work is given in [this extended
abstract](https://encrypto.de/papers/BCS21PriMLNeurIPS.pdf) which was accepted
at the [PriML@NeurIPS 2021](https://priml2021.github.io/) workshop.
It is the result of Lennart Braun's master's thesis in the [ENCRYPTO
group](https://encrypto.de) at [TU
Darmstadt](https://www.informatik.tu-darmstadt.de) supervised by Thomas
Schneider and Rosario Cammarota.

This code is provided as a experimental implementation for testing purposes and
should not be used in a productive environment. We cannot guarantee security
and correctness.


## Build Instructions


This software was developed and tested in the following environment (it might
also work with older versions):

- [Arch Linux](https://archlinux.org/)
- [GCC 11.1.0](https://gcc.gnu.org/) or [Clang/LLVM 12.0.1](https://clang.llvm.org/)
- [CMake 3.21.4](https://cmake.org/)
- [Boost 1.76.0](https://www.boost.org/)
- [fmt 8.0.1](https://github.com/fmtlib/fmt)
- [flatbuffers 2.0.0](https://github.com/google/flatbuffers)
- [GoogleTest 1.11.0 (optional, for tests, build automatically)](https://github.com/google/googletest)
- [Google Benchmark 1.6.0 (optional, for some benchmarks, build automatically)](https://github.com/google/benchmark)
- [HyCC (optional, for the HyCCAdapter)](https://gitlab.com/securityengineering/HyCC)
- [ONNX 1.10.2 (optional, for the ONNXAdapter)](https://github.com/onnx/onnx)

The build system downloads and builds GoogleTest and Benchmark if required.
It also tries to download and build Boost, fmt, and flatbuffers if it cannot
find these libraries in the system.

The framework can for example be compiled as follows:
```
$ CC=gcc CXX=g++ cmake \
    -B build_debwithrelinfo_gcc_foo \
    -DCMAKE_BUILD_TYPE=DebWithRelInfo \
    -DMOTION_BUILD_EXE=On \
    -DMOTION_BUILD_TESTS=On \
    -DMOTION_USE_AVX=AVX2
$ cmake --build build_debwithrelinfo_gcc
```
Explanation of the flags:

- `CC=gcc CXX=g++`: select GCC as compiler
- `-B build_debwithrelinfo_gcc`: create a build directory
- `-DCMAKE_BUILD_TYPE=DebWithRelInfo`: compile with optimization and also add
  debug symbols -- makes tests run faster and debugging easier
- `-DMOTION_BUILD_EXE=On`: build example executables and benchmarks
- `-DMOTION_BUILD_TESTS=On`: build tests
- `-DMOTION_USE_AVX=AVX2`: compile with AVX2 instructions (choose one of `AVX`/`AVX2`/`AVX512`)

### HyCC Support for Hybrid Circuits

To enable support for HyCC circuits, the HyCC library must be compiled and the
following flags need additionally be passed to CMake:

- `-DMOTION_BUILD_HYCC_ADAPTER=On`
- `-DMOTION_HYCC_PATH=/path/to/HyCC` where `/path/to/HyCC` points to the HyCC
  directory, i.e., the top-level directory of the cloned repository

This builds the library target `motion_hycc` and the `hycc2motion` executable.



### ONNX Support for Neural Networks

For ONNX support, the ONNX library must be installed and the following flag
needs additionally be passed to CMake:

- `-DMOTION_BUILD_ONNX_ADAPTER=On`

This builds the library target `motion_onnx` and the `onnx2motion` executable.

### Example

```
$ ./bin/onnx2motion \
    --my-id ${PARTY_ID} \
    --party 0,::1,7000 \
    --party 1,::1,7001 \
    --arithmetic-protocol GMW \
    --boolean-protocol GMW \
    --model /path/to/model.onnx \
    --json
```
with "${PARTY_ID}" either 0 or 1.
