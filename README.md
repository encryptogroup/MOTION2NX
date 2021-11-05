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

This code is provided as a experimental implementation for testing purposes and should not be used in a productive environment. We cannot guarantee security and correctness.
