# Example: Using the MOTION2NX Low-Level API

This example showcases the low-level API of MOTION2NX, which is mainly based on
the [`CircuitBuilder`](/src/motioncore/base/circuit_builder.h) and
[`GateFactory`](/src/motioncore/base/gate_factory.h) interfaces, as well as
`WireVector`s (which can be used as opaque handles).
Inputs to the computation are given via promises and outputs are obtained from
futures.


## Manually Building a Circuit

The [`create_circuit`
function](/src/examples/millionaires_problem/millionaires_problem.cpp#L182-L220)
uses the low-level API to create a simple circuit for Yao's Millionaires'
Problem.
It includes code for inputs, outputs, conversions, and the use of builtin
circuits.


## Running the Application

To run the example, the binary need to be executed twice with the arguments for
both parties, e.g., locally in two terminals for testing.

### Party 0 with Human-Readable Output

```
$ ./bin/millionaires_problem --my-id 0 --party 0,::1,7000 --party 1,::1,7001 --arithmetic-protocol beavy --boolean-protocol yao --repetitions 5 --input-value 42
Party 1 has at least the same amount of money as Party 0
Party 1 has at least the same amount of money as Party 0
Party 1 has at least the same amount of money as Party 0
Party 1 has at least the same amount of money as Party 0
Party 1 has at least the same amount of money as Party 0
===========================================================================
millionaires_problem
===========================================================================
MOTION version: 104331e-dirty @ motion2nx
invocation: ./bin/millionaires_problem --my-id 0 --party 0,::1,7000 --party 1,::1,7001 --arithmetic-protocol beavy --boolean-protocol yao --repetitions 5 --input-value 42
by lennart@euler, PID 2074143
===========================================================================
Run time statistics over 5 iterations
---------------------------------------------------------------------------
                          mean        median        stddev
---------------------------------------------------------------------------
MT Presetup              0.002 ms      0.002 ms      0.000 ms
MT Setup                 0.141 ms      0.120 ms      0.042 ms
SP Presetup              0.000 ms      0.000 ms      0.000 ms
SP Setup                 0.000 ms      0.000 ms      0.000 ms
SB Presetup              0.002 ms      0.002 ms      0.000 ms
SB Setup                 0.002 ms      0.002 ms      0.000 ms
Base OTs                93.759 ms    100.487 ms     12.419 ms
OT Extension Setup      70.189 ms     68.739 ms     15.763 ms
---------------------------------------------------------------------------
Preprocessing Total    164.688 ms    149.145 ms     20.495 ms
Gates Setup              7.402 ms      5.098 ms      3.208 ms
Gates Online            41.287 ms     49.674 ms     19.235 ms
---------------------------------------------------------------------------
Circuit Evaluation     214.675 ms    209.122 ms     14.711 ms
===========================================================================
Communication with each other party:
Sent: 0.049 MiB in 455 messages
Received: 0.037 MiB in 390 messages
===========================================================================
```

### Party 1 with JSON Output

```
$ ./bin/millionaires_problem --my-id 1 --party 0,::1,7000 --party 1,::1,7001 --arithmetic-protocol beavy --boolean-protocol yao --repetitions 5 --input-value 47 --json
{"experiment":"millionaires_problem","meta":{"invocation":"./bin/millionaires_problem --my-id 1 --party 0,::1,7000 --party 1,::1,7001 --arithmetic-protocol beavy --boolean-protocol yao --repetitions 5 --input-value 47 --json","timestamp":"2021-12-13T23:18:29+0100.","user":"lennart","hostname":"euler","pid":2074132,"build-type":"RelWithDebInfo","git-branch":"motion2nx","git-commit":"104331eed9c136994d767c8ae84f4198c5c7a44b","git-version":"104331e-dirty"},"runtime":{"repetitions":5,"mt_setup":{"mean":1.563186E-1,"median":1.75511E-1,"stddev":4.045331155097187E-2},"sp_setup":{"mean":0E0,"median":0E0,"stddev":0E0},"sb_setup":{"mean":1.8156E-3,"median":1.8859999999999999E-3,"stddev":3.565392545008198E-4},"linalgtriple_setup":{"mean":0E0,"median":0E0,"stddev":0E0},"base_ots":{"mean":1.283591858E2,"median":1.43234904E2,"stddev":2.1517884385839448E1},"ot_extension_setup":{"mean":9.795938E-1,"median":1.0326039999999999E0,"stddev":2.223273615994218E-1},"preprocessing":{"mean":1.6514558159999999E2,"median":1.4946446699999998E2,"stddev":2.0364333848757084E1},"gates_setup":{"mean":4.06664142E1,"median":4.7577908E1,"stddev":1.4854024843236319E1},"gates_online":{"mean":6.6454678000000005E0,"median":7.019398E0,"stddev":2.5451598342314288E0},"evaluate":{"mean":2.157014238E2,"median":2.11624257E2,"stddev":1.4172259950973633E1}},"communication":{"bytes_sent":38340,"num_messages_sent":390,"bytes_received":51188,"num_messages_received":455},"party_id":1,"arithmetic_protocol":"ArithmeticBEAVY","boolean_protocol":"Yao","simd":1,"threads":0,"sync_between_setup_and_online":false}
```
