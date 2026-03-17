# MARLUT

This repo implements Marlut, a maliciously secure lookup table (LUT) protocol over $\mathbb{Z}_{2^k}$ or $\mathbb{GF}(2^k)$.

We regret that the code might not be the most concise or readable as it contains tons of SIMD optimizations and employs metaprogramming techniques.

## Setup and Building

Rust Nightly is required. Evaluations in our paper were run with `rustc 1.91.0-nightly (040a98af7 2025-08-20)`.

Our protocols support 2- and 3-dimensional benchmarks. You can build them with:
```bash
# 2-dimensional
cargo build --release
# 3-dimensional
cargo build --release --no-default-features --features "clmul GF8_3 GF8NonPre_3 FP16Exp_3 FP8Add_3 Boolean_3"
```
AVX512 is optional, but machines with AVX512 support will exhibit significantly better performance due to tailored optimizations.

## Benchmarking
The CLI for the benchmark binary (run `target/release/marlut -h`) offers some description and help on the parameters. We have one single protocol with two implementations:
- `lut_sp` / `lut_sp_malsec`: Generic protocol for $\mathbb{Z}\_{2^k}$ and $\mathbb{GF}(2^k)$, but optimized for $\mathbb{Z}\_{2^k}$. Malicious security for $\mathbb{Z}\_{2^k}$.
- `lut_sp_boolean` / `lut_sp_boolean_malsec`: Optimized for $\mathbb{GF}(2^k)$. Malicious security for $\mathbb{GF}(2^k)$.

Benchmarks for our protocol are as follows
- with semi-honest security
    - `lut256-sp`: Evaluating single-input tables of size $2^8$ over ring $\mathbb{Z}_{2^8}$.
    - `lut256-sp-fp16-exp`: Evaluating the exponential function with a single-input table of size $2^{16}$ over ring $\mathbb{Z}_{2^{16}}$. The table can be replaced to compute the Sigmoid function.
    - `lut256-sp-fp8-add`: Evaluating floating-point addition with a two-input table of size $2^{16}$ over $\mathbb{Z}_{2^{8}}$. The table can be replaced to compute floating-point multiplication.
    - `lut256-sp-gf8`: Evaluating a single-input table of size $2^8$ over $\mathbb{GF}(2^8)$. This uses the generic implementation.
    - `lut256-sp-boolean`: Evaluating a single-input table of size $2^8$ over $\mathbb{GF}(2^8)$. This uses the optimized boolean implementation.

- with active security
    - `mal-lut256-sp`: The maliciously secure version of `lut256-sp`.
    - `mal-lut256-sp-fp16-exp`: The maliciously secure version of `lut256-sp-fp16-exp`.
    - `mal-lut256-sp-fp8-add`: The maliciously secure version of `lut256-sp-fp8-add`.
    - `mal-lut256-sp-boolean`: The maliciously secure version of `lut256-sp-boolean`.

In addition, most benchmarks from MAESTRO are retained. For a fair comparison, it is advised to run our version because we have made some optimizations to the networking framework.

To run the benchmarks, you may refer to `run_bench.sh`. You should run the script on the three machines
```bash
./run_bench.sh p1.toml # Machine 1
./run_bench.sh p2.toml # Machine 2
./run_bench.sh p3.toml # Machine 3
```
Remember to update the machine IP addresses in `p{1,2,3}.toml`. For more details, please also refer to `README_MAESTRO.md`.
