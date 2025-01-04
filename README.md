# crabml

[![](https://img.shields.io/discord/1111711408875393035?logo=discord&label=discord)](https://discord.gg/wbzqddT3QC)

`crabml` is a llama.cpp compatible (and equally fast!) AI inference engine written in 🦀 **Rust**, which runs everywhere with the help of 🎮 **WebGPU**.

## Project Goals

`crabml` is designed with the following objectives in mind:

- 🤖 Focus solely on inference.
- 🎮 Runs on browsers, desktops, and servers everywhere with the help of **WebGPU**.
- ⏩ **SIMD**-accelerated inference on inexpensive hardware.
- 💼 `mmap()` from day one, minimized memory requirement with various quantization support.
- 👾 Hackable & embeddable.

## Supported Models

`crabml` supports the following models in GGUF format:

- 🦙 Llama
- 🦙 CodeLlama
- 🦙 Gemma
- 〽️ Mistral
- 🚄 On the way: Mistral MoE, Phi, QWen, StarCoder, Llava, and more!

For more information, you can visit [How to Get GGUF Models](https://github.com/crabml/crabml/blob/main/docs/how-to-get-gguf-models.md) to learn how to download the GGUF files you need.

## Supported Quantization Methods

`crabml` supports the following quantization methods on CPUs with SIMD acceleration for ARM (including Apple Silicon) and x86 architectures:

|      | Bits   | Native CPU | NEON | AVX2 | RISC-V SIMD | WebGPU |
|------|--------|------------|------|------|-------------|--------|
| Q8_0 | 8 bits | ✅          | ✅    | ✅    | WIP         | WIP    |
| Q8_K | 8 bits | ✅          | ✅    | ✅    | WIP         | WIP    |
| Q6_K | 6 bits | ✅          | WIP  | WIP  | WIP         | WIP    |
| Q5_0 | 5 bits | ✅          | ✅    | ✅    | WIP         | WIP    |
| Q5_1 | 5 bits | ✅          | ✅    | ✅    | WIP         | WIP    |
| Q5_K | 5 bits | ✅          | WIP  | WIP  | WIP         | WIP    |
| Q4_0 | 4 bits | ✅          | ✅    | ✅    | WIP         | WIP    |
| Q4_1 | 4 bits | ✅          | ✅    | ✅    | WIP         | WIP    |
| Q4_K | 4 bits | ✅          | WIP  | WIP  | WIP         | WIP    |
| Q3_K | 3 bits | ✅          | WIP  | WIP  | WIP         | WIP    |
| Q2_K | 2 bits | ✅          | WIP  | WIP  | WIP         | WIP    |

As the table above suggests, WebGPU-accelerated quantizations are still under busy development, and `Q8_0`， `Q4_0`， `Q4_1` are currently the most recommended quantization methods on CPUs!

## Usage

### Building the Project

To build `crabml`, set the `RUSTFLAGS` environment variable to enable specific target features. For example, to enable NEON on ARM architectures, use `RUSTFLAGS="-C target-feature=+neon"`. Then build the project with the following command:

```bash
cargo build --release
```

This command compiles the project in release mode, which optimizes the binary for performance.

### Running an Example

After building the project, you can run an example inference by executing the `crabml-cli` binary with appropriate arguments. For instance, to use the `tinyllamas-stories-15m-f32.gguf` model to generate text based on the prompt "captain america", execute the command below:

```bash
./target/release/crabml-cli \
  -m ./testdata/tinyllamas-stories-15m-f32.gguf \
  "captain america" --steps 100 \
  -t 0.8 -p 1.0
```

In this command:

- `-m` specifies the checkpoint file.
- `--steps` defines the number of tokens to generate.
- `-t` sets the temperature, which controls the randomness of the output.
- `-p` sets the probability of sampling from the top-p.

## License

This contribution is licensed under Apache License, Version 2.0, ([LICENSE](LICENSE) or <http://www.apache.org/licenses/LICENSE-2.0>)
