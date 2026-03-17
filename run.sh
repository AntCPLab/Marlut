target/release/marlut --config p1.toml --threads 8 --simd 262144 --rep 10 --csv result-p1.csv lut256-sp &
target/release/marlut --config p2.toml --threads 8 --simd 262144 --rep 10 --csv result-p2.csv lut256-sp &
target/release/marlut --config p3.toml --threads 8 --simd 262144 --rep 10 --csv result-p3.csv lut256-sp &
wait
