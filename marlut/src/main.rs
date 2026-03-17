#![allow(dead_code)]
#![feature(bigint_helper_methods)]
#![feature(allocator_api)]
#![feature(portable_simd)]
#![feature(generic_const_exprs)]
#![feature(iter_array_chunks)]

pub mod aes;
pub mod chida;
pub mod lut256;
pub mod lut_sp;
pub mod lut_sp_malsec;
pub mod lut_sp_boolean;
pub mod lut_sp_boolean_malsec;
pub mod rep3_core;
pub mod share;
pub mod util;
pub mod wollut16;
pub mod wollut16_malsec;
#[macro_use]
pub mod benchmark;

use crate::benchmark::lut256_sp::{
    LUT256SPBenchmarkGF8, LUT256SPFp8AddBenchmark, LUT256SPFp16ExpBenchmark, LUT256SPMalFp8AddBenchmark, LUT256SPMalFp16ExpBenchmark
};
use crate::benchmark::lut256_sp_boolean::{LUT256SPBooleanBenchmark, LUT256SPBooleanMalBenchmark};
use crate::benchmark::{
    lut256::{
        LUT256Benchmark, Lut256SSBenchmark, Lut256SSMalBenchmark, Lut256SSMalOhvCheckBenchmark,
    },
    wollut16::LUT16Benchmark,
    wollut16_malsec::{MalLUT16BitStringBenchmark, MalLUT16OhvBenchmark},
};
use crate::rep3_core::network::{self, ConnectedParty};
use aes::AesVariant;
use benchmark::lut256_sp::IPBenchmark;
use benchmark::lut256_sp::LUT256SPBenchmark;
use benchmark::lut256_sp::LUT256SPMalBenchmark;
use benchmark::maepro10::Maepro10Benchmark;
use benchmark::random_bits::CutAndBucketBenchmark;
use itertools::Itertools;
use std::path::PathBuf;

use crate::benchmark::utils::{BenchmarkProtocol, BenchmarkResult};
use clap::{Parser, ValueEnum};

#[derive(Parser)]
struct Cli {
    #[arg(long, value_name = "FILE")]
    config: PathBuf,

    #[arg(
        long,
        value_name = "N_THREADS",
        help = "The number of worker threads. Set to 0 to indicate the number of cores on the machine. Optional, default single-threaded"
    )]
    threads: Option<usize>,

    #[arg(long, help = "The number of parallel AES calls to benchmark. You can pass multiple values.", num_args = 1..)]
    simd: Vec<usize>,

    #[arg(long, help = "The number repetitions of the protocol execution")]
    rep: usize,

    #[arg(
        long,
        help = "Path to write benchmark result data as CSV. Default: result.csv",
        default_value = "result.csv"
    )]
    csv: PathBuf,

    #[arg(
        long,
        help = "If set, benchmark all protocol variants and ignore specified targets.",
        default_value_t = false
    )]
    all: bool,

    #[arg(
        long,
        help = "If set, the benchmark will compute AES-256, otherwise AES-128 is computed",
        default_value_t = false
    )]
    aes256: bool,

    #[arg(value_enum)]
    target: Vec<ProtocolVariant>,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq, Hash)]
pub enum ProtocolVariant {
    /////////////// MAESTRO & others Protocols
    ///
    /// Implementation of the semi-honest variant of Protocol 3 using the lookup table protocol for length-16 one-hot vectors.
    ///
    /// - Security: semi-honest
    /// - Preprocessing: Generate length-16 random one-hot vectors (Protocol 8)
    /// - SubBytes step: GF(2^8) via tower GF(2^4)^2 (Protocol 3) and Protocol 4
    /// - Multiplication check: n/a
    Lut16,
    /// Implementation of the semi-honest variant described in Sect. 3.5.2
    ///
    /// - Security: semi-honest
    /// - Preprocessing: generate length-256 random one-hot vectors (Protocol 5)
    /// - SubBytes step: GF(2^8) inversion via length-256 table lookup (Protocol 4)
    /// - Multiplication check: n/a
    Lut256,
    /// Implementation of the semi-honest variant described in Sect. 3.5.3
    ///
    /// - Security: semi-honest
    /// - Preprocessing: generate 2x16 random one-hot vectors (Protocol 8 variant)
    /// - SubBytes step: GF(2^8) inversion via length-256 table lookup (Protocol 6)
    /// - Multiplication check: n/a
    Lut256Ss,
    /// Implementation of the actively secure variant described in Sect. 3.5.3
    ///
    /// - Security: active
    /// - Preprocessing: generate 2x16 random one-hot vectors (Protocol 8 variant)
    /// - SubBytes step: GF(2^8) inversion via length-256 table lookup (Protocol 6)
    /// - Multiplication check: Protocol 7 (VerifySbox)
    MalLut256Ss,
    /// Implementation of the actively secure variant described in Sect. 3.5.3
    ///
    /// - Security: active
    /// - Preprocessing: generate 2x16 random one-hot vectors (Protocol 8 variant) with reduced number of multiplication checks
    /// - SubBytes step: GF(2^8) inversion via length-256 table lookup (Protocol 6)
    /// - Multiplication check: Protocol 2 (Verify) + Protocol 7 (VerifySbox)
    MalLut256SsOpt,
    /// Implementation of the actively secure variant of Protocol 3 using the lookup table protocol for length-16 one-hot vectors.
    ///
    /// - Security: active
    /// - Preprocessing: Generate length-16 random one-hot vectors (Protocol 8)
    /// - SubBytes step: GF(2^8) via tower GF(2^4)^2 (Protocol 3) and Protocol 4
    /// - Multiplication check: Protocol 2 (Verify)
    MalLut16Bitstring,
    /// Implementation of the actively secure variant of Protocol 3 using the lookup table protocol for length-16 one-hot vectors.
    ///
    /// - Security: active
    /// - Preprocessing: Generate length-16 random one-hot vectors (Protocol 8) with reduced number of multiplication checks
    /// - SubBytes step: GF(2^8) via tower GF(2^4)^2 (Protocol 3) and Protocol 4 with reduced number of multiplication checks
    /// - Multiplication check: Protocol 2 (Verify)
    MalLut16Ohv,

    /////////////// MARLUT Protocols
    Lut256Sp,
    Lut256SpFp16Exp,
    Lut256SpFp8Add,
    Lut256SpGF8,
    MalLut256Sp,
    MalLut256SpFp16Exp,
    MalLut256SpFp8Add,

    Lut256SpBoolean,
    MalLut256SpBoolean,

    /////////////// Miscellaneous tests
    RandomBits,
    Maepro10,
    IPBenchmark,
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();

    let (party_index, config) = network::Config::from_file(&cli.config).unwrap();

    if cli.simd.is_empty() {
        return Err("simd parameter required".to_string());
    }
    if !cli.simd.iter().all_unique() {
        return Err(format!("Duplicate simd values in argument {:?}", cli.simd));
    }

    let mut boxed: Vec<Box<dyn BenchmarkProtocol>> = Vec::new();
    if !cli.all {
        // check non-empty and distinct targets
        if cli.target.is_empty() {
            let all_targets: Vec<_> = ProtocolVariant::value_variants()
                .iter()
                .map(|prot| prot.to_possible_value().unwrap().get_name().to_string())
                .collect();
            return Err(format!(
                "List of targets is empty: choose any number of targets: {:?}",
                all_targets
            ));
        }
        if !cli.target.iter().all_unique() {
            return Err(format!("Duplicate targets in argument {:?}", cli.target));
        }

        for v in cli.target {
            boxed.push(Box::new(v));
        }
    } else {
        // add all protocols to boxed
        for v in ProtocolVariant::value_variants() {
            boxed.push(Box::new(v.clone()));
        }
    }

    let variant = if cli.aes256 {
        AesVariant::Aes256
    } else {
        AesVariant::Aes128
    };

    benchmark::utils::benchmark_protocols(
        party_index,
        &config,
        variant,
        cli.rep,
        cli.simd,
        cli.threads,
        boxed,
        cli.csv,
    )
    .unwrap();
    Ok(())
}

impl ProtocolVariant {
    fn get_protocol(&self) -> &dyn BenchmarkProtocol {
        match self {
            ProtocolVariant::Lut16 => &LUT16Benchmark,
            ProtocolVariant::Lut256 => &LUT256Benchmark,
            ProtocolVariant::Lut256Ss => &Lut256SSBenchmark,
            ProtocolVariant::Lut256Sp => &LUT256SPBenchmark,
            ProtocolVariant::Lut256SpFp16Exp => &LUT256SPFp16ExpBenchmark,
            ProtocolVariant::Lut256SpFp8Add => &LUT256SPFp8AddBenchmark,
            ProtocolVariant::Lut256SpGF8 => &LUT256SPBenchmarkGF8,
            ProtocolVariant::MalLut256Sp => &LUT256SPMalBenchmark,
            ProtocolVariant::MalLut256Ss => &Lut256SSMalBenchmark,
            ProtocolVariant::MalLut256SsOpt => &Lut256SSMalOhvCheckBenchmark,
            ProtocolVariant::MalLut256SpFp16Exp => &LUT256SPMalFp16ExpBenchmark,
            ProtocolVariant::MalLut256SpFp8Add => &LUT256SPMalFp8AddBenchmark,
            ProtocolVariant::MalLut16Bitstring => &MalLUT16BitStringBenchmark,
            ProtocolVariant::MalLut16Ohv => &MalLUT16OhvBenchmark,
            ProtocolVariant::Lut256SpBoolean => &LUT256SPBooleanBenchmark,
            ProtocolVariant::MalLut256SpBoolean => &LUT256SPBooleanMalBenchmark,
            ProtocolVariant::RandomBits => &CutAndBucketBenchmark,
            ProtocolVariant::Maepro10 => &Maepro10Benchmark,
            ProtocolVariant::IPBenchmark => &IPBenchmark,
        }
    }
}

impl BenchmarkProtocol for ProtocolVariant {
    fn protocol_name(&self) -> String {
        self.get_protocol().protocol_name()
    }
    fn run(
        &self,
        conn: ConnectedParty,
        variant: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_str: Option<String>,
    ) -> BenchmarkResult {
        self.get_protocol()
            .run(conn, variant, simd, n_worker_threads, prot_str)
    }
}
