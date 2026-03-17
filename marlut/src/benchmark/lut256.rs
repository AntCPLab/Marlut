use crate::aes::AesVariant;
use crate::rep3_core::party::Party;
use crate::rep3_core::share::RssShare;
use crate::rep3_core::{network::ConnectedParty, party::CombinedCommStats};
use crate::share::gf8::GF8;
use crate::util::ArithmeticBlackBox;
use crate::{
    aes::{
        self, GF8InvBlackBox,
        ss::{GF8InvBlackBoxSS, GF8InvBlackBoxSSMal},
    },
    lut256::{
        LUT256Party,
        lut256_ss::{Lut256SSMalParty, Lut256SSParty},
    },
};
use std::time::{Duration, Instant};

use crate::benchmark::utils::{BenchmarkProtocol, BenchmarkResult};

pub struct LUT256Benchmark;

impl BenchmarkProtocol for LUT256Benchmark {
    fn protocol_name(&self) -> String {
        "lut256".to_string()
    }
    fn run(
        &self,
        conn: ConnectedParty,
        variant: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_str: Option<String>,
    ) -> BenchmarkResult {
        assert_eq!(
            AesVariant::Aes128,
            variant,
            "Only AES-128 is supported for {}",
            self.protocol_name()
        );
        let mut party = LUT256Party::setup(conn, n_worker_threads, prot_str).unwrap();
        let _setup_comm_stats = party.main_party_mut().io().reset_comm_stats();
        println!("After setup");

        assert!(simd % 16 == 0);

        let start_prep = Instant::now();
        party.do_preprocessing(0, simd / 16, variant).unwrap();
        let prep_duration = start_prep.elapsed();
        let prep_comm_stats = party.main_party_mut().io().reset_comm_stats();
        println!("After pre-processing");

        // create random input for benchmarking purposes
        let mut input: Vec<(Vec<GF8>, Vec<GF8>)> = (0..variant.n_rounds())
            .map(|_| {
                let shares: Vec<RssShare<GF8>> = party.generate_random(simd);
                shares.iter().map(|share| (share.si, share.sii)).unzip()
            })
            .collect::<Vec<_>>();

        let start = Instant::now();
        for (si, sii) in &mut input {
            party.gf8_inv(si, sii).unwrap();
        }
        let duration = start.elapsed();
        println!("After online");
        let online_comm_stats = party.main_party_mut().io().reset_comm_stats();
        party.main_party_mut().teardown().unwrap();
        println!("After teardown");

        BenchmarkResult::new(
            prep_duration,
            duration,
            Duration::from_secs(0),
            prep_comm_stats,
            online_comm_stats,
            CombinedCommStats::empty(),
            party.main_party_mut().get_additional_timers(),
        )
    }
}

pub struct Lut256SSBenchmark;

impl BenchmarkProtocol for Lut256SSBenchmark {
    fn protocol_name(&self) -> String {
        "lut256_ss".to_string()
    }
    fn run(
        &self,
        _conn: ConnectedParty,
        _variant: AesVariant,
        _simd: usize,
        _n_worker_threads: Option<usize>,
        _prot_str: Option<String>,
    ) -> BenchmarkResult {
        unimplemented!();
    }
}

fn lut256_ss_mal_run_benchmark(
    conn: ConnectedParty,
    simd: usize,
    use_ohv_check: bool,
    n_worker_threads: Option<usize>,
    prot_str: Option<String>,
) -> BenchmarkResult {
    let mut party =
        Lut256SSMalParty::setup(conn, use_ohv_check, n_worker_threads, prot_str).unwrap();
    let _setup_comm_stats = party.main_party_mut().io().reset_comm_stats();
    println!("After setup");

    assert!(simd % 16 == 0);

    let start_prep = Instant::now();
    party.do_preprocessing(0, simd / 16).unwrap();
    let prep_duration = start_prep.elapsed();
    let prep_comm_stats = party.main_party_mut().io().reset_comm_stats();
    println!("After pre-processing");

    let variant = AesVariant::Aes128;
    // create random input for benchmarking purposes
    let mut input: Vec<(Vec<GF8>, Vec<GF8>)> = (0..variant.n_rounds())
        .map(|_| {
            let shares: Vec<RssShare<GF8>> = party.main_party_mut().generate_random(simd);
            shares.iter().map(|share| (share.si, share.sii)).unzip()
        })
        .collect::<Vec<_>>();

    let start = Instant::now();
    for (si, sii) in &mut input {
        party.gf8_inv_rss(si, sii).unwrap();
    }
    let duration = start.elapsed();
    println!("After online");

    let online_comm_stats = party.main_party_mut().io().reset_comm_stats();
    let finalize_start = Instant::now();
    party.finalize().unwrap();
    let finalize_time = finalize_start.elapsed();
    let finalize_comm_stats = party.main_party_mut().io().reset_comm_stats();
    println!("After finalize");
    party.main_party_mut().teardown().unwrap();
    println!("After teardown");

    BenchmarkResult::new(
        prep_duration,
        duration,
        finalize_time,
        prep_comm_stats,
        online_comm_stats,
        finalize_comm_stats,
        party.main_party_mut().get_additional_timers(),
    )
}

pub struct Lut256SSMalBenchmark;
impl BenchmarkProtocol for Lut256SSMalBenchmark {
    fn protocol_name(&self) -> String {
        "mal-lut256".to_string()
    }
    fn run(
        &self,
        conn: ConnectedParty,
        variant: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_str: Option<String>,
    ) -> BenchmarkResult {
        assert_eq!(
            AesVariant::Aes128,
            variant,
            "Only AES-128 is supported for {}",
            self.protocol_name()
        );
        lut256_ss_mal_run_benchmark(conn, simd, false, n_worker_threads, prot_str)
    }
}

pub struct Lut256SSMalOhvCheckBenchmark;
impl BenchmarkProtocol for Lut256SSMalOhvCheckBenchmark {
    fn protocol_name(&self) -> String {
        "mal-lut256-opt".to_string()
    }
    fn run(
        &self,
        conn: ConnectedParty,
        variant: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_str: Option<String>,
    ) -> BenchmarkResult {
        assert_eq!(
            AesVariant::Aes128,
            variant,
            "Only AES-128 is supported for {}",
            self.protocol_name()
        );
        lut256_ss_mal_run_benchmark(conn, simd, true, n_worker_threads, prot_str)
    }
}
