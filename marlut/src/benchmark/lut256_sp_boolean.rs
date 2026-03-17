use std::thread;
use std::time::{Duration, Instant};

use crate::aes::{AesVariant, GF8InvBlackBox};
use crate::benchmark::utils::{BenchmarkProtocol, BenchmarkResult};
use crate::lut_sp_boolean::lut256_tables::GF8InvTable;
use crate::lut_sp_boolean::LUT256SP;
use crate::lut_sp_boolean::{NoVerificationRecording, VerificationRecordVec};
use crate::rep3_core::network::ConnectedParty;
use crate::rep3_core::party::{CombinedCommStats, CommStats};
use crate::rep3_core::share::RssShare;
use crate::share::bs_bool8::BsBool8;
use crate::share::gf8::GF8;
use crate::util::ArithmeticBlackBox;

pub struct LUT256SPBooleanBenchmark;

impl BenchmarkProtocol for LUT256SPBooleanBenchmark {
    fn protocol_name(&self) -> String {
        "lut_sp_boolean".to_string()
    }

    fn run(
        &self,
        conn: ConnectedParty,
        _variant: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_string: Option<String>,
    ) -> BenchmarkResult {
        let mut party = LUT256SP::<NoVerificationRecording, false>::setup(
            conn,
            n_worker_threads,
            prot_string,
        )
        .unwrap();
        let _setup_comm_stats = party.main_party_mut().io().reset_comm_stats();
        println!("After setup");

        assert!(simd % 512 == 0);

        let n_blocks = 10;
        let divider = (n_blocks) as u64;

        let (prep_time, prep_comm_stats) = {
            let start = Instant::now();
            party
                .preprocess::<GF8InvTable>(simd, n_blocks)
                .unwrap();
            let prep_time = start.elapsed();
            let prep_comm_stats = party.main_party_mut().io().reset_comm_stats();
            println!("After preprocessing");
            (prep_time, prep_comm_stats)
        };
        // compute pre_comms
        let prep_comms = CombinedCommStats {
            prev: CommStats {
                bytes_received: prep_comm_stats.prev.bytes_received / divider,
                bytes_sent: prep_comm_stats.prev.bytes_sent / divider,
                rounds: prep_comm_stats.prev.rounds / divider as usize,
            },
            next: CommStats {
                bytes_received: prep_comm_stats.next.bytes_received / divider,
                bytes_sent: prep_comm_stats.next.bytes_sent / divider,
                rounds: prep_comm_stats.next.rounds / divider as usize,
            },
        };

        // create random input for benchmarking purposes
        let mut input: Vec<(Vec<BsBool8>, Vec<BsBool8>)> = (0..n_blocks)
            .map(|_| {
                let shares: Vec<RssShare<BsBool8>> = party.generate_random(simd);
                shares.iter().map(|share| (share.si, share.sii)).unzip()
            })
            .collect::<Vec<_>>();

        let start = Instant::now();
        for (si, sii) in &mut input {
            party
                .lut::<true, GF8InvTable>(&si, &sii)
                .unwrap();

            let (new_si, new_sii) = party.temp_vecs.as_ref().unwrap();
            si.copy_from_slice(&new_si);
            sii.copy_from_slice(&new_sii);
        }
        let duration = start.elapsed();
        println!("After online");
        let online_comm_stats = party.main_party_mut().io().reset_comm_stats();

        // compute online_comms
        let online_comms = CombinedCommStats {
            prev: CommStats {
                bytes_received: online_comm_stats.prev.bytes_received / divider,
                bytes_sent: online_comm_stats.prev.bytes_sent / divider,
                rounds: online_comm_stats.prev.rounds / divider as usize,
            },
            next: CommStats {
                bytes_received: online_comm_stats.next.bytes_received / divider,
                bytes_sent: online_comm_stats.next.bytes_sent / divider,
                rounds: online_comm_stats.next.rounds / divider as usize,
            },
        };

        let (finalize_time, finalize_comm_stats) =
            (Duration::from_secs(0), CombinedCommStats::empty());

        party.main_party_mut().teardown().unwrap();
        println!("After teardown");

        BenchmarkResult::new(
            prep_time / divider as u32,
            duration / divider as u32,
            finalize_time,
            prep_comms,
            online_comms,
            finalize_comm_stats,
            party.main_party_mut().get_additional_timers(),
        )
    }
}

pub struct LUT256SPBooleanMalBenchmark;

impl BenchmarkProtocol for LUT256SPBooleanMalBenchmark {
    fn protocol_name(&self) -> String {
        "lut_sp_boolean_mal".to_string()
    }

    fn run(
        &self,
        conn: ConnectedParty,
        _variant: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_string: Option<String>,
    ) -> BenchmarkResult {
        let mut party = LUT256SP::<VerificationRecordVec, true>::setup(
            conn,
            n_worker_threads,
            prot_string,
        )
        .unwrap();
        let _setup_comm_stats = party.main_party_mut().io().reset_comm_stats();
        println!("After setup");

        assert!(simd % 512 == 0);

        let n_blocks = 10;
        let divider = (n_blocks) as u64;

        let (prep_time, prep_comm_stats) = {
            let start = Instant::now();
            party.preprocess::<GF8InvTable>(simd, n_blocks).unwrap();
            let prep_time = start.elapsed();
            let prep_comm_stats = party.main_party_mut().io().reset_comm_stats();
            println!("After preprocessing");
            (prep_time, prep_comm_stats)
        };

        // compute pre_comms

        let prep_comms = CombinedCommStats {
            prev: CommStats {
                bytes_received: prep_comm_stats.prev.bytes_received / divider,
                bytes_sent: prep_comm_stats.prev.bytes_sent / divider,
                rounds: prep_comm_stats.prev.rounds / divider as usize,
            },
            next: CommStats {
                bytes_received: prep_comm_stats.next.bytes_received / divider,
                bytes_sent: prep_comm_stats.next.bytes_sent / divider,
                rounds: prep_comm_stats.next.rounds / divider as usize,
            },
        };

        // create random input for benchmarking purposes
        let mut input: Vec<(Vec<BsBool8>, Vec<BsBool8>)> = (0..n_blocks)
            .map(|_| {
                let shares: Vec<RssShare<BsBool8>> = party.generate_random(simd);
                println!("thread = {:?}", thread::current().id());
                shares.iter().map(|share| (share.si, share.sii)).unzip()
            })
            .collect::<Vec<_>>();

        let start = Instant::now();
        for (si, sii) in &mut input {
            unsafe {
                party
                    .gf8_inv(
                        &mut *(&mut si[..] as *mut [BsBool8] as *mut [GF8]),
                        &mut *(&mut sii[..] as *mut [BsBool8] as *mut [GF8]),
                    )
                    .unwrap();
            }
        }
        party.finalize_online().unwrap();
        let duration = start.elapsed();
        println!("After online");
        let online_comm_stats = party.main_party_mut().io().reset_comm_stats();

        // compute online_comms
        let online_comms = CombinedCommStats {
            prev: CommStats {
                bytes_received: online_comm_stats.prev.bytes_received / divider,
                bytes_sent: online_comm_stats.prev.bytes_sent / divider,
                rounds: online_comm_stats.prev.rounds / divider as usize,
            },
            next: CommStats {
                bytes_received: online_comm_stats.next.bytes_received / divider,
                bytes_sent: online_comm_stats.next.bytes_sent / divider,
                rounds: online_comm_stats.next.rounds / divider as usize,
            },
        };

        let (finalize_time, finalize_comm_stats) =
            (Duration::from_secs(0), CombinedCommStats::empty());

        party.main_party_mut().teardown().unwrap();
        println!("After teardown");

        BenchmarkResult::new(
            prep_time / divider as u32,
            duration / divider as u32,
            finalize_time,
            prep_comms,
            online_comms,
            finalize_comm_stats,
            party.main_party_mut().get_additional_timers(),
        )
    }
}
