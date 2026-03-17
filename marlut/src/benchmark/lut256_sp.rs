use crate::aes::AesVariant;
use crate::lut_sp::lut256_tables::{FP8AddTable, GF8InvTableNonPreshifted};
use crate::lut_sp::{
    VerificationRecorder,
    lut256_tables::{FP16ExpTable, GF8InvTable},
};
use crate::rep3_core::network::task::Direction;
use crate::rep3_core::party::{Party, broadcast::BroadcastContext};
use crate::rep3_core::{network::ConnectedParty, party::CombinedCommStats, party::CommStats};
use crate::share::{Field, mersenne61::Mersenne61, unsigned_ring::UR16};
use crate::util::ArithmeticBlackBox;
use crate::{
    aes::GF8InvBlackBox,
    lut_sp::{LUT256SP, NoVerificationRecording, VerificationRecordVec},
    rep3_core::share::RssShare,
    share::{gf8::GF8, unsigned_ring::UR8},
};

use crate::lut_sp_malsec::mult_verification::{
    DummyMalTable, reconstruct, verify_multiplication_triples,
};
// test for multipilication
use std::thread;
use std::time::{Duration, Instant};

use crate::benchmark::utils::{BenchmarkProtocol, BenchmarkResult};

pub struct LUT256SPBenchmark;

impl BenchmarkProtocol for LUT256SPBenchmark {
    fn protocol_name(&self) -> String {
        "lut_sp".to_string()
    }

    fn run(
        &self,
        conn: ConnectedParty,
        _variant: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_string: Option<String>,
    ) -> BenchmarkResult {
        let mut party = LUT256SP::<UR8, NoVerificationRecording, false>::setup(
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
        let mut input: Vec<(Vec<UR8>, Vec<UR8>)> = (0..n_blocks)
            .map(|_| {
                let shares: Vec<RssShare<UR8>> = party.generate_random(simd);
                shares.iter().map(|share| (share.si, share.sii)).unzip()
            })
            .collect::<Vec<_>>();

        let start = Instant::now();
        for (si, sii) in &mut input {
            unsafe {
                party
                    .gf8_inv(
                        &mut *(&mut si[..] as *mut [UR8] as *mut [GF8]),
                        &mut *(&mut sii[..] as *mut [UR8] as *mut [GF8]),
                    )
                    .unwrap();
            }
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

pub struct LUT256SPBenchmarkGF8;

impl BenchmarkProtocol for LUT256SPBenchmarkGF8 {
    fn protocol_name(&self) -> String {
        "lut_sp_gf8".to_string()
    }

    fn run(
        &self,
        conn: ConnectedParty,
        _variant: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_string: Option<String>,
    ) -> BenchmarkResult {
        let mut party = LUT256SP::<GF8, NoVerificationRecording, false>::setup(
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
                .preprocess::<GF8InvTableNonPreshifted>(simd, n_blocks)
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
        let mut input: Vec<(Vec<GF8>, Vec<GF8>)> = (0..n_blocks)
            .map(|_| {
                let shares: Vec<RssShare<GF8>> = party.generate_random(simd);
                shares.iter().map(|share| (share.si, share.sii)).unzip()
            })
            .collect::<Vec<_>>();

        let start = Instant::now();
        for (si, sii) in &mut input {
            party
                .lut::<true, GF8InvTableNonPreshifted>(&si, &sii)
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

pub struct LUT256SPMalBenchmark;

impl BenchmarkProtocol for LUT256SPMalBenchmark {
    fn protocol_name(&self) -> String {
        "lut_sp_mal".to_string()
    }

    fn run(
        &self,
        conn: ConnectedParty,
        _variant: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_string: Option<String>,
    ) -> BenchmarkResult {
        let mut party = LUT256SP::<UR8, VerificationRecordVec<UR8>, true>::setup(
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
        let mut input: Vec<(Vec<UR8>, Vec<UR8>)> = (0..n_blocks)
            .map(|_| {
                let shares: Vec<RssShare<UR8>> = party.generate_random(simd);
                println!("thread = {:?}", thread::current().id());
                shares.iter().map(|share| (share.si, share.sii)).unzip()
            })
            .collect::<Vec<_>>();

        let start = Instant::now();
        for (si, sii) in &mut input {
            unsafe {
                party
                    .gf8_inv(
                        &mut *(&mut si[..] as *mut [UR8] as *mut [GF8]),
                        &mut *(&mut sii[..] as *mut [UR8] as *mut [GF8]),
                    )
                    .unwrap();
            }
        }
        party.finalize_online::<GF8InvTable>().unwrap();
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

pub struct LUT256SPFp16ExpBenchmark;

impl BenchmarkProtocol for LUT256SPFp16ExpBenchmark {
    fn protocol_name(&self) -> String {
        "lut_sp_fp16exp".to_string()
    }

    fn run(
        &self,
        conn: ConnectedParty,
        _variant: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_string: Option<String>,
    ) -> BenchmarkResult {
        let mut party = LUT256SP::<UR16, NoVerificationRecording, false>::setup(
            conn,
            n_worker_threads,
            prot_string,
        )
        .unwrap();
        let _setup_comm_stats = party.main_party_mut().io().reset_comm_stats();
        println!("After setup");

        assert!(simd % 256 == 0);

        let n_blocks = 1;
        let divider = (n_blocks) as u64;

        let (prep_time, prep_comm_stats) = {
            let start = Instant::now();
            party.preprocess::<FP16ExpTable>(simd, n_blocks).unwrap();
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
        let mut input: Vec<(Vec<UR16>, Vec<UR16>)> = (0..n_blocks)
            .map(|_| {
                let shares: Vec<RssShare<UR16>> = party.generate_random(simd);
                shares.iter().map(|share| (share.si, share.sii)).unzip()
            })
            .collect::<Vec<_>>();

        let start = Instant::now();
        for (si, sii) in &mut input {
            party.lut::<true, FP16ExpTable>(&si, &sii).unwrap();

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

pub struct LUT256SPMalFp16ExpBenchmark;

impl BenchmarkProtocol for LUT256SPMalFp16ExpBenchmark {
    fn protocol_name(&self) -> String {
        "lut_sp_mal_fp16exp".to_string()
    }

    fn run(
        &self,
        conn: ConnectedParty,
        _variant: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_string: Option<String>,
    ) -> BenchmarkResult {
        let mut party = LUT256SP::<UR16, VerificationRecordVec<UR16>, true>::setup(
            conn,
            n_worker_threads,
            prot_string,
        )
        .unwrap();
        let _setup_comm_stats = party.main_party_mut().io().reset_comm_stats();
        println!("After setup");

        assert!(simd % 256 == 0);

        let n_blocks = 1;

        let divider = (n_blocks) as u64;

        let (prep_time, prep_comm_stats) = {
            let start = Instant::now();
            party.preprocess::<FP16ExpTable>(simd, n_blocks).unwrap();
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
        let mut input: Vec<(Vec<UR16>, Vec<UR16>)> = (0..n_blocks)
            .map(|_| {
                let shares: Vec<RssShare<UR16>> = party.generate_random(simd);
                shares.iter().map(|share| (share.si, share.sii)).unzip()
            })
            .collect::<Vec<_>>();

        let start = Instant::now();
        for (si, sii) in &mut input {
            party.lut::<true, FP16ExpTable>(&si, &sii).unwrap();

            let (new_si, new_sii) = party.temp_vecs.as_ref().unwrap();
            si.copy_from_slice(&new_si);
            sii.copy_from_slice(&new_sii);
        }
        party.finalize_online::<FP16ExpTable>().unwrap();
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

pub struct IPBenchmark;

impl BenchmarkProtocol for IPBenchmark {
    fn protocol_name(&self) -> String {
        "ip_mul".to_string()
    }

    fn run(
        &self,
        conn: ConnectedParty,
        _variant: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_string: Option<String>,
    ) -> BenchmarkResult {

        type U = UR16;

        let mut party = LUT256SP::<U, VerificationRecordVec<U>, true>::setup(
            conn,
            n_worker_threads,
            prot_string,
        )
        .unwrap();
        let mut context = BroadcastContext::new();
        let mut recorder = <VerificationRecordVec<U> as VerificationRecorder<U>>::new();
        let mut recorders = vec![];


        // create random input for benchmarking purposes
        let shares1: Vec<RssShare<U>> = party.generate_random(simd);
        let shares2: Vec<RssShare<U>> = party.generate_random(simd);

        let mut ci = vec![U::new(0); simd];
        let mut cii = vec![U::new(0); simd];

        // 
        let ai: Vec<U> = shares1.iter().map(|s| s.si).collect();
        let aii: Vec<U> = shares1.iter().map(|s| s.sii).collect();
        let bi: Vec<U> = shares2.iter().map(|s| s.si).collect();
        let bii: Vec<U> = shares2.iter().map(|s| s.sii).collect();

        let _setup_comm_stats = party.main_party_mut().io().reset_comm_stats();
        println!("After setup");

        let start = Instant::now();
        let alpha = recorder.generate_alpha(party.inner.as_party_mut(), simd);

        // let p = party.inner.as_party_mut();

        debug_assert_eq!(ci.len(), ai.len());
        debug_assert_eq!(ci.len(), aii.len());
        debug_assert_eq!(ci.len(), bi.len());
        debug_assert_eq!(ci.len(), bii.len());
        debug_assert_eq!(ci.len(), cii.len());

        for (i, alpha_i) in alpha.into_iter().enumerate() {
            ci[i] = ai[i] * bi[i] + ai[i] * bii[i] + aii[i] * bi[i] + alpha_i;
        }
        let rcv = party.inner.as_party_mut().receive_field_slice(Direction::Next, &mut cii);
        party.inner.as_party_mut().send_field_slice(Direction::Previous, &mut ci);
        rcv.rcv().unwrap(); 

        recorder.record_mul_triple(&ai, &aii, &bi, &bii, &ci, &cii);

        recorders.push(recorder);

        let (ip_triples, mul_triples, zero_si, zero_sii) =
            <VerificationRecordVec<U> as VerificationRecorder<U>>::finalize(recorders);

        let semi = start.elapsed();

        let semi_comm = party.main_party_mut().io().reset_comm_stats();

        let a = reconstruct(party.inner.as_party_mut(), &mut context, shares1[1]).unwrap();

        let b = reconstruct(party.inner.as_party_mut(), &mut context, shares2[1]).unwrap();

        println!("a: {:?}, b: {:?}", a, b);

        let c_share = RssShare {
            si: ci[1],
            sii: cii[1],
        };

        let c = reconstruct(party.inner.as_party_mut(), &mut context, c_share).unwrap();

        assert_eq!(c, a * b);

        let start = Instant::now();

        let res = verify_multiplication_triples::<3, U, Mersenne61, DummyMalTable>(
            party.inner.as_party_mut(),
            &mut party.context,
            &ip_triples,
            &mul_triples,
            40,
        )
        .unwrap();
        assert!(res);

        let duration = start.elapsed();
        let dzkp_comm = party.main_party_mut().io().reset_comm_stats();

        party.main_party_mut().teardown().unwrap();
        println!("After teardown");

        BenchmarkResult::new(
            semi,
            duration,
            Duration::from_secs(0),
            semi_comm,
            dzkp_comm,
            CombinedCommStats::empty(),
            party.main_party_mut().get_additional_timers(),
        )
    }
}

pub struct LUT256SPFp8AddBenchmark;

impl BenchmarkProtocol for LUT256SPFp8AddBenchmark {
    fn protocol_name(&self) -> String {
        "lut_sp_fp8add".to_string()
    }

    fn run(
        &self,
        conn: ConnectedParty,
        _variant: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_string: Option<String>,
    ) -> BenchmarkResult {
        let mut party = LUT256SP::<UR8, NoVerificationRecording, false>::setup(
            conn,
            n_worker_threads,
            prot_string,
        )
        .unwrap();
        let _setup_comm_stats = party.main_party_mut().io().reset_comm_stats();
        println!("After setup");

        assert!(simd % 256 == 0);

        let n_blocks = 1;
        let divider = (n_blocks) as u64;

        let (prep_time, prep_comm_stats) = {
            let start = Instant::now();
            party.preprocess::<FP8AddTable>(simd, n_blocks).unwrap();
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
        let mut input: Vec<(Vec<UR8>, Vec<UR8>)> = (0..n_blocks)
            .map(|_| {
                let shares: Vec<RssShare<UR8>> = party.generate_random(simd * 2);
                shares.iter().map(|share| (share.si, share.sii)).unzip()
            })
            .collect::<Vec<_>>();

        let start = Instant::now();
        for (si, sii) in &mut input {
            party.lut::<true, FP8AddTable>(&si, &sii).unwrap();

            let (new_si, new_sii) = party.temp_vecs.as_ref().unwrap();
            assert_eq!(new_si.len(), si.len() / 2);
            assert_eq!(new_sii.len(), sii.len() / 2);
            si[..new_si.len()].copy_from_slice(&new_si);
            sii[..new_sii.len()].copy_from_slice(&new_sii);
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

pub struct LUT256SPMalFp8AddBenchmark;

impl BenchmarkProtocol for LUT256SPMalFp8AddBenchmark {
    fn protocol_name(&self) -> String {
        "lut_sp_mal_fp8add".to_string()
    }

    fn run(
        &self,
        conn: ConnectedParty,
        _variant: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_string: Option<String>,
    ) -> BenchmarkResult {
        let mut party = LUT256SP::<UR8, VerificationRecordVec<UR8>, true>::setup(
            conn,
            n_worker_threads,
            prot_string,
        )
        .unwrap();
        let _setup_comm_stats = party.main_party_mut().io().reset_comm_stats();
        println!("After setup");

        assert!(simd % 256 == 0);

        let n_blocks = 1;

        let divider = (n_blocks) as u64;

        let (prep_time, prep_comm_stats) = {
            let start = Instant::now();
            party.preprocess::<FP8AddTable>(simd, n_blocks).unwrap();
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
        let mut input: Vec<(Vec<UR8>, Vec<UR8>)> = (0..n_blocks)
            .map(|_| {
                let shares: Vec<RssShare<UR8>> = party.generate_random(simd * 2);
                shares.iter().map(|share| (share.si, share.sii)).unzip()
            })
            .collect::<Vec<_>>();

        let start = Instant::now();
        for (si, sii) in &mut input {
            party.lut::<true, FP8AddTable>(&si, &sii).unwrap();

            let (new_si, new_sii) = party.temp_vecs.as_ref().unwrap();
            si[..new_si.len()].copy_from_slice(&new_si);
            sii[..new_sii.len()].copy_from_slice(&new_sii);
        }
        party.finalize_online::<FP8AddTable>().unwrap();
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
