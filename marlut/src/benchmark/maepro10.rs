use crate::aes::AesVariant;
use crate::lut_sp::mae_pro10::{gen_random_bits, run_pro_10};
use crate::lut_sp::{
    VerificationRecordVec, VerificationRecorder,
};
use crate::rep3_core::party::{MainParty, broadcast::BroadcastContext};
use crate::rep3_core::share::RssShare;
use crate::rep3_core::{network::ConnectedParty, party::CombinedCommStats};
use crate::share::unsigned_ring::{UR8, UR16, UR32};
use crate::share::{Field, gf8::GF8, gf4::GF4};
use std::time::{Duration, Instant};

use crate::benchmark::utils::{BenchmarkProtocol, BenchmarkResult};

type F = GF8;
// type F = UR8;
pub struct Maepro10Benchmark;

impl BenchmarkProtocol for Maepro10Benchmark {
    fn protocol_name(&self) -> String {
        "mae_pro10".to_string()
    }

    fn run(
        &self,
        conn: ConnectedParty,
        _: AesVariant,
        simd: usize,
        n_worker_threads: Option<usize>,
        prot_str: Option<String>,
    ) -> BenchmarkResult {
        let mut party = MainParty::setup(conn, n_worker_threads, prot_str).unwrap();

        let mut context = BroadcastContext::new();
        let mut recorder = <VerificationRecordVec<F> as VerificationRecorder<F>>::new();

        let k = 8;
        let table = vec![F::new(14); 1 << k];

        let ind = gen_random_bits(&mut party, 1, &mut recorder);

        let index = RssShare {
            si: ind.si[0],
            sii: ind.sii[0],
        };

        // test
        let start = Instant::now();
        let (pre_time, pre_com, online_time, online_com) = run_pro_10(
            &table,
            &index,
            k,
            &mut party,
            &mut context,
            &mut recorder,
            simd,
        );
        let duration = start.elapsed();

        // communication statistics
        let comm_stats = party.io().reset_comm_stats();

        party.teardown().unwrap();

        BenchmarkResult::new(
            pre_time,
            online_time,
            Duration::from_secs(0),
            pre_com,
            online_com,
            CombinedCommStats::empty(),
            party.get_additional_timers(),
        )
    }
}
