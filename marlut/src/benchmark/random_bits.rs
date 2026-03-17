use crate::aes::AesVariant;
use crate::lut_sp::cut_bucket::cut_and_bucket;
use crate::lut_sp::{NoVerificationRecording, VerificationRecordVec, VerificationRecorder};
use crate::rep3_core::party::{MainParty, broadcast::BroadcastContext};
use crate::rep3_core::{network::ConnectedParty, party::CombinedCommStats};
use crate::share::{Field, gf8::GF8, unsigned_ring::UR8};
use std::time::{Duration, Instant};

use crate::benchmark::utils::{BenchmarkProtocol, BenchmarkResult};

pub struct CutAndBucketBenchmark;

impl BenchmarkProtocol for CutAndBucketBenchmark {
    fn protocol_name(&self) -> String {
        "cut_and_bucket".to_string()
    }

    fn run(
        &self,
        conn: ConnectedParty,
        _: AesVariant,
        _: usize,
        n_worker_threads: Option<usize>,
        prot_str: Option<String>,
    ) -> BenchmarkResult {
        let mut party = MainParty::setup(conn, n_worker_threads, prot_str).unwrap();
        // let setup_comm_stats = party.io().reset_comm_stats();

        let n = 1 << 20; //
        let b = 3; // bucket
        let c = 3; // cut stage check bits

        //
        let mut context = BroadcastContext::new();
        let mut recorder = <VerificationRecordVec<UR8> as VerificationRecorder<UR8>>::new();

        // test
        let start = Instant::now();
        let (bitsi, bitsii) = cut_and_bucket::<UR8>(
            &mut party,
            &mut context,
            n,
            &mut recorder,
            //expected_hash,
            b,
            c,
        );
        let duration = start.elapsed();

        // communication statistics
        let offline_comm_stats = party.io().reset_comm_stats();

        party.teardown().unwrap();

        BenchmarkResult::new(
            duration,
            Duration::from_secs(0),
            Duration::from_secs(0),
            offline_comm_stats,
            CombinedCommStats::empty(),
            CombinedCommStats::empty(),
            party.get_additional_timers(),
        )
    }
}
