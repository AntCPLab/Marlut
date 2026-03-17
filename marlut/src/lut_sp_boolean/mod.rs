//! This module implements the MARLUT protocol optimized for boolean-like fields (e.g. GF(2^8)).

use std::{time::Duration, vec};

use itertools::zip_eq;

use crate::{
    rep3_core::{
        network::{task::IoLayerOwned, ConnectedParty},
        party::{broadcast::BroadcastContext, error::MpcResult, Party},
        share::HasZero,
    },
    share::bs_bool8::BsBool8,
    util::{
        aligned_vec::{AlignedAllocator, AlignedVec},
        mul_triple_vec::{InnerProductTripleBoolean, ManyToOneMulTripleBoolean},
    },
};

pub mod lut256_tables;
use crate::{chida::ChidaParty, util::ArithmeticBlackBox};
pub mod our_offline;
pub mod our_online;

pub struct LUT256SP<Recorder: VerificationRecorder, const MAL: bool> {
    inner: ChidaParty,
    context: BroadcastContext,
    prep_ohv: Vec<RndOhvPrep>,
    online_recorders: Vec<Recorder>,
    pub temp_vecs: Option<(AlignedVec<BsBool8, 64>, AlignedVec<BsBool8, 64>)>,
    lut_time: Duration,
}

impl<Recorder: VerificationRecorder, const MAL: bool> LUT256SP<Recorder, MAL> {
    pub fn time(&self) -> Duration {
        self.lut_time
    }

    pub fn setup(
        connected: ConnectedParty,
        n_worker_threads: Option<usize>,
        prot_str: Option<String>,
    ) -> MpcResult<Self> {
        ChidaParty::setup(connected, n_worker_threads, prot_str).map(|party| Self {
            inner: party,
            context: BroadcastContext::new(),
            prep_ohv: Vec::new(),
            online_recorders: vec![],
            temp_vecs: None,
            lut_time: Duration::from_secs(0),
        })
    }

    pub fn io(&self) -> &IoLayerOwned {
        <ChidaParty as ArithmeticBlackBox<BsBool8>>::io(&self.inner)
    }

    // Total size,
    pub fn compute_dim_sizes(
        mut input_bits: usize,
        output_bits: usize,
        block_bits: usize,
        num_dim: usize,
    ) -> Vec<usize> {
        match num_dim {
            3 => match input_bits % 3 {
                0 | 1 => vec![
                    // last_dim_bits,
                    input_bits / 3,
                    input_bits / 3,
                    input_bits / 3 + input_bits % 3,
                ],
                2 => vec![
                    // last_dim_bits,
                    input_bits / 3,
                    input_bits / 3 + 1,
                    input_bits / 3 + 1,
                ],
                _ => unreachable!(),
            },
            2 => vec![
                // last_dim_bits,
                input_bits / 2,
                input_bits / 2 + input_bits % 2,
            ],
            _ => panic!("Unsupported number of dimensions: {}", num_dim),
        }
    }
}

pub trait VerificationRecorder: Send
where
    Self: Sized,
{
    fn new() -> Self;
    fn generate_alpha(&mut self, party: &mut impl Party, n: usize) -> Vec<BsBool8>;
    fn record_ip_triple(
        &mut self,
        ai: &[BsBool8],
        aii: &[BsBool8],
        bi: &[BsBool8],
        bii: &[BsBool8],
        ci: &[BsBool8],
        shifts: &[usize],
        bits_per_block: usize,
        bits_per_lookup: usize,
    );
    fn record_ip_triple_cii(&mut self, cii: &[BsBool8]);
    fn record_mul_triple(
        &mut self,
        ai: &[BsBool8],
        aii: &[BsBool8],
        bi: &[BsBool8],
        bii: &[BsBool8],
        ci: &[BsBool8],
        cii: &[BsBool8],
    );
    fn finalize(
        recorders: Vec<Self>,
    ) -> (
        Vec<InnerProductTripleBoolean<BsBool8>>,
        Vec<ManyToOneMulTripleBoolean<BsBool8>>,
    );
}

pub struct NoVerificationRecording;

impl VerificationRecorder for NoVerificationRecording {
    fn new() -> Self {
        Self
    }

    fn generate_alpha(&mut self, party: &mut impl Party, n: usize) -> Vec<BsBool8> {
        party.generate_alpha(n).collect()
    }

    fn record_ip_triple(
        &mut self,
        _ai: &[BsBool8],
        _aii: &[BsBool8],
        _bi: &[BsBool8],
        _bii: &[BsBool8],
        _ci: &[BsBool8],
        _shifts: &[usize],
        _bits_per_block: usize,
        _bits_per_lookup: usize,
    ) {
    }

    fn record_ip_triple_cii(&mut self, _cii: &[BsBool8]) {}

    fn record_mul_triple(
        &mut self,
        _ai: &[BsBool8],
        _aii: &[BsBool8],
        _bi: &[BsBool8],
        _bii: &[BsBool8],
        _ci: &[BsBool8],
        _cii: &[BsBool8],
    ) {
    }

    fn finalize(
        _recorders: Vec<Self>,
    ) -> (
        Vec<InnerProductTripleBoolean<BsBool8>>,
        Vec<ManyToOneMulTripleBoolean<BsBool8>>,
    ) {
        (vec![], vec![])
    }
}

pub struct VerificationRecordVec {
    pub ip_triples: Vec<InnerProductTripleBoolean<BsBool8>>,
    pub mul_triples: Vec<ManyToOneMulTripleBoolean<BsBool8>>,
    pub pending_alpha: Option<(Vec<BsBool8>, Vec<BsBool8>)>,
}

impl VerificationRecorder for VerificationRecordVec {
    fn new() -> Self {
        Self {
            ip_triples: vec![],
            mul_triples: vec![],
            pending_alpha: None,
        }
    }

    fn generate_alpha(&mut self, party: &mut impl Party, n: usize) -> Vec<BsBool8> {
        let (ri, rii) = party.generate_random_raw(n);
        let alpha = zip_eq(&ri, &rii)
            .map(|(prev, next)| *next - *prev)
            .collect::<Vec<_>>();
        self.pending_alpha = Some((ri, rii));
        alpha
    }

    fn record_ip_triple(
        &mut self,
        ai: &[BsBool8],
        aii: &[BsBool8],
        bi: &[BsBool8],
        bii: &[BsBool8],
        ci: &[BsBool8],
        shifts: &[usize],
        bits_per_block: usize,
        bits_per_lookup: usize,
    ) {
        let (ri, rii) = self.pending_alpha.take().unwrap();
        self.ip_triples.push(InnerProductTripleBoolean {
            ai: ai.to_vec(),
            aii: aii.to_vec(),
            bi: bi.to_vec(),
            bii: bii.to_vec(),
            ci: ci.to_vec(),
            cii: vec![],
            ri,
            rii,
            shifts: shifts.to_vec(),
            bits_per_block,
            bits_per_lookup,
        })
    }

    fn record_ip_triple_cii(&mut self, cii: &[BsBool8]) {
        self.ip_triples.last_mut().unwrap().cii = cii.to_vec();
    }

    fn record_mul_triple(
        &mut self,
        ai: &[BsBool8],
        aii: &[BsBool8],
        bi: &[BsBool8],
        bii: &[BsBool8],
        ci: &[BsBool8],
        cii: &[BsBool8],
    ) {
        let (ri, rii) = self.pending_alpha.take().unwrap();
        self.mul_triples.push(ManyToOneMulTripleBoolean {
            ai: ai.to_vec(),
            aii: aii.to_vec(),
            bi: bi.to_vec(),
            bii: bii.to_vec(),
            ci: ci.to_vec(),
            cii: cii.to_vec(),
            ri,
            rii,
        })
    }

    fn finalize(
        recorders: Vec<Self>,
    ) -> (
        Vec<InnerProductTripleBoolean<BsBool8>>,
        Vec<ManyToOneMulTripleBoolean<BsBool8>>,
    ) {
        let mut ip_triples = vec![];
        let mut mul_triples = vec![];
        for recorder in recorders {
            let (ip, mul) = (recorder.ip_triples, recorder.mul_triples);
            ip_triples.extend(ip);
            mul_triples.extend(mul);
        }
        (ip_triples, mul_triples)
    }
}

pub struct RndOhvOutput {
    pub e_si: AlignedVec<BsBool8, 64>,
    pub e_sii: AlignedVec<BsBool8, 64>,
    pub dim_bits: usize,
}

pub struct RndOhvPrep {
    pub ohvs: Vec<RndOhvOutput>,
    pub prep_r_si: AlignedVec<BsBool8, 64>,
    pub prep_r_sii: AlignedVec<BsBool8, 64>,
}

fn alloc_aligned_blocks(size: usize) -> (AlignedVec<BsBool8, 64>, AlignedVec<BsBool8, 64>) {
    let e_si = vec::from_elem_in(BsBool8::ZERO, size, AlignedAllocator::<64>);
    let e_sii = vec::from_elem_in(BsBool8::ZERO, size, AlignedAllocator::<64>);
    (e_si, e_sii)
}

#[cfg(test)]
mod test {
    use std::marker::PhantomData;

    use crate::{
        lut_sp_boolean::VerificationRecorder,
        rep3_core::{
            network::ConnectedParty,
            test::{localhost_connect, TestSetup},
        },
    };

    use super::LUT256SP;

    pub struct LUT256Setup<R: VerificationRecorder, const MAL: bool>(PhantomData<R>);

    pub fn localhost_setup_lut_sp_boolean<
        R: VerificationRecorder,
        const MAL: bool,
        T1: Send,
        F1: Send + FnOnce(&mut LUT256SP<R, MAL>) -> T1,
        T2: Send,
        F2: Send + FnOnce(&mut LUT256SP<R, MAL>) -> T2,
        T3: Send,
        F3: Send + FnOnce(&mut LUT256SP<R, MAL>) -> T3,
    >(
        f1: F1,
        f2: F2,
        f3: F3,
        n_worker_threads: Option<usize>,
    ) -> (
        (T1, LUT256SP<R, MAL>),
        (T2, LUT256SP<R, MAL>),
        (T3, LUT256SP<R, MAL>),
    ) {
        fn adapter<
            T,
            R: VerificationRecorder,
            const MAL: bool,
            Fx: FnOnce(&mut LUT256SP<R, MAL>) -> T,
        >(
            conn: ConnectedParty,
            f: Fx,
            n_worker_threads: Option<usize>,
        ) -> (T, LUT256SP<R, MAL>) {
            let mut party = LUT256SP::setup(conn, n_worker_threads, None).unwrap();
            let t = f(&mut party);
            // party.finalize().unwrap();
            party.inner.teardown().unwrap();
            (t, party)
        }
        localhost_connect(
            move |conn_party| adapter(conn_party, f1, n_worker_threads),
            move |conn_party| adapter(conn_party, f2, n_worker_threads),
            move |conn_party| adapter(conn_party, f3, n_worker_threads),
        )
    }

    impl<R: VerificationRecorder, const MAL: bool> TestSetup<LUT256SP<R, MAL>> for LUT256Setup<R, MAL> {
        fn localhost_setup<
            T1: Send,
            F1: Send + FnOnce(&mut LUT256SP<R, MAL>) -> T1,
            T2: Send,
            F2: Send + FnOnce(&mut LUT256SP<R, MAL>) -> T2,
            T3: Send,
            F3: Send + FnOnce(&mut LUT256SP<R, MAL>) -> T3,
        >(
            f1: F1,
            f2: F2,
            f3: F3,
        ) -> (
            (T1, LUT256SP<R, MAL>),
            (T2, LUT256SP<R, MAL>),
            (T3, LUT256SP<R, MAL>),
        ) {
            localhost_setup_lut_sp_boolean(f1, f2, f3, None)
        }

        fn localhost_setup_multithreads<
            T1: Send,
            F1: Send + FnOnce(&mut LUT256SP<R, MAL>) -> T1,
            T2: Send,
            F2: Send + FnOnce(&mut LUT256SP<R, MAL>) -> T2,
            T3: Send,
            F3: Send + FnOnce(&mut LUT256SP<R, MAL>) -> T3,
        >(
            n_threads: usize,
            f1: F1,
            f2: F2,
            f3: F3,
        ) -> (
            (T1, LUT256SP<R, MAL>),
            (T2, LUT256SP<R, MAL>),
            (T3, LUT256SP<R, MAL>),
        ) {
            localhost_setup_lut_sp_boolean(f1, f2, f3, Some(n_threads))
        }
    }
}
