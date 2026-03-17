//! This module implements the MARLUT protocols.

use std::{marker::PhantomData, time::Duration, vec};

use itertools::{izip, zip_eq};

use crate::{
    rep3_core::{
        network::{ConnectedParty, task::IoLayerOwned},
        party::{Party, broadcast::BroadcastContext, error::MpcResult},
        share::HasZero,
    },
    share::{Field, bs_bool8::BsBool8},
    util::{
        aligned_vec::{AlignedAllocator, AlignedVec},
        mul_triple_vec::{InnerProductTriple, ManyToOneMulTriple},
    },
};

pub mod lut256_tables;
use crate::{chida::ChidaParty, util::ArithmeticBlackBox};
pub mod cut_bucket;
pub mod mae_pro10;
pub mod our_offline;
pub mod our_online;

pub struct LUT256SP<T: Field, Recorder: VerificationRecorder<T>, const MAL: bool> {
    pub inner: ChidaParty,
    pub context: BroadcastContext,
    prep_ohv: Vec<RndOhvPrep<T>>,
    online_recorders: Vec<Recorder>,
    lut_time: Duration,
    pub temp_vecs: Option<(AlignedVec<T, 64>, AlignedVec<T, 64>)>,
    _phantom: PhantomData<Recorder>,
}

impl<T: Field, Recorder: VerificationRecorder<T>, const MAL: bool> LUT256SP<T, Recorder, MAL> {
    pub fn time(&self) -> Duration {
        self.lut_time
    }

    pub fn setup(
        connected: ConnectedParty,
        n_worker_threads: Option<usize>,
        prot_str: Option<String>,
    ) -> MpcResult<Self> {
        ChidaParty::setup(connected, n_worker_threads, prot_str).and_then(|party| {
            Ok(Self {
                inner: party,
                context: BroadcastContext::new(),
                prep_ohv: Vec::new(),
                online_recorders: Vec::new(),
                lut_time: Duration::from_secs(0),
                temp_vecs: None,
                _phantom: PhantomData,
            })
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

pub trait VerificationRecorder<F: Field>: Send
where
    Self: Sized,
{
    fn new() -> Self;
    fn generate_alpha(&mut self, party: &mut impl Party, n: usize) -> Vec<F>;
    fn generate_alpha_aligned(&mut self, party: &mut impl Party, n: usize) -> AlignedVec<F, 64>;
    fn record_ip_triple(
        &mut self,
        ai: &[F],
        aii: &[F],
        bi: &[F],
        bii: &[F],
        ci: &[F],
        elems_per_block: usize,
        elems_per_lookup: usize,
    );
    fn record_ip_triple_cii(&mut self, cii: &[F]);
    fn record_mul_triple(&mut self, ai: &[F], aii: &[F], bi: &[F], bii: &[F], ci: &[F], cii: &[F]);
    fn record_zero_check(&mut self, ci: &[F], cii: &[F]);
    fn finalize(
        recorders: Vec<Self>,
    ) -> (
        Vec<InnerProductTriple<F>>,
        Vec<ManyToOneMulTriple<F>>,
        Vec<F>,
        Vec<F>,
    );
}

pub struct NoVerificationRecording;

impl<F: Field> VerificationRecorder<F> for NoVerificationRecording {
    fn new() -> Self {
        Self
    }

    fn generate_alpha(&mut self, party: &mut impl Party, n: usize) -> Vec<F> {
        party.generate_alpha(n).collect()
    }

    fn generate_alpha_aligned(&mut self, party: &mut impl Party, n: usize) -> AlignedVec<F, 64> {
        let mut result = Vec::with_capacity_in(n, AlignedAllocator::<64>);
        result.extend(party.generate_alpha::<F>(n));
        result
    }

    fn record_ip_triple(
        &mut self,
        _ai: &[F],
        _aii: &[F],
        _bi: &[F],
        _bii: &[F],
        _ci: &[F],
        _elems_per_block: usize,
        _elems_per_lookup: usize,
    ) {
    }

    fn record_ip_triple_cii(&mut self, _cii: &[F]) {}

    fn record_mul_triple(
        &mut self,
        _ai: &[F],
        _aii: &[F],
        _bi: &[F],
        _bii: &[F],
        _ci: &[F],
        _cii: &[F],
    ) {
    }

    fn record_zero_check(&mut self, _ci: &[F], _cii: &[F]) {}

    fn finalize(
        _recorders: Vec<Self>,
    ) -> (
        Vec<InnerProductTriple<F>>,
        Vec<ManyToOneMulTriple<F>>,
        Vec<F>,
        Vec<F>,
    ) {
        (vec![], vec![], vec![], vec![])
    }
}

pub struct VerificationRecordVec<F: Field> {
    pub ip_triples: Vec<InnerProductTriple<F>>,
    pub mul_triples: Vec<ManyToOneMulTriple<F>>,
    pub zero_check: (Vec<F>, Vec<F>),
    pub pending_alpha: Option<(AlignedVec<F, 64>, AlignedVec<F, 64>)>,
}

impl<F: Field> VerificationRecorder<F> for VerificationRecordVec<F> {
    fn new() -> Self {
        Self {
            ip_triples: vec![],
            mul_triples: vec![],
            zero_check: (vec![], vec![]),
            pending_alpha: None,
        }
    }

    fn generate_alpha(&mut self, party: &mut impl Party, n: usize) -> Vec<F> {
        let (ri, rii) = party.generate_random_raw_aligned(n);
        let alpha = zip_eq(&ri, &rii)
            .map(|(prev, next)| *next - *prev)
            .collect::<Vec<_>>();
        self.pending_alpha = Some((ri, rii));
        alpha
    }

    fn generate_alpha_aligned(&mut self, party: &mut impl Party, n: usize) -> AlignedVec<F, 64> {
        let (ri, rii) = party.generate_random_raw_aligned(n);
        let mut result = vec::from_elem_in(F::ZERO, n, AlignedAllocator::<64>);
        izip!(&ri, &rii, &mut result).for_each(|(prev, next, out)| *out = *next - *prev);
        self.pending_alpha = Some((ri, rii));
        result
    }

    fn record_ip_triple(
        &mut self,
        in_ai: &[F],
        in_aii: &[F],
        in_bi: &[F],
        in_bii: &[F],
        in_ci: &[F],
        elems_per_block: usize,
        elems_per_lookup: usize,
    ) {
        let (ri, rii) = self.pending_alpha.take().unwrap();
        let mut ai = vec::from_elem_in(F::ZERO, in_ai.len(), AlignedAllocator::<64>);
        ai.copy_from_slice(in_ai);
        let mut aii = vec::from_elem_in(F::ZERO, in_aii.len(), AlignedAllocator::<64>);
        aii.copy_from_slice(in_aii);
        let mut bi = vec::from_elem_in(F::ZERO, in_bi.len(), AlignedAllocator::<64>);
        bi.copy_from_slice(in_bi);
        let mut bii = vec::from_elem_in(F::ZERO, in_bii.len(), AlignedAllocator::<64>);
        bii.copy_from_slice(in_bii);
        let mut ci = vec::from_elem_in(F::ZERO, in_ci.len(), AlignedAllocator::<64>);
        ci.copy_from_slice(in_ci);
        self.ip_triples.push(InnerProductTriple {
            ai,
            aii,
            bi,
            bii,
            ci,
            cii: vec::from_elem_in(F::ZERO, 0, AlignedAllocator::<64>),
            ri,
            rii,
            elems_per_block,
            elems_per_lookup,
        })
    }

    fn record_ip_triple_cii(&mut self, cii: &[F]) {
        let last_cii = &mut self.ip_triples.last_mut().unwrap().cii;
        *last_cii = vec::from_elem_in(F::ZERO, cii.len(), AlignedAllocator::<64>);
        last_cii.copy_from_slice(cii);
    }

    fn record_mul_triple(&mut self, in_ai: &[F], in_aii: &[F], in_bi: &[F], in_bii: &[F], in_ci: &[F], in_cii: &[F]) {
        let (ri, rii) = self.pending_alpha.take().unwrap();
        let mut ai = vec::from_elem_in(F::ZERO, in_ai.len(), AlignedAllocator::<64>);
        ai.copy_from_slice(in_ai);
        let mut aii = vec::from_elem_in(F::ZERO, in_aii.len(), AlignedAllocator::<64>);
        aii.copy_from_slice(in_aii);
        let mut bi = vec::from_elem_in(F::ZERO, in_bi.len(), AlignedAllocator::<64>);
        bi.copy_from_slice(in_bi);
        let mut bii = vec::from_elem_in(F::ZERO, in_bii.len(), AlignedAllocator::<64>);
        bii.copy_from_slice(in_bii);
        let mut ci = vec::from_elem_in(F::ZERO, in_ci.len(), AlignedAllocator::<64>);
        ci.copy_from_slice(in_ci);
        let mut cii = vec::from_elem_in(F::ZERO, in_cii.len(), AlignedAllocator::<64>);
        cii.copy_from_slice(in_cii);
        self.mul_triples.push(ManyToOneMulTriple {
            ai,
            aii,
            bi,
            bii,
            ci,
            cii,
            ri,
            rii,
        })
    }

    fn record_zero_check(&mut self, ci: &[F], cii: &[F]) {
        self.zero_check.0.extend(ci);
        self.zero_check.1.extend(cii);
    }

    fn finalize(
        recorders: Vec<Self>,
    ) -> (
        Vec<InnerProductTriple<F>>,
        Vec<ManyToOneMulTriple<F>>,
        Vec<F>,
        Vec<F>,
    ) {
        let mut ip_triples = vec![];
        let mut mul_triples = vec![];
        let mut zero_si = vec![];
        let mut zero_sii = vec![];
        for recorder in recorders {
            let (ip, mul, (zi, zii)) = (
                recorder.ip_triples,
                recorder.mul_triples,
                recorder.zero_check,
            );
            ip_triples.extend(ip);
            mul_triples.extend(mul);
            zero_si.extend(zi);
            zero_sii.extend(zii);
        }
        (ip_triples, mul_triples, zero_si, zero_sii)
    }
}

pub struct RndOhvOutput<T> {
    pub e_si: AlignedVec<T, 64>,
    pub e_sii: AlignedVec<T, 64>,
    pub dim_bits: usize,
}

pub struct RndOhvPrep<T> {
    pub ohvs: Vec<RndOhvOutput<T>>,
    pub prep_r_si: Vec<T>,
    pub prep_r_sii: Vec<T>,
}

fn alloc_aligned_blocks<T: HasZero + Clone>(size: usize) -> (AlignedVec<T, 64>, AlignedVec<T, 64>) {
    let e_si = vec::from_elem_in(T::ZERO, size, AlignedAllocator::<64>);
    let e_sii = vec::from_elem_in(T::ZERO, size, AlignedAllocator::<64>);
    (e_si, e_sii)
}

#[cfg(test)]
mod test {
    use std::marker::PhantomData;

    use crate::{
        lut_sp::VerificationRecorder,
        rep3_core::{
            network::ConnectedParty,
            test::{TestSetup, localhost_connect},
        },
        share::Field,
    };

    use super::LUT256SP;

    pub struct LUT256Setup<T: Field, Recorder: VerificationRecorder<T>, const MAL: bool>(
        (PhantomData<T>, PhantomData<Recorder>),
    );

    pub fn localhost_setup_lut_sp<
        U: Field,
        R: VerificationRecorder<U>,
        const MAL: bool,
        T1: Send,
        F1: Send + FnOnce(&mut LUT256SP<U, R, MAL>) -> T1,
        T2: Send,
        F2: Send + FnOnce(&mut LUT256SP<U, R, MAL>) -> T2,
        T3: Send,
        F3: Send + FnOnce(&mut LUT256SP<U, R, MAL>) -> T3,
    >(
        f1: F1,
        f2: F2,
        f3: F3,
        n_worker_threads: Option<usize>,
    ) -> (
        (T1, LUT256SP<U, R, MAL>),
        (T2, LUT256SP<U, R, MAL>),
        (T3, LUT256SP<U, R, MAL>),
    ) {
        fn adapter<
            U: Field,
            R: VerificationRecorder<U>,
            const MAL: bool,
            T,
            Fx: FnOnce(&mut LUT256SP<U, R, MAL>) -> T,
        >(
            conn: ConnectedParty,
            f: Fx,
            n_worker_threads: Option<usize>,
        ) -> (T, LUT256SP<U, R, MAL>) {
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

    impl<T: Field, R: VerificationRecorder<T>, const MAL: bool> TestSetup<LUT256SP<T, R, MAL>>
        for LUT256Setup<T, R, MAL>
    {
        fn localhost_setup<
            T1: Send,
            F1: Send + FnOnce(&mut LUT256SP<T, R, MAL>) -> T1,
            T2: Send,
            F2: Send + FnOnce(&mut LUT256SP<T, R, MAL>) -> T2,
            T3: Send,
            F3: Send + FnOnce(&mut LUT256SP<T, R, MAL>) -> T3,
        >(
            f1: F1,
            f2: F2,
            f3: F3,
        ) -> (
            (T1, LUT256SP<T, R, MAL>),
            (T2, LUT256SP<T, R, MAL>),
            (T3, LUT256SP<T, R, MAL>),
        ) {
            localhost_setup_lut_sp(f1, f2, f3, None)
        }

        fn localhost_setup_multithreads<
            T1: Send,
            F1: Send + FnOnce(&mut LUT256SP<T, R, MAL>) -> T1,
            T2: Send,
            F2: Send + FnOnce(&mut LUT256SP<T, R, MAL>) -> T2,
            T3: Send,
            F3: Send + FnOnce(&mut LUT256SP<T, R, MAL>) -> T3,
        >(
            n_threads: usize,
            f1: F1,
            f2: F2,
            f3: F3,
        ) -> (
            (T1, LUT256SP<T, R, MAL>),
            (T2, LUT256SP<T, R, MAL>),
            (T3, LUT256SP<T, R, MAL>),
        ) {
            localhost_setup_lut_sp(f1, f2, f3, Some(n_threads))
        }
    }
}
