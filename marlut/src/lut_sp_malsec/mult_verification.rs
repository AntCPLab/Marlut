///! This module implements the DZKP protocol of MARLUT.
///  It is insanely long because the protocol itself is quite complicated,
///  plus all kinds of SIMD optimizations.

use std::fmt::Debug;
use std::iter::zip;
use std::ops::{Add, Mul, MulAssign, Sub};
use std::time::Instant;
use std::{ptr, slice, vec};

use crate::lut_sp::our_online::LUT256SPMalTable;
use crate::rep3_core::party::correlated_randomness::GlobalRng;
use crate::rep3_core::share::Lift;
use crate::share::{Field, InnerProduct};
use crate::share::{FieldLike, LagrangeInterExtrapolate, PrimeField};
use crate::util::aligned_vec::{AlignedAllocator, AlignedVec};
use crate::util::mul_triple_vec::ManyToOneMulTriple;
use crate::{
    rep3_core::{
        network::task::Direction,
        party::{
            DigestExt, MainParty, Party,
            broadcast::{Broadcast, BroadcastContext},
            error::MpcResult,
        },
        share::{EmptyRssShare, HasZero, RssShare, RssShareGeneral, RssShareVec},
    },
    share::Empty,
    util::mul_triple_vec::InnerProductTriple,
};
use itertools::{Itertools, izip, zip_eq};
use rand::SeedableRng;
use rand::{CryptoRng, Rng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

macro_rules! ordered {
    ($index:expr, $self_ins:expr, $next_ins:expr, $prev_ins:expr) => {
        match $index {
            0 => {
                let self_ret = $self_ins;
                let next_ret = $next_ins;
                let prev_ret = $prev_ins;
                (self_ret, next_ret, prev_ret)
            }
            1 => {
                let prev_ret = $prev_ins;
                let self_ret = $self_ins;
                let next_ret = $next_ins;
                (self_ret, next_ret, prev_ret)
            }
            2 => {
                let next_ret = $next_ins;
                let prev_ret = $prev_ins;
                let self_ret = $self_ins;
                (self_ret, next_ret, prev_ret)
            }
            3.. => panic!("Only 3PC is supported"),
        }
    };
}

pub struct DummyMalTable;

impl<F: Field> LUT256SPMalTable<F> for DummyMalTable {
    fn process_inner_product_triple_self(
        _triple: &InnerProductTriple<F>,
        _gammas: &[u64],
        _coeff: &[u8],
        _x1: &mut [RssShareGeneral<Empty, F>],
        _x2: &mut [RssShareGeneral<F, Empty>],
    ) -> RssShare<F> {
        unimplemented!();
    }

    fn process_inner_product_triple_next(
        _triple: &InnerProductTriple<F>,
        _gammas: &[u64],
        _coeff: &[u8],
        _x2: &mut [RssShareGeneral<Empty, F>],
    ) -> RssShareGeneral<Empty, F> {
        unimplemented!();
    }

    fn process_inner_product_triple_prev(
        _triple: &InnerProductTriple<F>,
        _gammas: &[u64],
        _coeff: &[u8],
        _x1: &mut [RssShareGeneral<F, Empty>],
    ) -> RssShareGeneral<F, Empty> {
        unimplemented!();
    }
}

/// This protocol checks that
/// (1) the many-to-one multiplication tuples are correctly computed;
/// (2) the inner product tuples are correctly computed;
/// (3) the shared values claimed to be binary (0 or 1) are actually binary.
pub fn verify_multiplication_triples<
    const DEGREE: usize,
    F: Field + DigestExt + Debug + Lift<Fp, Output = Fp>,
    Fp: PrimeField + InnerProduct + Debug + LagrangeInterExtrapolate<DEGREE>,
    Table: LUT256SPMalTable<F>,
>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    triples: &[InnerProductTriple<F>],
    mul_triples: &[ManyToOneMulTriple<F>],
    kappa: usize,
) -> MpcResult<bool> {
    // Collect original inner-product instances

    let mut global_rng = GlobalRng::setup_global(party)?;

    // Batch the instances. Beware that global_rng is mutated, so each party must process the instances in the same order.
    let (mut self_global_rng, mut next_global_rng, mut prev_global_rng) = ordered!(
        party.i,
        ChaCha20Rng::from_rng(global_rng.as_mut()).unwrap(),
        ChaCha20Rng::from_rng(global_rng.as_mut()).unwrap(),
        ChaCha20Rng::from_rng(global_rng.as_mut()).unwrap()
    );

    let t_len = kappa * (Fp::NBITS - F::NBITS);
    let next_t_share = party
        .io()
        .receive_field::<RssShareGeneral<Empty, Fp>>(Direction::Next, t_len);
    let prev_t_share = party
        .io()
        .receive_field::<RssShareGeneral<Fp, Empty>>(Direction::Previous, t_len);

    let start = Instant::now();
    // t_i shared according to (t_{i, 1}, 0, t_{i, 2})
    // Self owns (t_{i2}, t_{i1}), Next owns (t_{i1}, 0), Prev owns (0, t_{i2})
    // New instance is x = (2 t_{i, 1}, 0, 0), y = (0, 0, t_{i, 2}), z = (z_1, 0, z_2)
    // where z_1 =  (t_{i, 1}^2 - t_{i, 1}) , z_2 = (t_{i, 2}^2 - t_{i, 2})
    let (self_instance, (next_instance, prev_instance)) = rayon::join(
        || {
            let self_instance = if triples.is_empty() {
                get_self_mul_tuples(mul_triples, &mut self_global_rng, kappa)
            } else if mul_triples.is_empty() {
                get_self_inner_product_tuples::<_, Table>(triples, &mut self_global_rng, kappa)
            } else {
                combine_instances(
                    get_self_inner_product_tuples::<_, Table>(triples, &mut self_global_rng, kappa),
                    get_self_mul_tuples(mul_triples, &mut self_global_rng, kappa),
                )
            };

            let self_t = prove_lift_instances::<F, Fp>(&self_instance);

            let (t_s1, t_s2, t_s3) = secret_share_two_party(&mut party.random_local, &self_t);
            party.io().send_field_slice(Direction::Next, &t_s2);
            party.io().send_field_slice(Direction::Previous, &t_s3);

            let t_instance = binary_shares_to_self_instance(&t_s1);

            let self_t = t_s1
                .par_chunks_exact(Fp::NBITS - F::NBITS)
                .map(|t_shares| {
                    reconstruct_t::<F, Fp, _, _>(t_shares)
                        - RssShare::from(
                            Fp::from(1 << (F::NBITS - 1)),
                            Fp::from(1 << (F::NBITS - 1)),
                        )
                })
                .collect::<Vec<_>>();

            (self_instance, self_t, t_instance)
        },
        || {
            rayon::join(
                || {
                    let next_instance = if triples.is_empty() {
                        get_next_mul_tuples(mul_triples, &mut next_global_rng, kappa)
                    } else if mul_triples.is_empty() {
                        get_next_inner_product_tuples::<_, Table>(
                            triples,
                            &mut next_global_rng,
                            kappa,
                        )
                    } else {
                        combine_instances(
                            get_next_inner_product_tuples::<_, Table>(
                                triples,
                                &mut next_global_rng,
                                kappa,
                            ),
                            get_next_mul_tuples(mul_triples, &mut next_global_rng, kappa),
                        )
                    };
                    let next_t = next_t_share.rcv().unwrap();

                    let t_instance = binary_shares_to_next_instance(&next_t);

                    let next_t = next_t
                        .par_chunks_exact(Fp::NBITS - F::NBITS)
                        .map(|t_shares| {
                            reconstruct_t::<F, Fp, _, _>(t_shares)
                                - RssShareGeneral::from(Empty, Fp::from(1 << (F::NBITS - 1)))
                        })
                        .collect::<Vec<_>>();

                    (next_instance, next_t, t_instance)
                },
                || {
                    let prev_instance = if triples.is_empty() {
                        get_prev_mul_tuples(mul_triples, &mut prev_global_rng, kappa)
                    } else if mul_triples.is_empty() {
                        get_prev_inner_product_tuples::<_, Table>(
                            triples,
                            &mut prev_global_rng,
                            kappa,
                        )
                    } else {
                        combine_instances(
                            get_prev_inner_product_tuples::<_, Table>(
                                triples,
                                &mut prev_global_rng,
                                kappa,
                            ),
                            get_prev_mul_tuples(mul_triples, &mut prev_global_rng, kappa),
                        )
                    };
                    let prev_t = prev_t_share.rcv().unwrap();

                    let t_instance = binary_shares_to_prev_instance(&prev_t);

                    let prev_t = prev_t
                        .par_chunks_exact(Fp::NBITS - F::NBITS)
                        .map(|t_shares| {
                            reconstruct_t::<F, Fp, _, _>(t_shares)
                                - RssShareGeneral::from(Fp::from(1 << (F::NBITS - 1)), Empty)
                        })
                        .collect::<Vec<_>>();

                    (prev_instance, prev_t, t_instance)
                },
            )
        },
    );

    println!("{} prepare instances: {:?}", party.i, start.elapsed());

    // Lift the instances

    let r: Fp = coin_flip(party, context)?;

    let start = Instant::now();
    let (self_instance, (next_instance, prev_instance)) = rayon::join(
        || batch_fp_instances(&self_instance, r),
        || {
            rayon::join(
                || batch_fp_instances(&next_instance, r),
                || batch_fp_instances(&prev_instance, r),
            )
        },
    );
    println!("{} batch instances: {:?}", party.i, start.elapsed());

    let start = Instant::now();
    println!(
        "Instance size: {} {}",
        self_instance.0.0.len(),
        self_instance.0.1.len()
    );
    let next_instance = (next_instance.0, next_instance.1, next_instance.2.into());
    let prev_instance = (prev_instance.0, prev_instance.1, prev_instance.2.into());
    let result =
        verify_dot_product_batched(party, context, self_instance, next_instance, prev_instance)?;

    println!("{} dot products: {:?}", party.i, start.elapsed());

    Ok(result)
}

fn linear_combination<F: Field>(a: &[F], b: &[u8]) -> Vec<F> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
    if F::NBYTES == 1 {
        return linear_combination_8(a, b);
    } else if F::NBYTES == 2 {
        return linear_combination_16(a, b);
    }

    let stride = a.len() / b.len();
    let mut out = vec![F::ZERO; stride];
    for j in 0..b.len() {
        if b[j] != 0 {
            for i in 0..stride {
                out[i] += a[j * stride + i];
            }
        }
    }
    out
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
fn linear_combination_8<F: Field>(a: &[F], b: &[u8]) -> Vec<F> {
    let stride = a.len() / b.len();
    let mut out = vec![F::ZERO; stride];

    let mut a_ptr = a as *const [F] as *const i8;
    let mut out_ptr = &mut out[..] as *mut [F] as *mut i8;
    for _ in (0..stride).step_by(64) {
        unsafe {
            use std::arch::x86_64::{
                _mm512_add_epi8, _mm512_loadu_epi8, _mm512_setzero_si512, _mm512_storeu_epi8,
            };
            let mut val = _mm512_setzero_si512();
            for j in 0..b.len() {
                if b[j] != 0 {
                    let a = _mm512_loadu_epi8(a_ptr.offset((j * stride) as isize));
                    val = _mm512_add_epi8(val, a);
                }
            }
            _mm512_storeu_epi8(out_ptr, val);
            a_ptr = a_ptr.offset(64);
            out_ptr = out_ptr.offset(64);
        }
    }
    let start = stride / 64 * 64;
    if start == stride {
        return out;
    }
    for j in 0..b.len() {
        if b[j] != 0 {
            for i in start..stride {
                out[i] += a[j * stride + i];
            }
        }
    }
    out
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
fn linear_combination_16<F: Field>(a: &[F], b: &[u8]) -> Vec<F> {
    let stride = a.len() / b.len();
    let mut out = vec![F::ZERO; stride];

    let mut a_ptr = a as *const [F] as *const i16;
    let mut out_ptr = &mut out[..] as *mut [F] as *mut i16;
    for _ in (0..stride).step_by(32) {
        unsafe {
            use std::arch::x86_64::{
                _mm512_add_epi16, _mm512_loadu_epi16, _mm512_setzero_si512, _mm512_storeu_epi16,
            };
            let mut val = _mm512_setzero_si512();
            for j in 0..b.len() {
                if b[j] != 0 {
                    let a = _mm512_loadu_epi16(a_ptr.offset((j * stride) as isize));
                    val = _mm512_add_epi16(val, a);
                }
            }
            _mm512_storeu_epi16(out_ptr, val);
            a_ptr = a_ptr.offset(32);
            out_ptr = out_ptr.offset(32);
        }
    }
    let start = stride / 32 * 32;
    if start == stride {
        return out;
    }
    for j in 0..b.len() {
        if b[j] != 0 {
            for i in start..stride {
                out[i] += a[j * stride + i];
            }
        }
    }
    out
}

fn linear_combination_alt<F: Field, const ELEMS_PER_LOOKUP: usize, const ELEMS_PER_BLOCK: usize>(
    out: &mut [F; ELEMS_PER_LOOKUP / ELEMS_PER_BLOCK],
    a: &[F; ELEMS_PER_LOOKUP],
    b: &[u8; ELEMS_PER_BLOCK],
) {
    for j in 0..b.len() {
        if b[j] != 0 {
            for i in 0..out.len() {
                out[i] += a[i * b.len() + j];
            }
        }
    }
}

fn linear_combination_single<F: Field>(a: &[F], b: &[u8]) -> F {
    let mut out = F::ZERO;
    for i in 0..a.len() {
        if b[i] != 0 {
            out += a[i];
        }
    }
    out
}

fn gen_rand_bools(rng: &mut (impl Rng + CryptoRng), n: usize) -> Vec<u8> {
    let mut data = vec![0; n];
    rng.fill_bytes(&mut data);
    for x in &mut data {
        *x = *x & 1;
    }
    data
}

fn gen_rand_u64s(rng: &mut (impl Rng + CryptoRng), n: usize) -> Vec<u64> {
    let mut data = vec![0; n];
    unsafe {
        rng.fill_bytes(slice::from_raw_parts_mut(
            &mut data[..] as *mut [u64] as *mut u8,
            n * 8,
        ));
    }
    data
}

fn mul_bool<F: Field>(x: F, y: u8) -> F {
    if y == 0 { F::ZERO } else { x }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
fn conditional_fill_8<F: Field, T1, U1>(
    out: &mut [RssShareGeneral<T1, U1>],
    source: &[F],
    gammas: &[u64],
) {
    debug_assert_eq!(std::mem::size_of::<RssShareGeneral<T1, U1>>(), 1);
    debug_assert_eq!(std::mem::size_of::<F>(), 1);
    debug_assert_eq!(std::mem::align_of::<RssShareGeneral<T1, U1>>(), 1);
    debug_assert_eq!(std::mem::align_of::<F>(), 1);
    let mut source_ptr = source as *const [F] as *const i8;
    let mut out_ptr = out as *mut [RssShareGeneral<T1, U1>] as *mut i8;
    unsafe {
        for gamma in gammas.iter() {
            use std::arch::x86_64::{_cvtu64_mask64, _mm512_maskz_loadu_epi8, _mm512_storeu_epi8};

            let mask = _cvtu64_mask64(*gamma);
            let data = _mm512_maskz_loadu_epi8(mask, source_ptr);
            _mm512_storeu_epi8(out_ptr, data);
            source_ptr = source_ptr.offset(64);
            out_ptr = out_ptr.offset(64);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
fn conditional_fill_16<F: Field, T1, U1>(
    out: &mut [RssShareGeneral<T1, U1>],
    source: &[F],
    gammas: &[u64],
) {
    debug_assert_eq!(std::mem::size_of::<RssShareGeneral<T1, U1>>(), 2);
    debug_assert_eq!(std::mem::size_of::<F>(), 2);
    debug_assert_eq!(std::mem::align_of::<RssShareGeneral<T1, U1>>(), 2);
    debug_assert_eq!(std::mem::align_of::<F>(), 2);
    let mut source_ptr = source as *const [F] as *const i16;
    let mut out_ptr = out as *mut [RssShareGeneral<T1, U1>] as *mut i16;
    unsafe {
        for gamma in gammas.iter() {
            use std::arch::x86_64::{
                _cvtu32_mask32, _mm512_maskz_loadu_epi16, _mm512_storeu_epi16,
            };

            let mask_low = _cvtu32_mask32(*gamma as u32);
            let mask_high = _cvtu32_mask32((*gamma >> 32) as u32);
            let data = _mm512_maskz_loadu_epi16(mask_low, source_ptr);
            _mm512_storeu_epi16(out_ptr, data);
            source_ptr = source_ptr.offset(32);
            out_ptr = out_ptr.offset(32);

            let data = _mm512_maskz_loadu_epi16(mask_high, source_ptr);
            _mm512_storeu_epi16(out_ptr, data);
            source_ptr = source_ptr.offset(32);
            out_ptr = out_ptr.offset(32);
        }
    }
}

fn conditional_fill<F: Field, T1, U1>(
    out: &mut [RssShareGeneral<T1, U1>],
    source: &[F],
    gamma: &[u64],
) {
    let mut start = 0;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
    if F::NBYTES == 1 {
        let gamma_len = source.len() / 64;
        start = gamma_len * 64;
        conditional_fill_8(&mut out[..start], &source[..start], &gamma[..gamma_len]);
    } else if F::NBYTES == 2 {
        let gamma_len = source.len() / 64;
        start = gamma_len * 64;
        conditional_fill_16(&mut out[..start], &source[..start], &gamma[..gamma_len]);
    }

    for i in start..source.len() {
        if gamma[i / 64] & (1 << (i % 64)) != 0 {
            unsafe {
                out[i] = std::mem::transmute_copy(&source[i]);
            }
        }
    }
}

// fn conditional_fill<F: Field, T1: Send, U1: Send>(
//     out: &mut [RssShareGeneral<T1, U1>],
//     source: &[F],
//     gamma: &[u64],
// ) {
//     if out.len() >= 1 << 13 {
//         let gamma_chunk_size =
//             (gamma.len() + rayon::current_num_threads() - 1) / rayon::current_num_threads();
//         let chunk_size = gamma_chunk_size * 64;
//         (
//             out.par_chunks_mut(chunk_size),
//             source.par_chunks(chunk_size),
//             gamma.par_chunks(gamma_chunk_size),
//         )
//             .into_par_iter()
//             .for_each(|(out, source, gamma)| conditional_fill_serial(out, source, gamma));
//     } else {
//         conditional_fill_serial(out, source, gamma)
//     }
// }

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
fn self_mul_compute_8<F: Field>(
    ci: &[F],
    ai: &[F],
    bi: &[F],
    ri: &[F],
    rii: &[F],
    gammas: &[u64],
) -> RssShare<F> {
    debug_assert_eq!(std::mem::size_of::<F>(), 1);
    debug_assert!(ci.len() % 64 == 0);

    let mut ci_ptr = ci as *const [F] as *const i8;
    let mut ai_ptr = ai as *const [F] as *const i8;
    let mut bi_ptr = bi as *const [F] as *const i8;
    let mut ri_ptr = ri as *const [F] as *const i8;
    let mut rii_ptr = rii as *const [F] as *const i8;
    unsafe {
        use std::arch::x86_64::{
            _cvtu32_mask32, _mm256_add_epi8, _mm256_maskz_loadu_epi8, _mm256_reduce_add_epi8,
            _mm256_setzero_si256, _mm256_sub_epi8, _mm512_cvtepi16_epi8, _mm512_cvtepu8_epi16,
            _mm512_mullo_epi16,
        };

        let mut sum_si = _mm256_setzero_si256();
        let mut sum_sii = _mm256_setzero_si256();
        for gamma in gammas.iter() {
            let mask_low = _cvtu32_mask32(*gamma as u32);
            let mask_high = _cvtu32_mask32((*gamma >> 32) as u32);

            for mask in [mask_low, mask_high] {
                let ci = _mm256_maskz_loadu_epi8(mask, ci_ptr);
                let ai = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, ai_ptr));
                let bi = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, bi_ptr));
                let ri = _mm256_maskz_loadu_epi8(mask, ri_ptr);
                let rii = _mm256_maskz_loadu_epi8(mask, rii_ptr);

                // ci + ri - (ai * bi)
                let ai_times_bi = _mm512_cvtepi16_epi8(_mm512_mullo_epi16(ai, bi));
                sum_si = _mm256_sub_epi8(
                    _mm256_add_epi8(_mm256_add_epi8(sum_si, ci), ri),
                    ai_times_bi,
                );
                sum_sii = _mm256_sub_epi8(sum_sii, rii);

                ci_ptr = ci_ptr.offset(32);
                ai_ptr = ai_ptr.offset(32);
                bi_ptr = bi_ptr.offset(32);
                ri_ptr = ri_ptr.offset(32);
                rii_ptr = rii_ptr.offset(32);
            }
        }
        RssShare::from(
            std::mem::transmute_copy(&_mm256_reduce_add_epi8(sum_si)),
            std::mem::transmute_copy(&_mm256_reduce_add_epi8(sum_sii)),
        )
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
fn mm512_reduce_add_epi16(x: std::arch::x86_64::__m512i) -> u16 {
    unsafe {
        use std::arch::x86_64::{
            _mm256_reduce_add_epi16, _mm512_castsi512_si256, _mm512_extracti64x4_epi64,
        };

        let low = _mm512_castsi512_si256(x);
        let high = _mm512_extracti64x4_epi64::<1>(x);
        (_mm256_reduce_add_epi16(low) as u16).wrapping_add(_mm256_reduce_add_epi16(high) as u16)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
fn self_mul_compute_16<F: Field>(
    ci: &[F],
    ai: &[F],
    bi: &[F],
    ri: &[F],
    rii: &[F],
    gammas: &[u64],
) -> RssShare<F> {
    debug_assert_eq!(std::mem::size_of::<F>(), 2);
    debug_assert!(ci.len() % 64 == 0);

    let mut ci_ptr = ci as *const [F] as *const i16;
    let mut ai_ptr = ai as *const [F] as *const i16;
    let mut bi_ptr = bi as *const [F] as *const i16;
    let mut ri_ptr = ri as *const [F] as *const i16;
    let mut rii_ptr = rii as *const [F] as *const i16;
    unsafe {
        use std::arch::x86_64::{
            _cvtu32_mask32, _mm512_add_epi16, _mm512_maskz_loadu_epi16, _mm512_mullo_epi16,
            _mm512_setzero_si512, _mm512_sub_epi16,
        };

        let mut sum_si = _mm512_setzero_si512();
        let mut sum_sii = _mm512_setzero_si512();
        for gamma in gammas.iter() {
            let mask_low = _cvtu32_mask32(*gamma as u32);
            let mask_high = _cvtu32_mask32((*gamma >> 32) as u32);

            for mask in [mask_low, mask_high] {
                let ci = _mm512_maskz_loadu_epi16(mask, ci_ptr);
                let ai = _mm512_maskz_loadu_epi16(mask, ai_ptr);
                let bi = _mm512_maskz_loadu_epi16(mask, bi_ptr);
                let ri = _mm512_maskz_loadu_epi16(mask, ri_ptr);
                let rii = _mm512_maskz_loadu_epi16(mask, rii_ptr);

                // ci + ri - (ai * bi)
                let ai_times_bi = _mm512_mullo_epi16(ai, bi);
                sum_si = _mm512_sub_epi16(
                    _mm512_add_epi16(_mm512_add_epi16(sum_si, ci), ri),
                    ai_times_bi,
                );
                sum_sii = _mm512_sub_epi16(sum_sii, rii);

                ci_ptr = ci_ptr.offset(32);
                ai_ptr = ai_ptr.offset(32);
                bi_ptr = bi_ptr.offset(32);
                ri_ptr = ri_ptr.offset(32);
                rii_ptr = rii_ptr.offset(32);
            }
        }

        RssShare::from(
            std::mem::transmute_copy(&mm512_reduce_add_epi16(sum_si)),
            std::mem::transmute_copy(&mm512_reduce_add_epi16(sum_sii)),
        )
    }
}

fn self_mul_compute<F: Field>(
    ci: &[F],
    ai: &[F],
    bi: &[F],
    ri: &[F],
    rii: &[F],
    gammas: &[u64],
) -> RssShare<F> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
    if std::mem::size_of::<F>() == 1 && ci.len() % 64 == 0 {
        return self_mul_compute_8(ci, ai, bi, ri, rii, gammas);
    } else if std::mem::size_of::<F>() == 2 && ci.len() % 64 == 0 {
        return self_mul_compute_16(ci, ai, bi, ri, rii, gammas);
    }

    izip!(ci, ai, bi, ri, rii)
        .enumerate()
        .map(|(i, (ci, xi, bi, ri, rii))| {
            if gammas[i / 64] & (1 << (i % 64)) != 0 {
                RssShare::from(*ci - *xi * *bi + *ri, -*rii)
            } else {
                RssShare::ZERO
            }
        })
        .sum()
}

fn get_self_mul_tuples<F: Field + Debug>(
    triples: &[ManyToOneMulTriple<F>],
    global_rng: &mut (impl Rng + CryptoRng),
    kappa: usize,
) -> (
    (
        Vec<AlignedVec<RssShareGeneral<Empty, F>, 64>>,
        Vec<AlignedVec<RssShareGeneral<F, Empty>, 64>>,
    ),
    (
        AlignedVec<RssShareGeneral<F, Empty>, 64>,
        AlignedVec<RssShareGeneral<Empty, F>, 64>,
    ),
    Vec<RssShare<F>>,
) {
    let mut triple_indices = vec![0; triples.len()];
    let mut offset = 0;
    for (i, triple) in triples.iter().enumerate() {
        triple_indices[i] = offset;
        offset += triple.bi.len();
    }

    let total_len = offset;

    let mut y1 = vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
    let mut y2 = vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
    let y1_ptr = &mut y1[..] as *mut [_] as *mut RssShareGeneral<F, Empty> as usize;
    let y2_ptr = &mut y2[..] as *mut [_] as *mut RssShareGeneral<Empty, F> as usize;
    (triples, &triple_indices)
        .into_par_iter()
        .for_each(|(triple, index)| unsafe {
            let y1_ptr = (y1_ptr as *mut RssShareGeneral<F, Empty>).offset(*index as isize);
            let y2_ptr = (y2_ptr as *mut RssShareGeneral<Empty, F>).offset(*index as isize);
            ptr::copy_nonoverlapping(
                &triple.bi as &[_] as *const [_] as *const F,
                y1_ptr as *mut F,
                triple.bi.len(),
            );
            ptr::copy_nonoverlapping(
                &triple.bii as &[_] as *const [_] as *const F,
                y2_ptr as *mut F,
                triple.bii.len(),
            );
        });

    let mut rngs = (0..kappa)
        .map(|_| ChaCha20Rng::from_rng(&mut *global_rng).unwrap())
        .collect::<Vec<_>>();
    let (x, z): (Vec<_>, Vec<_>) = rngs
        .par_iter_mut()
        .map(|global_rng| {
            let mut x1 =
                vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
            let mut x2 =
                vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
            let (gammas, coeffs): (Vec<_>, Vec<_>) = triples
                .iter()
                .map(|triple| {
                    (
                        gen_rand_u64s(global_rng, (triple.bi.len() + 63) / 64),
                        gen_rand_bools(global_rng, triple.ai.len() / triple.bi.len()),
                    )
                })
                .unzip();

            // This is safe because each thread accesses non-overlapping portions of the original slice
            let x1_ptr = &mut x1[..] as *mut [_] as *mut RssShareGeneral<Empty, F> as usize;
            let x2_ptr = &mut x2[..] as *mut [_] as *mut RssShareGeneral<F, Empty> as usize;
            let z: RssShare<F> = (triples, &triple_indices, gammas, coeffs)
                .into_par_iter()
                .map(|(triple, index, gammas, coeffs)| {
                    let x1_slice = unsafe {
                        let x1_ptr =
                            (x1_ptr as *mut RssShareGeneral<Empty, F>).offset(*index as isize);
                        slice::from_raw_parts_mut(x1_ptr, triple.bi.len())
                    };
                    let x2_slice = unsafe {
                        let x2_ptr =
                            (x2_ptr as *mut RssShareGeneral<F, Empty>).offset(*index as isize);
                        slice::from_raw_parts_mut(x2_ptr, triple.bi.len())
                    };

                    if triple.ai.len() == triple.bi.len() {
                        conditional_fill(x1_slice, &triple.aii, &gammas);
                        conditional_fill(x2_slice, &triple.ai, &gammas);
                        self_mul_compute(
                            &triple.ci,
                            &triple.ai,
                            &triple.bi,
                            &triple.ri,
                            &triple.rii,
                            &gammas,
                        )
                    } else {
                        let xi = linear_combination(&triple.ai, &coeffs);
                        let xii = linear_combination(&triple.aii, &coeffs);
                        let ri = linear_combination(&triple.ri, &coeffs);
                        let rii = linear_combination(&triple.rii, &coeffs);
                        let ci = linear_combination(&triple.ci, &coeffs);
                        conditional_fill(x1_slice, &xii, &gammas);
                        conditional_fill(x2_slice, &xi, &gammas);
                        self_mul_compute(&ci, &xi, &triple.bi, &ri, &rii, &gammas)
                    }
                })
                .sum();
            ((x1, x2), z)
        })
        .unzip();
    let (x1, x2) = x.into_iter().unzip();
    ((x1, x2), (y1, y2), z)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
fn next_mul_compute_8<F: Field>(
    ci: &[F],
    ai: &[F],
    bi: &[F],
    ri: &[F],
    gammas: &[u64],
) -> RssShareGeneral<Empty, F> {
    debug_assert_eq!(std::mem::size_of::<F>(), 1);
    debug_assert!(ci.len() % 64 == 0);

    let mut ci_ptr = ci as *const [F] as *const i8;
    let mut ai_ptr = ai as *const [F] as *const i8;
    let mut bi_ptr = bi as *const [F] as *const i8;
    let mut ri_ptr = ri as *const [F] as *const i8;
    unsafe {
        use std::arch::x86_64::{
            _cvtu32_mask32, _mm256_add_epi8, _mm256_maskz_loadu_epi8, _mm256_reduce_add_epi8,
            _mm256_setzero_si256, _mm256_sub_epi8, _mm512_cvtepi16_epi8, _mm512_cvtepu8_epi16,
            _mm512_mullo_epi16,
        };

        let mut sum_si = _mm256_setzero_si256();
        for gamma in gammas.iter() {
            let mask_low = _cvtu32_mask32(*gamma as u32);
            let mask_high = _cvtu32_mask32((*gamma >> 32) as u32);

            for mask in [mask_low, mask_high] {
                let ci = _mm256_maskz_loadu_epi8(mask, ci_ptr);
                let ai = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, ai_ptr));
                let bi = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, bi_ptr));
                let ri = _mm256_maskz_loadu_epi8(mask, ri_ptr);

                // ci + ri - (ai * bi)
                let ai_times_bi = _mm512_cvtepi16_epi8(_mm512_mullo_epi16(ai, bi));
                sum_si = _mm256_sub_epi8(
                    _mm256_add_epi8(_mm256_add_epi8(sum_si, ci), ri),
                    ai_times_bi,
                );

                ci_ptr = ci_ptr.offset(32);
                ai_ptr = ai_ptr.offset(32);
                bi_ptr = bi_ptr.offset(32);
                ri_ptr = ri_ptr.offset(32);
            }
        }
        RssShareGeneral::from(
            Empty,
            std::mem::transmute_copy(&_mm256_reduce_add_epi8(sum_si)),
        )
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
fn next_mul_compute_16<F: Field>(
    ci: &[F],
    ai: &[F],
    bi: &[F],
    ri: &[F],
    gammas: &[u64],
) -> RssShareGeneral<Empty, F> {
    debug_assert_eq!(std::mem::size_of::<F>(), 2);
    debug_assert!(ci.len() % 64 == 0);

    let mut ci_ptr = ci as *const [F] as *const i16;
    let mut ai_ptr = ai as *const [F] as *const i16;
    let mut bi_ptr = bi as *const [F] as *const i16;
    let mut ri_ptr = ri as *const [F] as *const i16;
    unsafe {
        use std::arch::x86_64::{
            _cvtu32_mask32, _mm512_add_epi16, _mm512_maskz_loadu_epi16, _mm512_mullo_epi16,
            _mm512_setzero_si512, _mm512_sub_epi16,
        };

        let mut sum_si = _mm512_setzero_si512();
        for gamma in gammas.iter() {
            let mask_low = _cvtu32_mask32(*gamma as u32);
            let mask_high = _cvtu32_mask32((*gamma >> 32) as u32);

            for mask in [mask_low, mask_high] {
                let ci = _mm512_maskz_loadu_epi16(mask, ci_ptr);
                let ai = _mm512_maskz_loadu_epi16(mask, ai_ptr);
                let bi = _mm512_maskz_loadu_epi16(mask, bi_ptr);
                let ri = _mm512_maskz_loadu_epi16(mask, ri_ptr);

                // ci + ri - (ai * bi)
                let ai_times_bi = _mm512_mullo_epi16(ai, bi);
                sum_si = _mm512_sub_epi16(
                    _mm512_add_epi16(_mm512_add_epi16(sum_si, ci), ri),
                    ai_times_bi,
                );

                ci_ptr = ci_ptr.offset(32);
                ai_ptr = ai_ptr.offset(32);
                bi_ptr = bi_ptr.offset(32);
                ri_ptr = ri_ptr.offset(32);
            }
        }

        RssShareGeneral::from(
            Empty,
            std::mem::transmute_copy(&mm512_reduce_add_epi16(sum_si)),
        )
    }
}

fn next_mul_compute<F: Field>(
    cii: &[F],
    aii: &[F],
    bii: &[F],
    rii: &[F],
    gammas: &[u64],
) -> RssShareGeneral<Empty, F> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
    if std::mem::size_of::<F>() == 1 && cii.len() % 64 == 0 {
        return next_mul_compute_8(cii, aii, bii, rii, gammas);
    } else if std::mem::size_of::<F>() == 2 && cii.len() % 64 == 0 {
        return next_mul_compute_16(cii, aii, bii, rii, gammas);
    }
    izip!(cii, aii, bii, rii)
        .enumerate()
        .map(|(i, (cii, xii, bii, rii))| {
            if gammas[i / 64] & (1 << (i % 64)) != 0 {
                RssShareGeneral::from(Empty, *cii - *xii * *bii + *rii)
            } else {
                RssShareGeneral::ZERO
            }
        })
        .sum()
}

fn get_next_mul_tuples<F: Field + Debug>(
    triples: &[ManyToOneMulTriple<F>],
    global_rng: &mut (impl Rng + CryptoRng),
    kappa: usize,
) -> (
    (
        Vec<AlignedVec<RssShare<Empty>, 64>>,
        Vec<AlignedVec<RssShareGeneral<Empty, F>, 64>>,
    ),
    (
        AlignedVec<RssShareGeneral<Empty, F>, 64>,
        AlignedVec<RssShare<Empty>, 64>,
    ),
    Vec<RssShareGeneral<Empty, F>>,
) {
    let mut triple_indices = vec![0; triples.len()];
    let mut offset = 0;
    for (i, triple) in triples.iter().enumerate() {
        triple_indices[i] = offset;
        offset += triple.bi.len();
    }

    let total_len = offset;

    let mut y1 = vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
    let y1_ptr = &mut y1[..] as *mut [_] as *mut RssShareGeneral<Empty, F> as usize;
    (triples, &triple_indices)
        .into_par_iter()
        .for_each(|(triple, index)| unsafe {
            let y1_ptr = (y1_ptr as *mut RssShareGeneral<Empty, F>).offset(*index as isize);
            ptr::copy_nonoverlapping(
                &triple.bii as &[_] as *const [_] as *const F,
                y1_ptr as *mut F,
                triple.bii.len(),
            );
        });

    let y2 = vec::from_elem_in(EmptyRssShare, total_len, AlignedAllocator::<64>);

    let mut rngs = (0..kappa)
        .map(|_| ChaCha20Rng::from_rng(&mut *global_rng).unwrap())
        .collect::<Vec<_>>();
    let (x, z): (Vec<_>, Vec<_>) = rngs
        .par_iter_mut()
        .map(|global_rng| {
            let x1 = vec::from_elem_in(EmptyRssShare, total_len, AlignedAllocator::<64>);
            let mut x2 =
                vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
            let (gammas, coeffs): (Vec<_>, Vec<_>) = triples
                .iter()
                .map(|triple| {
                    (
                        gen_rand_u64s(global_rng, (triple.bi.len() + 63) / 64),
                        gen_rand_bools(global_rng, triple.ai.len() / triple.bi.len()),
                    )
                })
                .unzip();

            // This is safe because each thread accesses non-overlapping portions of the original slice
            let x2_ptr = &mut x2[..] as *mut [_] as *mut RssShareGeneral<Empty, F> as usize;
            let z: RssShareGeneral<Empty, F> = (triples, &triple_indices, gammas, coeffs)
                .into_par_iter()
                .map(|(triple, index, gammas, coeffs)| {
                    let x2_slice = unsafe {
                        let x2_ptr =
                            (x2_ptr as *mut RssShareGeneral<Empty, F>).offset(*index as isize);
                        slice::from_raw_parts_mut(x2_ptr, triple.bi.len())
                    };

                    if triple.ai.len() == triple.bi.len() {
                        conditional_fill(x2_slice, &triple.aii, &gammas);
                        next_mul_compute(
                            &triple.cii,
                            &triple.aii,
                            &triple.bii,
                            &triple.rii,
                            &gammas,
                        )
                    } else {
                        let xii = linear_combination(&triple.aii, &coeffs);
                        let rii = linear_combination(&triple.rii, &coeffs);
                        let cii = linear_combination(&triple.cii, &coeffs);
                        conditional_fill(x2_slice, &xii, &gammas);
                        next_mul_compute(&cii, &xii, &triple.bii, &rii, &gammas)
                    }
                })
                .sum();
            ((x1, x2), z)
        })
        .unzip();
    let (x1, x2) = x.into_iter().unzip();
    ((x1, x2), (y1, y2), z)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
fn prev_mul_compute_8<F: Field>(rii: &[F], gammas: &[u64]) -> RssShareGeneral<F, Empty> {
    debug_assert_eq!(std::mem::size_of::<F>(), 1);
    debug_assert!(rii.len() % 64 == 0);

    let mut rii_ptr = rii as *const [F] as *const i8;
    unsafe {
        use std::arch::x86_64::{
            _cvtu32_mask32, _mm256_maskz_loadu_epi8, _mm256_reduce_add_epi8, _mm256_setzero_si256,
            _mm256_sub_epi8,
        };

        let mut sum_sii = _mm256_setzero_si256();
        for gamma in gammas.iter() {
            let mask_low = _cvtu32_mask32(*gamma as u32);
            let mask_high = _cvtu32_mask32((*gamma >> 32) as u32);

            for mask in [mask_low, mask_high] {
                let rii = _mm256_maskz_loadu_epi8(mask, rii_ptr);

                // ci + ri - (ai * bi)
                sum_sii = _mm256_sub_epi8(sum_sii, rii);
                rii_ptr = rii_ptr.offset(32);
            }
        }
        RssShareGeneral::from(
            std::mem::transmute_copy(&_mm256_reduce_add_epi8(sum_sii)),
            Empty,
        )
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
fn prev_mul_compute_16<F: Field>(rii: &[F], gammas: &[u64]) -> RssShareGeneral<F, Empty> {
    debug_assert_eq!(std::mem::size_of::<F>(), 2);
    debug_assert!(rii.len() % 64 == 0);

    let mut rii_ptr = rii as *const [F] as *const i16;
    unsafe {
        use std::arch::x86_64::{
            _cvtu32_mask32, _mm512_maskz_loadu_epi16, _mm512_setzero_si512, _mm512_sub_epi16,
        };

        let mut sum_sii = _mm512_setzero_si512();
        for gamma in gammas.iter() {
            let mask_low = _cvtu32_mask32(*gamma as u32);
            let mask_high = _cvtu32_mask32((*gamma >> 32) as u32);

            for mask in [mask_low, mask_high] {
                let rii = _mm512_maskz_loadu_epi16(mask, rii_ptr);
                sum_sii = _mm512_sub_epi16(sum_sii, rii);
                rii_ptr = rii_ptr.offset(32);
            }
        }

        RssShareGeneral::from(
            std::mem::transmute_copy(&mm512_reduce_add_epi16(sum_sii)),
            Empty,
        )
    }
}

fn prev_mul_compute<F: Field>(ri: &[F], gammas: &[u64]) -> RssShareGeneral<F, Empty> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
    if std::mem::size_of::<F>() == 1 && ri.len() % 64 == 0 {
        return prev_mul_compute_8(ri, gammas);
    } else if std::mem::size_of::<F>() == 2 && ri.len() % 64 == 0 {
        return prev_mul_compute_16(ri, gammas);
    }
    ri.iter()
        .enumerate()
        .map(|(i, ri)| {
            if gammas[i / 64] & (1 << (i % 64)) != 0 {
                RssShareGeneral::from(-*ri, Empty)
            } else {
                RssShareGeneral::ZERO
            }
        })
        .sum()
}

fn get_prev_mul_tuples<F: Field + Debug>(
    triples: &[ManyToOneMulTriple<F>],
    global_rng: &mut (impl Rng + CryptoRng),
    kappa: usize,
) -> (
    (
        Vec<AlignedVec<RssShareGeneral<F, Empty>, 64>>,
        Vec<AlignedVec<RssShare<Empty>, 64>>,
    ),
    (
        AlignedVec<RssShare<Empty>, 64>,
        AlignedVec<RssShareGeneral<F, Empty>, 64>,
    ),
    Vec<RssShareGeneral<F, Empty>>,
) {
    let mut triple_indices = vec![0; triples.len()];
    let mut offset = 0;
    for (i, triple) in triples.iter().enumerate() {
        triple_indices[i] = offset;
        offset += triple.bi.len();
    }

    let total_len = offset;

    let mut y2 = vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
    let y2_ptr = &mut y2[..] as *mut [_] as *mut RssShareGeneral<F, Empty> as usize;
    (triples, &triple_indices)
        .into_par_iter()
        .for_each(|(triple, index)| unsafe {
            let y2_ptr = (y2_ptr as *mut RssShareGeneral<F, Empty>).offset(*index as isize);
            ptr::copy_nonoverlapping(
                &triple.bi as &[_] as *const [_] as *const F,
                y2_ptr as *mut F,
                triple.bi.len(),
            );
        });

    let y1 = vec::from_elem_in(EmptyRssShare, total_len, AlignedAllocator::<64>);

    let mut rngs = (0..kappa)
        .map(|_| ChaCha20Rng::from_rng(&mut *global_rng).unwrap())
        .collect::<Vec<_>>();
    let (x, z): (Vec<_>, Vec<_>) = rngs
        .par_iter_mut()
        .map(|global_rng| {
            let mut x1 =
                vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
            let x2 = vec::from_elem_in(EmptyRssShare, total_len, AlignedAllocator::<64>);
            let (gammas, coeffs): (Vec<_>, Vec<_>) = triples
                .iter()
                .map(|triple| {
                    (
                        gen_rand_u64s(global_rng, (triple.bi.len() + 63) / 64),
                        gen_rand_bools(global_rng, triple.ai.len() / triple.bi.len()),
                    )
                })
                .unzip();

            // This is safe because each thread accesses non-overlapping portions of the original slice
            let x1_ptr = &mut x1[..] as *mut [_] as *mut RssShareGeneral<F, Empty> as usize;
            let z: RssShareGeneral<F, Empty> = (triples, &triple_indices, gammas, coeffs)
                .into_par_iter()
                .map(|(triple, index, gammas, coeffs)| {
                    let x1_slice = unsafe {
                        let x1_ptr =
                            (x1_ptr as *mut RssShareGeneral<F, Empty>).offset(*index as isize);
                        slice::from_raw_parts_mut(x1_ptr, triple.bi.len())
                    };

                    if triple.ai.len() == triple.bi.len() {
                        conditional_fill(x1_slice, &triple.ai, &gammas);
                        prev_mul_compute(&triple.ri, &gammas)
                    } else {
                        let xi = linear_combination(&triple.ai, &coeffs);
                        let ri = linear_combination(&triple.ri, &coeffs);
                        conditional_fill(x1_slice, &xi, &gammas);
                        prev_mul_compute(&ri, &gammas)
                    }
                })
                .sum();
            ((x1, x2), z)
        })
        .unzip();
    let (x1, x2) = x.into_iter().unzip();
    ((x1, x2), (y1, y2), z)
}

pub fn process_inner_product_triple_self_impl<
    F: Field,
    const ELEMS_PER_LOOKUP: usize,
    const ELEMS_PER_BLOCK: usize,
>(
    triple: &InnerProductTriple<F>,
    gammas: &[u64],
    coeffs: &[u8],
    x1_slice: &mut [RssShareGeneral<Empty, F>],
    x2_slice: &mut [RssShareGeneral<F, Empty>],
) -> RssShare<F>
where
    [(); ELEMS_PER_LOOKUP / ELEMS_PER_BLOCK]:,
{
    if ELEMS_PER_BLOCK == 1 {
        return (
            triple.ai.as_chunks::<ELEMS_PER_LOOKUP>().0,
            triple.aii.as_chunks::<ELEMS_PER_LOOKUP>().0,
            triple.ci.par_iter(),
            triple.ri.par_iter(),
            triple.rii.par_iter(),
            x1_slice.as_chunks_mut::<ELEMS_PER_LOOKUP>().0,
            x2_slice.as_chunks_mut::<ELEMS_PER_LOOKUP>().0,
        )
            .into_par_iter()
            .enumerate()
            .map(|(i, (ai, aii, ci, ri, rii, x1, x2))| {
                if gammas[i / 64] & (1 << (i % 64)) == 0 {
                    return RssShare::ZERO;
                }
                unsafe {
                    ptr::copy_nonoverlapping(
                        aii as &[F] as *const [F] as *const F,
                        x1 as *mut [_] as *mut F,
                        aii.len(),
                    );
                    ptr::copy_nonoverlapping(
                        ai as &[F] as *const [F] as *const F,
                        x2 as *mut [_] as *mut F,
                        ai.len(),
                    );
                }
                RssShare::from(
                    *ci - ai
                        .iter()
                        .enumerate()
                        .map(|(k, xi)| *xi * triple.bi[k * triple.ci.len() + i])
                        .sum::<F>()
                        + *ri,
                    -*rii,
                )
            })
            .sum();
    }

    (
        triple.ai.as_chunks::<ELEMS_PER_LOOKUP>().0,
        triple.aii.as_chunks::<ELEMS_PER_LOOKUP>().0,
        triple.ci.as_chunks::<ELEMS_PER_BLOCK>().0,
        triple.ri.as_chunks::<ELEMS_PER_BLOCK>().0,
        triple.rii.as_chunks::<ELEMS_PER_BLOCK>().0,
        x1_slice
            .as_chunks::<{ ELEMS_PER_LOOKUP / ELEMS_PER_BLOCK }>()
            .0,
        x2_slice
            .as_chunks::<{ ELEMS_PER_LOOKUP / ELEMS_PER_BLOCK }>()
            .0,
        coeffs.as_chunks::<ELEMS_PER_BLOCK>().0,
    )
        .into_par_iter()
        .enumerate()
        .map(|(i, (ai, aii, ci, ri, rii, x1, x2, coeffs))| {
            if gammas[i / 64] & (1 << (i % 64)) == 0 {
                return RssShare::ZERO;
            }
            unsafe {
                // Transmuting &mut [F] -> &mut [RssShare<Empty, F>]: should be safe
                linear_combination_alt(std::mem::transmute_copy(&x2), ai, coeffs);
                linear_combination_alt(std::mem::transmute_copy(&x1), aii, coeffs);
            }
            let ri = linear_combination_single(ri, coeffs);
            let rii = linear_combination_single(rii, coeffs);
            let ci = linear_combination_single(ci, coeffs);
            let num_lookups = triple.ci.len() / triple.elems_per_block;
            RssShare::from(
                ci - x2
                    .iter()
                    .enumerate()
                    .map(|(k, xi)| xi.si * triple.bi[k * num_lookups + i])
                    .sum::<F>()
                    + ri,
                -rii,
            )
        })
        .sum()
}

fn get_self_inner_product_tuples<F: Field + Debug, Table: LUT256SPMalTable<F>>(
    triples: &[InnerProductTriple<F>],
    global_rng: &mut (impl Rng + CryptoRng),
    kappa: usize,
) -> (
    (
        Vec<AlignedVec<RssShareGeneral<Empty, F>, 64>>,
        Vec<AlignedVec<RssShareGeneral<F, Empty>, 64>>,
    ),
    (
        AlignedVec<RssShareGeneral<F, Empty>, 64>,
        AlignedVec<RssShareGeneral<Empty, F>, 64>,
    ),
    Vec<RssShare<F>>,
) {
    // Self's instance (functioning as P_i)

    let mut triple_indices = vec![0; triples.len()];
    let mut offset = 0;
    for (i, triple) in triples.iter().enumerate() {
        triple_indices[i] = offset;
        offset += triple.bi.len();
    }
    let total_len = offset;

    let mut y1 = vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
    let mut y2 = vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
    let y1_ptr = &mut y1[..] as *mut [_] as *mut RssShareGeneral<F, Empty> as usize;
    let y2_ptr = &mut y2[..] as *mut [_] as *mut RssShareGeneral<Empty, F> as usize;
    (triples, &triple_indices)
        .into_par_iter()
        .for_each(|(triple, index)| {
            let y1_ptr = y1_ptr as *mut RssShareGeneral<F, Empty>;
            let y2_ptr = y2_ptr as *mut RssShareGeneral<Empty, F>;

            let num_lookups = triple.ci.len() / triple.elems_per_block;
            let b_len = triple.elems_per_lookup / triple.elems_per_block;
            let mut index = *index;
            for i in 0..num_lookups {
                for j in 0..b_len {
                    unsafe {
                        *y1_ptr.add(index) =
                            RssShareGeneral::from(triple.bi[j * num_lookups + i], Empty);
                        *y2_ptr.add(index) =
                            RssShareGeneral::from(Empty, triple.bii[j * num_lookups + i]);
                    }

                    index += 1;
                }
            }
        });

    let mut rngs = (0..kappa)
        .map(|_| ChaCha20Rng::from_rng(&mut *global_rng).unwrap())
        .collect::<Vec<_>>();
    let (x, z): (Vec<_>, Vec<_>) = rngs
        .par_iter_mut()
        .map(|global_rng| {
            let mut x1 =
                vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
            let mut x2 =
                vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
            let (gammas, coeffs): (Vec<_>, Vec<_>) = triples
                .iter()
                .map(|triple| {
                    let num_lookups = triple.ci.len() / triple.elems_per_block;
                    (
                        gen_rand_u64s(global_rng, (num_lookups + 63) / 64),
                        gen_rand_bools(global_rng, triple.ci.len()),
                    )
                })
                .unzip();

            // This is safe because each thread accesses non-overlapping portions of the original slice
            let x1_ptr = &mut x1[..] as *mut [_] as *mut RssShareGeneral<Empty, F> as usize;
            let x2_ptr = &mut x2[..] as *mut [_] as *mut RssShareGeneral<F, Empty> as usize;
            let z: RssShare<F> = (triples, &triple_indices, gammas, coeffs)
                .into_par_iter()
                .map(|(triple, index, gammas, coeffs)| {
                    let x1_slice = unsafe {
                        let x1_ptr =
                            (x1_ptr as *mut RssShareGeneral<Empty, F>).offset(*index as isize);
                        slice::from_raw_parts_mut(x1_ptr, triple.bi.len())
                    };
                    let x2_slice = unsafe {
                        let x2_ptr =
                            (x2_ptr as *mut RssShareGeneral<F, Empty>).offset(*index as isize);
                        slice::from_raw_parts_mut(x2_ptr, triple.bi.len())
                    };
                    Table::process_inner_product_triple_self(
                        triple, &gammas, &coeffs, x1_slice, x2_slice,
                    )
                })
                .sum();
            ((x1, x2), z)
        })
        .unzip();
    let (x1, x2) = x.into_iter().unzip();
    ((x1, x2), (y1, y2), z)
}

pub fn process_inner_product_triple_next_impl<
    F: Field,
    const ELEMS_PER_LOOKUP: usize,
    const ELEMS_PER_BLOCK: usize,
>(
    triple: &InnerProductTriple<F>,
    gammas: &[u64],
    coeffs: &[u8],
    x2_slice: &mut [RssShareGeneral<Empty, F>],
) -> RssShareGeneral<Empty, F>
where
    [(); ELEMS_PER_LOOKUP / ELEMS_PER_BLOCK]:,
{
    if ELEMS_PER_BLOCK == 1 {
        return (
            triple.aii.as_chunks::<ELEMS_PER_LOOKUP>().0,
            triple.cii.par_iter(),
            triple.rii.par_iter(),
            x2_slice.as_chunks_mut::<ELEMS_PER_LOOKUP>().0,
        )
            .into_par_iter()
            .enumerate()
            .map(|(i, (aii, cii, rii, x2))| {
                if gammas[i / 64] & (1 << (i % 64)) == 0 {
                    return RssShareGeneral::ZERO;
                }

                unsafe {
                    ptr::copy_nonoverlapping(
                        aii as &[F] as *const [F] as *const F,
                        x2 as *mut [_] as *mut F,
                        aii.len(),
                    );
                }

                RssShareGeneral::from(
                    Empty,
                    *cii - aii
                        .iter()
                        .enumerate()
                        .map(|(k, xii)| *xii * triple.bii[k * triple.ci.len() + i])
                        .sum::<F>()
                        + *rii,
                )
            })
            .sum();
    }

    (
        triple.aii.as_chunks::<ELEMS_PER_LOOKUP>().0,
        triple.cii.as_chunks::<ELEMS_PER_BLOCK>().0,
        triple.rii.as_chunks::<ELEMS_PER_BLOCK>().0,
        x2_slice
            .as_chunks_mut::<{ ELEMS_PER_LOOKUP / ELEMS_PER_BLOCK }>()
            .0,
        coeffs.as_chunks::<ELEMS_PER_BLOCK>().0,
    )
        .into_par_iter()
        .enumerate()
        .map(|(i, (aii, cii, rii, x2, coeffs))| {
            if gammas[i / 64] & (1 << (i % 64)) == 0 {
                return RssShareGeneral::ZERO;
            }

            unsafe {
                linear_combination_alt(std::mem::transmute_copy(&x2), aii, coeffs);
            }
            let rii = linear_combination_single(rii, coeffs);
            let cii = linear_combination_single(cii, coeffs);

            let num_lookups = triple.ci.len() / triple.elems_per_block;
            RssShareGeneral::from(
                Empty,
                cii - x2
                    .iter()
                    .enumerate()
                    .map(|(k, xii)| xii.sii * triple.bii[k * num_lookups + i])
                    .sum::<F>()
                    + rii,
            )
        })
        .sum()
}

fn get_next_inner_product_tuples<F: Field + Debug, Table: LUT256SPMalTable<F>>(
    triples: &[InnerProductTriple<F>],
    global_rng: &mut (impl Rng + CryptoRng),
    kappa: usize,
) -> (
    (
        Vec<AlignedVec<RssShare<Empty>, 64>>,
        Vec<AlignedVec<RssShareGeneral<Empty, F>, 64>>,
    ),
    (
        AlignedVec<RssShareGeneral<Empty, F>, 64>,
        AlignedVec<RssShare<Empty>, 64>,
    ),
    Vec<RssShareGeneral<Empty, F>>,
) {
    // Next's instance (self as P_{i-1})

    let mut triple_indices = vec![0; triples.len()];
    let mut offset = 0;
    for (i, triple) in triples.iter().enumerate() {
        triple_indices[i] = offset;
        offset += triple.bi.len();
    }
    let total_len = offset;

    let mut y1 = vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
    let y1_ptr = &mut y1[..] as *mut [_] as *mut RssShareGeneral<Empty, F> as usize;
    (triples, &triple_indices)
        .into_par_iter()
        .for_each(|(triple, index)| {
            let y1_ptr = y1_ptr as *mut RssShareGeneral<Empty, F>;

            let num_lookups = triple.ci.len() / triple.elems_per_block;
            let b_len = triple.elems_per_lookup / triple.elems_per_block;
            let mut index = *index;
            for i in 0..num_lookups {
                for j in 0..b_len {
                    unsafe {
                        *y1_ptr.add(index) =
                            RssShareGeneral::from(Empty, triple.bii[j * num_lookups + i]);
                    }

                    index += 1;
                }
            }
        });

    let y2 = vec::from_elem_in(EmptyRssShare, total_len, AlignedAllocator::<64>);

    let mut rngs = (0..kappa)
        .map(|_| ChaCha20Rng::from_rng(&mut *global_rng).unwrap())
        .collect::<Vec<_>>();
    let (x, z): (Vec<_>, Vec<_>) = rngs
        .par_iter_mut()
        .map(|global_rng| {
            let x1 = vec::from_elem_in(EmptyRssShare, total_len, AlignedAllocator::<64>);
            let mut x2 =
                vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
            let (gammas, coeffs): (Vec<_>, Vec<_>) = triples
                .iter()
                .map(|triple| {
                    let num_lookups = triple.ci.len() / triple.elems_per_block;
                    (
                        gen_rand_u64s(global_rng, (num_lookups + 63) / 64),
                        gen_rand_bools(global_rng, triple.ci.len()),
                    )
                })
                .unzip();

            let x2_ptr = &mut x2[..] as *mut [_] as *mut RssShareGeneral<Empty, F> as usize;
            let z: RssShareGeneral<Empty, F> = (triples, &triple_indices, gammas, coeffs)
                .into_par_iter()
                .map(|(triple, index, gammas, coeffs)| {
                    let x2_slice = unsafe {
                        let x2_ptr =
                            (x2_ptr as *mut RssShareGeneral<Empty, F>).offset(*index as isize);
                        slice::from_raw_parts_mut(x2_ptr, triple.bi.len())
                    };
                    Table::process_inner_product_triple_next(triple, &gammas, &coeffs, x2_slice)
                })
                .sum();

            ((x1, x2), z)
        })
        .unzip();
    let (x1, x2) = x.into_iter().unzip();
    ((x1, x2), (y1, y2), z)
}

pub fn process_inner_product_triple_prev_impl<
    F: Field,
    const ELEMS_PER_LOOKUP: usize,
    const ELEMS_PER_BLOCK: usize,
>(
    triple: &InnerProductTriple<F>,
    gammas: &[u64],
    coeffs: &[u8],
    x1_slice: &mut [RssShareGeneral<F, Empty>],
) -> RssShareGeneral<F, Empty>
where
    [(); ELEMS_PER_LOOKUP / ELEMS_PER_BLOCK]:,
{
    if ELEMS_PER_BLOCK == 1 {
        return (
            triple.ai.as_chunks::<ELEMS_PER_LOOKUP>().0,
            triple.ri.par_iter(),
            x1_slice.as_chunks_mut::<ELEMS_PER_LOOKUP>().0,
        )
            .into_par_iter()
            .enumerate()
            .map(|(i, (ai, ri, x1))| {
                if gammas[i / 64] & (1 << (i % 64)) == 0 {
                    return RssShareGeneral::ZERO;
                }

                unsafe {
                    ptr::copy_nonoverlapping(
                        ai as &[F] as *const [F] as *const F,
                        x1 as *mut [_] as *mut F,
                        ai.len(),
                    );
                }

                RssShareGeneral::from(-*ri, Empty)
            })
            .sum();
    }

    (
        triple.ai.as_chunks::<ELEMS_PER_LOOKUP>().0,
        triple.ri.as_chunks::<ELEMS_PER_BLOCK>().0,
        x1_slice
            .as_chunks_mut::<{ ELEMS_PER_LOOKUP / ELEMS_PER_BLOCK }>()
            .0,
        coeffs.as_chunks::<ELEMS_PER_BLOCK>().0,
    )
        .into_par_iter()
        .enumerate()
        .map(|(i, (ai, ri, x1, coeffs))| {
            if gammas[i / 64] & (1 << (i % 64)) == 0 {
                return RssShareGeneral::ZERO;
            }

            unsafe {
                linear_combination_alt(std::mem::transmute_copy(&x1), ai, coeffs);
            }
            let ri = linear_combination_single(ri, coeffs);
            RssShareGeneral::from(-ri, Empty)
        })
        .sum()
}

fn get_prev_inner_product_tuples<F: Field + Debug, Table: LUT256SPMalTable<F>>(
    triples: &[InnerProductTriple<F>],
    global_rng: &mut (impl Rng + CryptoRng),
    kappa: usize,
) -> (
    (
        Vec<AlignedVec<RssShareGeneral<F, Empty>, 64>>,
        Vec<AlignedVec<RssShare<Empty>, 64>>,
    ),
    (
        AlignedVec<RssShare<Empty>, 64>,
        AlignedVec<RssShareGeneral<F, Empty>, 64>,
    ),
    Vec<RssShareGeneral<F, Empty>>,
) {
    // Prev's instance (self as P_{i+1})

    let mut triple_indices = vec![0; triples.len()];
    let mut offset = 0;
    for (i, triple) in triples.iter().enumerate() {
        triple_indices[i] = offset;
        offset += triple.bi.len();
    }
    let total_len = offset;

    let mut y2 = vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
    let y2_ptr = &mut y2[..] as *mut [_] as *mut RssShareGeneral<F, Empty> as usize;
    (triples, &triple_indices)
        .into_par_iter()
        .for_each(|(triple, index)| {
            let y2_ptr = y2_ptr as *mut RssShareGeneral<F, Empty>;

            let num_lookups = triple.ci.len() / triple.elems_per_block;
            let b_len = triple.elems_per_lookup / triple.elems_per_block;
            let mut index = *index;
            for i in 0..num_lookups {
                for j in 0..b_len {
                    unsafe {
                        *y2_ptr.add(index) =
                            RssShareGeneral::from(triple.bi[j * num_lookups + i], Empty);
                    }

                    index += 1;
                }
            }
        });
    let y1 = vec::from_elem_in(EmptyRssShare, total_len, AlignedAllocator::<64>);

    let mut rngs = (0..kappa)
        .map(|_| ChaCha20Rng::from_rng(&mut *global_rng).unwrap())
        .collect::<Vec<_>>();
    let (x, z): (Vec<_>, Vec<_>) = rngs
        .par_iter_mut()
        .map(|global_rng| {
            let mut x1 =
                vec::from_elem_in(RssShareGeneral::ZERO, total_len, AlignedAllocator::<64>);
            let x2 = vec::from_elem_in(EmptyRssShare, total_len, AlignedAllocator::<64>);

            let (gammas, coeffs): (Vec<_>, Vec<_>) = triples
                .iter()
                .map(|triple| {
                    let num_lookups = triple.ci.len() / triple.elems_per_block;
                    (
                        gen_rand_u64s(global_rng, (num_lookups + 63) / 64),
                        gen_rand_bools(global_rng, triple.ci.len()),
                    )
                })
                .unzip();

            let x1_ptr = &mut x1[..] as *mut [_] as *mut RssShareGeneral<Empty, F> as usize;

            let z: RssShareGeneral<F, Empty> = (triples, &triple_indices, gammas, coeffs)
                .into_par_iter()
                .map(|(triple, index, gammas, coeffs)| {
                    let x1_slice = unsafe {
                        let x1_ptr =
                            (x1_ptr as *mut RssShareGeneral<F, Empty>).offset(*index as isize);
                        slice::from_raw_parts_mut(x1_ptr, triple.bi.len())
                    };

                    Table::process_inner_product_triple_prev(triple, &gammas, &coeffs, x1_slice)
                })
                .sum();
            ((x1, x2), z)
        })
        .unzip();
    let (x1, x2) = x.into_iter().unzip();
    ((x1, x2), (y1, y2), z)
}

fn binary_shares_to_self_instance<F: Field + Debug>(
    binary_shares: &[RssShare<F>],
) -> (
    Vec<RssShareGeneral<Empty, F>>,
    Vec<RssShareGeneral<F, Empty>>,
    Vec<RssShare<F>>,
) {
    let x = binary_shares
        .iter()
        .map(|share| RssShareGeneral::from(Empty, share.sii + share.sii))
        .collect::<Vec<_>>();
    let y = binary_shares
        .iter()
        .map(|share| RssShareGeneral::from(-share.si, Empty))
        .collect::<Vec<_>>();
    let z = binary_shares
        .iter()
        .map(|share| {
            let z1 = (share.sii - F::ONE) * share.sii;
            let z2 = (share.si - F::ONE) * share.si;
            RssShare::from(z2, z1)
        })
        .collect::<Vec<_>>();
    (x, y, z)
}

fn binary_shares_to_next_instance<F: Field + Debug>(
    binary_shares: &[RssShareGeneral<Empty, F>],
) -> (
    Vec<RssShare<Empty>>,
    Vec<RssShareGeneral<Empty, F>>,
    Vec<RssShareGeneral<Empty, F>>,
) {
    let x = vec![EmptyRssShare; binary_shares.len()];
    let y = binary_shares
        .iter()
        .map(|share| RssShareGeneral::from(Empty, -share.sii))
        .collect::<Vec<_>>();
    let z = binary_shares
        .iter()
        .map(|share| RssShareGeneral::from(Empty, (share.sii - F::ONE) * share.sii))
        .collect::<Vec<_>>();
    (x, y, z)
}

fn binary_shares_to_prev_instance<F: Field + Debug>(
    binary_shares: &[RssShareGeneral<F, Empty>],
) -> (
    Vec<RssShareGeneral<F, Empty>>,
    Vec<RssShare<Empty>>,
    Vec<RssShareGeneral<F, Empty>>,
) {
    let x = binary_shares
        .iter()
        .map(|share| RssShareGeneral::from(share.si + share.si, Empty))
        .collect::<Vec<_>>();
    let y = vec![EmptyRssShare; binary_shares.len()];
    let z = binary_shares
        .iter()
        .map(|share| RssShareGeneral::from((share.si - F::ONE) * share.si, Empty))
        .collect::<Vec<_>>();
    (x, y, z)
}

fn batch_mul_tuples<
    F: Field + Debug,
    T1: FieldLike + Mul<F, Output = T1>,
    U1: FieldLike + Mul<F, Output = U1>,
    T2: FieldLike + Mul<F, Output = T2>,
    U2: FieldLike + Mul<F, Output = U2>,
    T3: FieldLike + Mul<F, Output = T3>,
    U3: FieldLike + Mul<F, Output = U3>,
>(
    (x, y, z): (
        Vec<RssShareGeneral<T1, U1>>,
        Vec<RssShareGeneral<T2, U2>>,
        Vec<RssShareGeneral<T3, U3>>,
    ),
    global_rng: &mut (impl Rng + CryptoRng),
    kappa: usize,
) -> (
    Vec<Vec<RssShareGeneral<T1, U1>>>,
    Vec<RssShareGeneral<T2, U2>>,
    Vec<RssShareGeneral<T3, U3>>,
) {
    let gammas = (0..kappa)
        .map(|_| F::generate(global_rng, x.len()))
        .collect::<Vec<_>>();
    let (x, z): (Vec<_>, Vec<_>) = gammas
        .par_iter()
        .map(|gamma| {
            (
                x.iter()
                    .zip(gamma.iter())
                    .map(|(x, gamma)| *x * *gamma)
                    .collect::<Vec<_>>(),
                z.iter()
                    .zip(gamma.iter())
                    .map(|(z, gamma)| *z * *gamma)
                    .sum::<RssShareGeneral<T3, U3>>(),
            )
        })
        .unzip();
    (x, y, z)
}

fn combine_instances<
    T1: FieldLike,
    U1: FieldLike,
    T2: FieldLike,
    U2: FieldLike,
    T3: FieldLike,
    U3: FieldLike,
>(
    ((mut x11, mut x12), (mut y11, mut y12), mut z1): (
        (
            Vec<AlignedVec<RssShareGeneral<T1, U1>, 64>>,
            Vec<AlignedVec<RssShareGeneral<T2, U2>, 64>>,
        ),
        (
            AlignedVec<RssShareGeneral<T2, U2>, 64>,
            AlignedVec<RssShareGeneral<T1, U1>, 64>,
        ),
        Vec<RssShareGeneral<T3, U3>>,
    ),
    ((x21, x22), (mut y21, mut y22), z2): (
        (
            Vec<AlignedVec<RssShareGeneral<T1, U1>, 64>>,
            Vec<AlignedVec<RssShareGeneral<T2, U2>, 64>>,
        ),
        (
            AlignedVec<RssShareGeneral<T2, U2>, 64>,
            AlignedVec<RssShareGeneral<T1, U1>, 64>,
        ),
        Vec<RssShareGeneral<T3, U3>>,
    ),
) -> (
    (
        Vec<AlignedVec<RssShareGeneral<T1, U1>, 64>>,
        Vec<AlignedVec<RssShareGeneral<T2, U2>, 64>>,
    ),
    (
        AlignedVec<RssShareGeneral<T2, U2>, 64>,
        AlignedVec<RssShareGeneral<T1, U1>, 64>,
    ),
    Vec<RssShareGeneral<T3, U3>>,
) {
    for (x1, mut x2) in zip_eq(&mut x11, x21) {
        x1.append(&mut x2);
    }
    for (x1, mut x2) in zip_eq(&mut x12, x22) {
        x1.append(&mut x2);
    }
    y11.append(&mut y21);
    y12.append(&mut y22);
    for (z1, z2) in zip_eq(&mut z1, z2) {
        *z1 += z2;
    }
    ((x11, x12), (y11, y12), z1)
}

fn bit_decompose<Fp: From<u64>>(input: usize, num_var: usize) -> Vec<Fp> {
    let mut res = Vec::with_capacity(num_var);
    let mut i = input;
    for _ in 0..num_var {
        res.push(Fp::from((i & 1 == 1) as u64));
        i >>= 1;
    }
    res
}

fn reconstruct_t<
    F: Field,
    Fp: PrimeField,
    T1: FieldLike + Mul<Fp, Output = T1>,
    U1: FieldLike + Mul<Fp, Output = U1>,
>(
    t_shares: &[RssShareGeneral<T1, U1>],
) -> RssShareGeneral<T1, U1> {
    let mut res = RssShareGeneral::ZERO;
    let mut multiplier = Fp::from(1 << F::NBITS);
    for t_share in t_shares {
        res += *t_share * multiplier;
        multiplier += multiplier;
    }
    res
}

fn prove_lift_instances<F: Field, Fp: PrimeField>(
    ((x1, x2), (y1, y2), z): &(
        (
            Vec<AlignedVec<RssShareGeneral<Empty, F>, 64>>,
            Vec<AlignedVec<RssShareGeneral<F, Empty>, 64>>,
        ),
        (
            AlignedVec<RssShareGeneral<F, Empty>, 64>,
            AlignedVec<RssShareGeneral<Empty, F>, 64>,
        ),
        Vec<RssShare<F>>,
    ),
) -> Vec<Fp> {
    let h = (x1, x2)
        .into_par_iter()
        .map(|(x1, x2)| {
            x1.iter()
                .zip_eq(y1.iter())
                .map(|(x, y)| x.sii.as_raw() * y.si.as_raw())
                .sum::<usize>()
                + x2.iter()
                    .zip_eq(y2.iter())
                    .map(|(x, y)| x.si.as_raw() * y.sii.as_raw())
                    .sum::<usize>()
        })
        .collect::<Vec<_>>();
    let t = h
        .iter()
        .zip_eq(z.iter())
        .map(|(h, z)| {
            let val = h + (1 << F::NBITS) - (z.si.as_raw() + z.sii.as_raw());
            // debug_assert!(val % (1 << F::NBITS) == 0);
            val >> F::NBITS
        })
        .collect::<Vec<_>>();
    t.iter()
        .flat_map(|t| bit_decompose::<Fp>(*t, Fp::NBITS - F::NBITS))
        .collect()
}

fn generate_powers<Fp: PrimeField>(r: Fp, len: usize) -> Vec<Fp> {
    let mut result = vec![Fp::ONE; len];
    for i in 1..result.len() {
        result[i] = result[i - 1] * r;
    }
    result
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f",))]
fn batch_and_lift_x_8_mersenne<
    Fp: PrimeField,
    T1: FieldLike + Lift<Fp>,
    U1: FieldLike + Lift<Fp>,
>(
    out: &mut [<RssShareGeneral<T1, U1> as Lift<Fp>>::Output],
    x: &[AlignedVec<RssShareGeneral<T1, U1>, 64>],
    offset: usize,
    powers: &[Fp],
) {
    use std::arch::x86_64::{
        __m128i, _mm_loadl_epi64, _mm512_add_epi64, _mm512_and_si512, _mm512_cvtepu8_epi64,
        _mm512_mul_epu32, _mm512_set1_epi64, _mm512_setzero_epi32, _mm512_slli_epi64,
        _mm512_srli_epi64, _mm512_store_epi64,
    };
    unsafe {
        let modulus = _mm512_set1_epi64(0x1FFFFFFFFFFFFFFF);
        let mut out_ptr =
            out as *mut [_] as *mut <RssShareGeneral<T1, U1> as Lift<Fp>>::Output as *mut i64;
        let mut x_ptrs = x
            .iter()
            .map(|x_vec| {
                (&x_vec[..] as *const [_] as *const RssShareGeneral<T1, U1> as *const i8)
                    .add(offset)
            })
            .collect::<Vec<_>>();
        let x_ptr_len = x_ptrs.len();
        for _ in (0..out.len()).step_by(8) {
            let mut sum = _mm512_setzero_epi32();

            for (i, x_ptr) in x_ptrs.iter_mut().enumerate() {
                let bytes = _mm_loadl_epi64(*x_ptr as *const __m128i);
                let x = _mm512_cvtepu8_epi64(bytes);
                let p = _mm512_set1_epi64(std::mem::transmute_copy(&powers[i]));
                let p_high = _mm512_set1_epi64(
                    (std::mem::transmute_copy::<_, u64>(&powers[i]) >> 32) as i64,
                );

                // Multiplication, with special provision that x does not have a high

                // In the following analysis we assume x has 16 bits
                let r1 = _mm512_mul_epu32(x, p_high); // at most 48 bits
                let r3 = _mm512_mul_epu32(x, p); // at most 45 bits

                let h0 = _mm512_and_si512(_mm512_slli_epi64::<32>(r1), modulus); // 61 bits
                let rl = _mm512_add_epi64(r3, h0);

                let h1 = _mm512_srli_epi64::<29>(r1); // at most 16 bits
                let rl = _mm512_add_epi64(rl, h1); // rl can have at most 62 bits

                sum = _mm512_add_epi64(sum, rl);

                if i % 4 == 3 || i == x_ptr_len - 1 {
                    let rh = _mm512_srli_epi64::<61>(sum);
                    sum = _mm512_add_epi64(_mm512_and_si512(sum, modulus), rh);
                }

                *x_ptr = x_ptr.add(8);
            }

            _mm512_store_epi64(out_ptr, sum);
            out_ptr = out_ptr.add(8);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f",))]
fn batch_and_lift_x_16_mersenne<
    Fp: PrimeField,
    T1: FieldLike + Lift<Fp>,
    U1: FieldLike + Lift<Fp>,
>(
    out: &mut [<RssShareGeneral<T1, U1> as Lift<Fp>>::Output],
    x: &[AlignedVec<RssShareGeneral<T1, U1>, 64>],
    offset: usize,
    powers: &[Fp],
) {
    use std::arch::x86_64::{
        __m128i, _mm_load_si128, _mm512_add_epi64, _mm512_and_si512, _mm512_cvtepu16_epi64,
        _mm512_mul_epu32, _mm512_set1_epi64, _mm512_setzero_epi32, _mm512_slli_epi64,
        _mm512_srli_epi64, _mm512_store_epi64,
    };
    unsafe {
        let modulus = _mm512_set1_epi64(0x1FFFFFFFFFFFFFFF);
        let mut out_ptr =
            out as *mut [_] as *mut <RssShareGeneral<T1, U1> as Lift<Fp>>::Output as *mut i64;
        let mut x_ptrs = x
            .iter()
            .map(|x_vec| {
                (&x_vec[..] as *const [_] as *const RssShareGeneral<T1, U1> as *const i16)
                    .add(offset)
            })
            .collect::<Vec<_>>();
        let x_ptr_len = x_ptrs.len();
        for _ in (0..out.len()).step_by(8) {
            let mut sum = _mm512_setzero_epi32();

            for (i, x_ptr) in x_ptrs.iter_mut().enumerate() {
                let bytes = _mm_load_si128(*x_ptr as *const __m128i);
                let x = _mm512_cvtepu16_epi64(bytes);
                let p = _mm512_set1_epi64(std::mem::transmute_copy(&powers[i]));
                let p_high = _mm512_set1_epi64(
                    (std::mem::transmute_copy::<_, u64>(&powers[i]) >> 32) as i64,
                );

                // Multiplication, with special provision that x does not have a high

                // In the following analysis we assume x has 16 bits
                let r1 = _mm512_mul_epu32(x, p_high); // at most 48 bits
                let r3 = _mm512_mul_epu32(x, p); // at most 45 bits

                let h0 = _mm512_and_si512(_mm512_slli_epi64::<32>(r1), modulus); // 61 bits
                let rl = _mm512_add_epi64(r3, h0);

                let h1 = _mm512_srli_epi64::<29>(r1); // at most 16 bits
                let rl = _mm512_add_epi64(rl, h1); // rl can have at most 62 bits

                sum = _mm512_add_epi64(sum, rl);

                if i % 4 == 3 || i == x_ptr_len - 1 {
                    let rh = _mm512_srli_epi64::<61>(sum);
                    sum = _mm512_add_epi64(_mm512_and_si512(sum, modulus), rh);
                }

                *x_ptr = x_ptr.add(8);
            }

            _mm512_store_epi64(out_ptr, sum);
            out_ptr = out_ptr.add(8);
        }
    }
}

fn batch_and_lift_x<Fp: PrimeField, T1: FieldLike + Lift<Fp>, U1: FieldLike + Lift<Fp>>(
    out: &mut [<RssShareGeneral<T1, U1> as Lift<Fp>>::Output],
    x: &[AlignedVec<RssShareGeneral<T1, U1>, 64>],
    offset: usize,
    powers: &[Fp],
) {
    let mut start = 0;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f",))]
    if out.len() >= 8 && std::mem::size_of::<RssShareGeneral<T1, U1>>() == 1 {
        start = out.len() / 8 * 8;
        batch_and_lift_x_8_mersenne(&mut out[..start], x, offset, powers);
    } else if out.len() >= 8 && std::mem::size_of::<RssShareGeneral<T1, U1>>() == 2 {
        start = out.len() / 8 * 8;
        batch_and_lift_x_16_mersenne(&mut out[..start], x, offset, powers);
    }
    for (xs, power) in zip_eq(x, powers) {
        for (out, x) in zip(&mut out[start..], &xs[offset + start..]) {
            *out += x.lift() * *power;
        }
    }
}

fn batch_and_lift_x_par<Fp: PrimeField, T1: FieldLike + Lift<Fp>, U1: FieldLike + Lift<Fp>>(
    out: &mut [<RssShareGeneral<T1, U1> as Lift<Fp>>::Output],
    x: &[AlignedVec<RssShareGeneral<T1, U1>, 64>],
    powers: &[Fp],
) {
    let size = std::mem::size_of::<RssShareGeneral<T1, U1>>();
    if size == 0 {
        return;
    }

    // The inherent concurrency is 4
    let mut num_workers = rayon::current_num_threads() * 2 / 4;
    if num_workers == 0 {
        num_workers = 1;
    }
    let mut chunk_size = out.len() / num_workers;
    chunk_size = (chunk_size + 7) / 8;
    out.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, out)| {
            let start = i * chunk_size;
            batch_and_lift_x(out, x, start, powers);
        });
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f",))]
fn lift_y_8<Fp: PrimeField, T1: FieldLike + Lift<Fp>, U1: FieldLike + Lift<Fp>>(
    out: &mut [<RssShareGeneral<T1, U1> as Lift<Fp>>::Output],
    x: &[RssShareGeneral<T1, U1>],
) {
    use std::arch::x86_64::{__m128i, _mm_loadl_epi64, _mm512_cvtepu8_epi64, _mm512_store_epi64};
    unsafe {
        let mut out_ptr =
            out as *mut [_] as *mut <RssShareGeneral<T1, U1> as Lift<Fp>>::Output as *mut i64;
        let mut x_ptr = x as *const [_] as *const RssShareGeneral<T1, U1> as *const i8;
        for _ in (0..out.len()).step_by(8) {
            let bytes = _mm_loadl_epi64(x_ptr as *const __m128i);
            let x = _mm512_cvtepu8_epi64(bytes);

            _mm512_store_epi64(out_ptr, x);
            out_ptr = out_ptr.add(8);
            x_ptr = x_ptr.add(8);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f",))]
fn lift_y_16<Fp: PrimeField, T1: FieldLike + Lift<Fp>, U1: FieldLike + Lift<Fp>>(
    out: &mut [<RssShareGeneral<T1, U1> as Lift<Fp>>::Output],
    x: &[RssShareGeneral<T1, U1>],
) {
    use std::arch::x86_64::{__m128i, _mm_load_si128, _mm512_cvtepu16_epi64, _mm512_store_epi64};
    unsafe {
        let mut out_ptr =
            out as *mut [_] as *mut <RssShareGeneral<T1, U1> as Lift<Fp>>::Output as *mut i64;
        let mut x_ptr = x as *const [_] as *const RssShareGeneral<T1, U1> as *const i16;
        for _ in (0..out.len()).step_by(8) {
            let bytes = _mm_load_si128(x_ptr as *const __m128i);
            let x = _mm512_cvtepu16_epi64(bytes);

            _mm512_store_epi64(out_ptr, x);
            out_ptr = out_ptr.add(8);
            x_ptr = x_ptr.add(8);
        }
    }
}

fn lift_y<Fp: PrimeField, T1: FieldLike + Lift<Fp>, U1: FieldLike + Lift<Fp>>(
    out: &mut [<RssShareGeneral<T1, U1> as Lift<Fp>>::Output],
    x: &[RssShareGeneral<T1, U1>],
) {
    let mut start = 0;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f",))]
    if out.len() >= 8 && std::mem::size_of::<RssShareGeneral<T1, U1>>() == 1 {
        start = out.len() / 8 * 8;
        lift_y_8(&mut out[..start], x);
    } else if out.len() >= 8 && std::mem::size_of::<RssShareGeneral<T1, U1>>() == 2 {
        start = out.len() / 8 * 8;
        lift_y_16(&mut out[..start], x);
    }
    for (out, x) in zip(&mut out[start..], &x[start..]) {
        *out = x.lift();
    }
}

fn lift_y_par<Fp: PrimeField, T1: FieldLike + Lift<Fp>, U1: FieldLike + Lift<Fp>>(
    out: &mut [<RssShareGeneral<T1, U1> as Lift<Fp>>::Output],
    x: &[RssShareGeneral<T1, U1>],
) {
    let size = std::mem::size_of::<RssShareGeneral<T1, U1>>();
    if size == 0 {
        return;
    }

    // The inherent concurrency is 4
    let mut num_workers = rayon::current_num_threads() * 2 / 4;
    if num_workers == 0 {
        num_workers = 1;
    }
    let mut chunk_size = out.len() / num_workers;
    chunk_size = (chunk_size + 7) / 8;
    out.par_chunks_mut(chunk_size)
        .zip(x.par_chunks(chunk_size))
        .for_each(|(out, x)| {
            lift_y(out, x);
        });
}

fn batch_fp_instances<
    Fp: PrimeField,
    T1: FieldLike + Lift<Fp>,
    U1: FieldLike + Lift<Fp>,
    T2: FieldLike + Lift<Fp>,
    U2: FieldLike + Lift<Fp>,
    T3: FieldLike + Lift<Fp>,
    U3: FieldLike + Lift<Fp>,
>(
    (((x1, x2), (y1, y2), z), t, (t_x, t_y, t_z)): &(
        (
            (
                Vec<AlignedVec<RssShareGeneral<T1, U1>, 64>>,
                Vec<AlignedVec<RssShareGeneral<T2, U2>, 64>>,
            ),
            (
                AlignedVec<RssShareGeneral<T2, U2>, 64>,
                AlignedVec<RssShareGeneral<T1, U1>, 64>,
            ),
            Vec<RssShareGeneral<T3, U3>>,
        ),
        Vec<<RssShareGeneral<T3, U3> as Lift<Fp>>::Output>,
        (
            Vec<<RssShareGeneral<T1, U1> as Lift<Fp>>::Output>,
            Vec<<RssShareGeneral<T2, U2> as Lift<Fp>>::Output>,
            Vec<<RssShareGeneral<T3, U3> as Lift<Fp>>::Output>,
        ),
    ),
    r: Fp,
) -> (
    (
        AlignedVec<<RssShareGeneral<T1, U1> as Lift<Fp>>::Output, 64>,
        AlignedVec<<RssShareGeneral<T2, U2> as Lift<Fp>>::Output, 64>,
    ),
    (
        AlignedVec<<RssShareGeneral<T2, U2> as Lift<Fp>>::Output, 64>,
        AlignedVec<<RssShareGeneral<T1, U1> as Lift<Fp>>::Output, 64>,
    ),
    <RssShareGeneral<T3, U3> as Lift<Fp>>::Output,
) {
    let powers = generate_powers(r, x1.len() + t_x.len());
    let ((x1, x2), ((y1, y2), z)) = rayon::join(
        || {
            rayon::join(
                || {
                    let mut out_x1 = vec::from_elem_in(
                        RssShareGeneral::ZERO,
                        x1[0].len() + t_x.len(),
                        AlignedAllocator::<64>,
                    );
                    batch_and_lift_x_par(&mut out_x1[..x1[0].len()], &x1, &powers[..x1.len()]);
                    izip!(&mut out_x1[x1[0].len()..], t_x, &powers[x1.len()..])
                        .for_each(|(out, xs, gamma)| *out = *xs * *gamma);
                    out_x1
                },
                || {
                    let mut out_x2 = vec::from_elem_in(
                        RssShareGeneral::ZERO,
                        x2[0].len(),
                        AlignedAllocator::<64>,
                    );
                    batch_and_lift_x_par(&mut out_x2, &x2, &powers[..x2.len()]);
                    out_x2
                },
            )
        },
        || {
            rayon::join(
                || {
                    rayon::join(
                        || {
                            let mut out_y1 = vec::from_elem_in(
                                RssShareGeneral::ZERO,
                                y1.len() + t_y.len(),
                                AlignedAllocator::<64>,
                            );
                            lift_y_par(&mut out_y1[..y1.len()], &y1);
                            unsafe {
                                ptr::copy_nonoverlapping(
                                    &t_y[..] as *const [_]
                                        as *const <RssShareGeneral<T2, U2> as Lift<Fp>>::Output,
                                    &mut out_y1[y1.len()..] as *mut [_]
                                        as *mut <RssShareGeneral<T2, U2> as Lift<Fp>>::Output,
                                    t_y.len(),
                                );
                            }
                            out_y1
                        },
                        || {
                            let mut out_y2 = vec::from_elem_in(
                                RssShareGeneral::ZERO,
                                y2.len(),
                                AlignedAllocator::<64>,
                            );
                            lift_y_par(&mut out_y2, &y2);
                            out_y2
                        },
                    )
                },
                || {
                    izip!(z, &powers, t)
                        .map(|(z, power, t)| (z.lift() + *t) * *power)
                        .chain(
                            t_z.iter()
                                .zip(powers[z.len()..].iter())
                                .map(|(z, power)| *z * *power),
                        )
                        .sum()
                },
            )
        },
    );
    ((x1, x2), (y1, y2), z)
}

pub fn secret_share_two_party<F: Field, R: Rng + CryptoRng>(
    rng: &mut R,
    x: &[F],
) -> (
    Vec<RssShare<F>>,
    Vec<RssShareGeneral<F, Empty>>,
    Vec<RssShareGeneral<Empty, F>>,
) {
    let r = F::generate(rng, x.len());
    assert_eq!(r.len(), x.len());
    let (s1, (s2, s3)) = x
        .iter()
        .zip(r.iter())
        .map(|(value, r)| {
            let x1 = RssShare::from(*r, value.clone() - *r);
            let x2 = RssShareGeneral::from(value.clone() - *r, Empty);
            let x3 = RssShareGeneral::from(Empty, *r);
            (x1, (x2, x3))
        })
        .unzip();
    (s1, s2, s3)
}

#[inline]
fn compute_poly<
    F: Field,
    T1: FieldLike + Mul<F, Output = T1>,
    U1: FieldLike + Mul<F, Output = U1>,
>(
    x: &mut [RssShareGeneral<T1, U1>],
    r: F,
) {
    let mut i = 0;
    for k in 0..x.len() / 2 {
        x[k] = x[i] + (x[i + 1] - x[i]) * r;
        i += 2;
    }
    if x.len() % 2 == 1 {
        let last_idx = x.len() / 2;
        x[last_idx] = *x.last().unwrap() * (F::ONE - r);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512vbmi2",))]
pub fn compute_poly_dst_mersenne61<F, U>(dst: &mut [F], x: &[F], r: U) {
    use std::arch::x86_64::{
        _mm512_add_epi64, _mm512_and_si512, _mm512_cmpge_epi64_mask, _mm512_load_epi64,
        _mm512_mask_sub_epi64, _mm512_mul_epu32, _mm512_permutex2var_epi64, _mm512_set_epi64,
        _mm512_set1_epi64, _mm512_shrdi_epi64, _mm512_slli_epi64, _mm512_srli_epi64,
        _mm512_store_epi64, _mm512_sub_epi64,
    };
    debug_assert!(dst.len() % 8 == 0);

    unsafe {
        let modulus = _mm512_set1_epi64(0x1FFFFFFFFFFFFFFF);

        let r = _mm512_set1_epi64(std::mem::transmute_copy(&r));
        let r_high = _mm512_srli_epi64::<32>(r);

        let mut x_ptr = x as *const [F] as *const i64;
        let mut dst_ptr = dst as *mut [F] as *mut i64;
        let low_index = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
        let high_index = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        for _ in (0..dst.len()).step_by(8) {
            let x1 = _mm512_load_epi64(x_ptr);
            x_ptr = x_ptr.add(8);
            let x2 = _mm512_load_epi64(x_ptr);
            let x_low = _mm512_permutex2var_epi64(x1, low_index, x2);
            let x_high = _mm512_permutex2var_epi64(x1, high_index, x2);

            // Compute (high - low) * r
            let a = _mm512_add_epi64(_mm512_sub_epi64(x_high, x_low), modulus);
            let a_high = _mm512_srli_epi64::<32>(a);

            // a was 62 bits, b was 61 bits

            let r0 = _mm512_mul_epu32(a_high, r_high); // 59 bits
            let r1 = _mm512_mul_epu32(a_high, r); // 62 bits
            let r2 = _mm512_mul_epu32(a, r_high); // 62 bits
            let r3 = _mm512_mul_epu32(a, r); // 64 bits

            let r1 = _mm512_add_epi64(r1, r2); // 63 bits

            let rl = _mm512_and_si512(r3, modulus); // 61 bits
            let rh = _mm512_shrdi_epi64::<61>(r3, r0); // 62 bits

            let rl = _mm512_add_epi64(rl, rh); // 61 + 62 bits

            let h0 = _mm512_and_si512(_mm512_slli_epi64::<32>(r1), modulus); // 61 bits
            let rl = _mm512_add_epi64(rl, h0); // 61 + 62 + 61 bits

            let h1 = _mm512_srli_epi64::<29>(r1); // 34 bits
            let rl = _mm512_add_epi64(rl, h1); // 61 + 62 + 61 + 34 bits

            // low + (high - low) * r
            // 62 + (61 + 61) + 61 + 34 bits = 63 + 61 + 34 bits < 64 bits
            let rl = _mm512_add_epi64(rl, x_low);

            // Reduce. Doing it twice is always fine for 64 bits or fewer...
            let rh = _mm512_srli_epi64::<61>(rl);
            let rl = _mm512_add_epi64(_mm512_and_si512(rl, modulus), rh);

            // The second time we can also save an instruction
            let mask = _mm512_cmpge_epi64_mask(rl, modulus);
            let rl = _mm512_mask_sub_epi64(rl, mask, rl, modulus);

            _mm512_store_epi64(dst_ptr, rl);

            dst_ptr = dst_ptr.add(8);
            x_ptr = x_ptr.add(8);
        }
    }
}

#[inline]
fn compute_poly_dst<
    F: Field,
    T1: FieldLike + Mul<F, Output = T1>,
    U1: FieldLike + Mul<F, Output = U1>,
>(
    dst: &mut [RssShareGeneral<T1, U1>],
    x: &[RssShareGeneral<T1, U1>],
    r: F,
) {
    let mut start = 0;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512vbmi2",))]
    if dst.len() >= 8 && std::mem::size_of::<RssShareGeneral<T1, U1>>() == 8 {
        start = dst.len() / 8 * 8;
        if x.len() < start * 2 {
            // This can be the case if x's length is odd
            start -= 8;
        }
        compute_poly_dst_mersenne61(&mut dst[..start], &x[..start * 2], r);
    }

    // debug_assert_eq!(2 * dst.len(), x.len());
    let mut i = start * 2;
    let max_idx = if x.len() % 2 == 1 {
        dst.len() - 1
    } else {
        dst.len()
    };
    for k in start..max_idx {
        dst[k] = x[i] + (x[i + 1] - x[i]) * r;
        i += 2;
    }
    if x.len() % 2 == 1 {
        *dst.last_mut().unwrap() = *x.last().unwrap() * (F::ONE - r);
    }
}

fn verify_dot_product_compute_h<
    F: PrimeField + Debug,
    T1: Debug + FieldLike + Mul<F, Output = T1> + MulAssign<F>,
    U1: Debug + FieldLike + Mul<F, Output = U1> + MulAssign<F>,
    T2: Debug + FieldLike + Mul<F, Output = T2> + MulAssign<F>,
    U2: Debug + FieldLike + Mul<F, Output = U2> + MulAssign<F>,
    T3: Debug + FieldLike + Mul<F, Output = T3> + Sub<F, Output = F>,
    U3: Debug + FieldLike + Mul<F, Output = U3> + Sub<F, Output = F>,
>(
    ((x1_vec, x2_vec), (y1_vec, y2_vec), _z): &(
        (
            AlignedVec<RssShareGeneral<T1, U1>, 64>,
            AlignedVec<RssShareGeneral<T2, U2>, 64>,
        ),
        (
            AlignedVec<RssShareGeneral<T2, U2>, 64>,
            AlignedVec<RssShareGeneral<T1, U1>, 64>,
        ),
        RssShareGeneral<T3, U3>,
    ),
    chunk_size_1: usize,
    chunk_size_2: usize,
) -> [F; 2]
where
    F: GenericInnerProduct<T1, U1, T2, U2> + GenericInnerProduct<T2, U2, T1, U1> + InnerProduct,
{
    // let inner_prod_time = Instant::now();
    let n2_val = if x2_vec.len() == 1 {
        <F as GenericInnerProduct<_, _, _, _>>::weak_inner_product(&[x2_vec[0]], &[y2_vec[0]])
    } else {
        F::ZERO
    };

    let mut hs = [F::ZERO; 2];
    if chunk_size_1 == 1 {
        hs[0] = F::weak_inner_product_2(&x1_vec[1..], &y1_vec[1..]);
        hs[1] = F::weak_inner_product_3(&x1_vec, &y1_vec);
    } else {
        let mut h0 = F::ZERO;
        let mut h1 = F::ZERO;
        rayon::scope(|scope| {
            scope.spawn(|_| {
                h0 = x1_vec[1..]
                    .par_chunks(chunk_size_1)
                    .zip_eq(y1_vec[1..].par_chunks(chunk_size_1))
                    .map(|(x, y)| F::weak_inner_product_2(x, y))
                    .reduce(|| F::ZERO, |sum, v| sum + v);
            });
            scope.spawn(|_| {
                h1 = x1_vec
                    .par_chunks(chunk_size_1)
                    .zip_eq(y1_vec.par_chunks(chunk_size_1))
                    .map(|(x, y)| F::weak_inner_product_3(x, y))
                    .reduce(|| F::ZERO, |sum, v| sum + v);
            });
        });
        hs[0] = h0;
        hs[1] = h1;
    }

    if x2_vec.len() == 1 {
        hs[1] += n2_val;
    } else if chunk_size_2 == 1 {
        hs[0] += F::weak_inner_product_2(&x2_vec[1..], &y2_vec[1..]);
        hs[1] += F::weak_inner_product_3(&x2_vec, &y2_vec);
    } else {
        let mut h0 = F::ZERO;
        let mut h1 = F::ZERO;
        rayon::scope(|scope| {
            scope.spawn(|_| {
                h0 = x2_vec[1..]
                    .par_chunks(chunk_size_2)
                    .zip_eq(y2_vec[1..].par_chunks(chunk_size_2))
                    .map(|(x, y)| F::weak_inner_product_2(x, y))
                    .reduce(|| F::ZERO, |sum, v| sum + v);
            });
            scope.spawn(|_| {
                h1 = x2_vec
                    .par_chunks(chunk_size_2)
                    .zip_eq(y2_vec.par_chunks(chunk_size_2))
                    .map(|(x, y)| F::weak_inner_product_3(x, y))
                    .reduce(|| F::ZERO, |sum, v| sum + v);
            });
        });
        hs[0] += h0;
        hs[1] += h1;
    }

    hs
}

fn verify_dot_product_reduce_poly<
    F: PrimeField + Debug,
    T1: Debug + FieldLike + Mul<F, Output = T1> + MulAssign<F>,
    U1: Debug + FieldLike + Mul<F, Output = U1> + MulAssign<F>,
    T2: Debug + FieldLike + Mul<F, Output = T2> + MulAssign<F>,
    U2: Debug + FieldLike + Mul<F, Output = U2> + MulAssign<F>,
    T3: Debug + FieldLike + Mul<F, Output = T3> + Sub<F, Output = F>,
    U3: Debug + FieldLike + Mul<F, Output = U3> + Sub<F, Output = F>,
>(
    ((mut x1_vec, mut x2_vec), (mut y1_vec, mut y2_vec), _z): (
        (
            AlignedVec<RssShareGeneral<T1, U1>, 64>,
            AlignedVec<RssShareGeneral<T2, U2>, 64>,
        ),
        (
            AlignedVec<RssShareGeneral<T2, U2>, 64>,
            AlignedVec<RssShareGeneral<T1, U1>, 64>,
        ),
        RssShareGeneral<T3, U3>,
    ),
    chunk_size_1: usize,
    chunk_size_2: usize,
    r: F,
) -> (
    (
        AlignedVec<RssShareGeneral<T1, U1>, 64>,
        AlignedVec<RssShareGeneral<T2, U2>, 64>,
    ),
    (
        AlignedVec<RssShareGeneral<T2, U2>, 64>,
        AlignedVec<RssShareGeneral<T1, U1>, 64>,
    ),
) {
    // let poly_time = Instant::now();
    // Compute polynomials
    let (f1, g1) = if chunk_size_1 == 1 {
        compute_poly(&mut x1_vec, r);
        compute_poly(&mut y1_vec, r);
        x1_vec.truncate((x1_vec.len() + 1) / 2);
        y1_vec.truncate((y1_vec.len() + 1) / 2);
        (x1_vec, y1_vec)
    } else {
        let mut f1 = vec::from_elem_in(
            RssShareGeneral::ZERO,
            (x1_vec.len() + 1) / 2,
            AlignedAllocator::<64>,
        );
        let mut g1 = vec::from_elem_in(
            RssShareGeneral::ZERO,
            (x1_vec.len() + 1) / 2,
            AlignedAllocator::<64>,
        );

        rayon::scope(|scope| {
            scope.spawn(|_| {
                f1.par_chunks_mut(chunk_size_1 / 2)
                    .zip_eq(x1_vec.par_chunks(chunk_size_1))
                    .for_each(|(dst, x)| {
                        compute_poly_dst(dst, x, r);
                    });
            });

            scope.spawn(|_| {
                g1.par_chunks_mut(chunk_size_1 / 2)
                    .zip_eq(y1_vec.par_chunks(chunk_size_1))
                    .for_each(|(dst, y)| {
                        compute_poly_dst(dst, y, r);
                    });
            });
        });
        (f1, g1)
    };

    let (f2, g2) = if x2_vec.len() == 1 {
        x2_vec[0] *= F::ONE - r;
        y2_vec[0] *= F::ONE - r;
        (x2_vec, y2_vec)
    } else if chunk_size_2 == 1 {
        compute_poly(&mut x2_vec, r);
        compute_poly(&mut y2_vec, r);
        x2_vec.truncate((x2_vec.len() + 1) / 2);
        y2_vec.truncate((y2_vec.len() + 1) / 2);
        (x2_vec, y2_vec)
    } else {
        let mut f2 = vec::from_elem_in(
            RssShareGeneral::ZERO,
            (x2_vec.len() + 1) / 2,
            AlignedAllocator::<64>,
        );
        let mut g2 = vec::from_elem_in(
            RssShareGeneral::ZERO,
            (x2_vec.len() + 1) / 2,
            AlignedAllocator::<64>,
        );

        rayon::scope(|scope| {
            scope.spawn(|_| {
                f2.par_chunks_mut(chunk_size_2 / 2)
                    .zip_eq(x2_vec.par_chunks(chunk_size_2))
                    .for_each(|(dst, x)| {
                        compute_poly_dst(dst, x, r);
                    });
            });

            scope.spawn(|_| {
                g2.par_chunks_mut(chunk_size_2 / 2)
                    .zip_eq(y2_vec.par_chunks(chunk_size_2))
                    .for_each(|(dst, y)| {
                        compute_poly_dst(dst, y, r);
                    });
            });
        });
        (f2, g2)
    };
    ((f1, f2), (g1, g2))
}

fn check_triples<F: PrimeField + Debug + InnerProduct>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    instances: [(
        (RssShare<F>, RssShare<F>),
        (RssShare<F>, RssShare<F>),
        RssShare<F>,
    ); 3],
) -> MpcResult<bool> {
    let x_prime = party.generate_random::<F>(6);
    let x = instances
        .iter()
        .flat_map(|((x1, x2), _, _)| [*x1, *x2])
        .collect::<Vec<_>>();
    let y = instances
        .iter()
        .flat_map(|(_, (y1, y2), _)| [*y1, *y2])
        .collect::<Vec<_>>();
    let z_p = zip_eq(&x_prime, &y)
        .map(|(x, y)| F::weak_inner_product(&[*x], &[*y]))
        .collect::<Vec<_>>();
    let z_p = ss_to_rss_shares(party, &z_p)?;

    let t: F = coin_flip(party, context)?;

    let rho = zip_eq(&x, &x_prime)
        .map(|(x, x_p)| *x + *x_p * t)
        .collect::<Vec<_>>();
    let rho = reconstruct_slice(party, context, &rho)?;

    let verify = izip!(
        &instances,
        z_p.as_chunks::<2>().0,
        y.as_chunks::<2>().0,
        rho.as_chunks::<2>().0
    )
    .map(|(instance, z_p, y, rho)| {
        let z = instance.2;
        z + z_p[0] * t + z_p[1] * t - y[0] * rho[0] - y[1] * rho[1]
    })
    .collect::<Vec<_>>();
    reconstruct_slice(party, context, &verify).map(|x| x.iter().all(|x| x.is_zero()))
}

fn verify_dot_product_batched<F: PrimeField + Debug>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    self_instance: (
        (
            AlignedVec<RssShareGeneral<Empty, F>, 64>,
            AlignedVec<RssShareGeneral<F, Empty>, 64>,
        ),
        (
            AlignedVec<RssShareGeneral<F, Empty>, 64>,
            AlignedVec<RssShareGeneral<Empty, F>, 64>,
        ),
        RssShare<F>,
    ),
    next_instance: (
        (
            AlignedVec<RssShare<Empty>, 64>,
            AlignedVec<RssShareGeneral<Empty, F>, 64>,
        ),
        (
            AlignedVec<RssShareGeneral<Empty, F>, 64>,
            AlignedVec<RssShare<Empty>, 64>,
        ),
        RssShare<F>,
    ),
    prev_instance: (
        (
            AlignedVec<RssShareGeneral<F, Empty>, 64>,
            AlignedVec<RssShare<Empty>, 64>,
        ),
        (
            AlignedVec<RssShare<Empty>, 64>,
            AlignedVec<RssShareGeneral<F, Empty>, 64>,
        ),
        RssShare<F>,
    ),
) -> MpcResult<bool>
where
    F: InnerProduct,
{
    let mut n1 = self_instance.0.0.len();
    let mut n2 = self_instance.0.1.len();
    debug_assert!(n1 >= n2);

    if n1 == 1 {
        let self_instance = (
            (self_instance.0.0[0].into(), self_instance.0.1[0].into()),
            (self_instance.1.0[0].into(), self_instance.1.1[0].into()),
            self_instance.2,
        );
        let next_instance = (
            (next_instance.0.0[0].into(), next_instance.0.1[0].into()),
            (next_instance.1.0[0].into(), next_instance.1.1[0].into()),
            next_instance.2,
        );
        let prev_instance = (
            (prev_instance.0.0[0].into(), prev_instance.0.1[0].into()),
            (prev_instance.1.0[0].into(), prev_instance.1.1[0].into()),
            prev_instance.2,
        );
        let instances = match party.i {
            0 => [self_instance, next_instance, prev_instance],
            1 => [prev_instance, self_instance, next_instance],
            2 => [next_instance, prev_instance, self_instance],
            3.. => panic!("Only 3PC is supported"),
        };
        let result = check_triples(party, context, instances)?;
        return Ok(result);
    }
    if n1 % 2 == 1 {
        n1 += 1;
    }
    if n2 % 2 == 1 && n2 != 1 {
        n2 += 1;
    }

    let num_workers = rayon::current_num_threads();
    let multi_threading_1 = num_workers > 1 && n1 >= (1 << 13);
    let multi_threading_2 = num_workers > 1 && n2 >= (1 << 13);
    let mut chunk_size_1 = 1;
    let mut chunk_size_2 = 1;
    // make sure chunk size is a multiple of 16 (for alignment with AVX512)
    if multi_threading_1 {
        chunk_size_1 = (n1 + num_workers - 1) / num_workers;
        chunk_size_1 = (chunk_size_1 + 15) / 16 * 16;
    }
    if multi_threading_2 {
        chunk_size_2 = (n2 + num_workers - 1) / num_workers;
        chunk_size_2 = (chunk_size_2 + 15) / 16 * 16;
    }

    let h_self = verify_dot_product_compute_h(&self_instance, chunk_size_1, chunk_size_2);

    let h_next = [F::ZERO; 2];
    let h_prev = [F::ZERO; 2];
    let h = match party.party_id() {
        0 => vec![h_self, h_next, h_prev],
        1 => vec![h_prev, h_self, h_next],
        2 => vec![h_next, h_prev, h_self],
        3.. => panic!("Only 3PC is supported"),
    }
    .concat();
    let h = ss_to_rss_shares(party, &h)?;
    let (h1_self, h2_self, h1_next, h2_next, h1_prev, h2_prev) = match party.i {
        0 => (h[0], h[1], h[2], h[3], h[4], h[5]),
        1 => (h[2], h[3], h[4], h[5], h[0], h[1]),
        2 => (h[4], h[5], h[0], h[1], h[2], h[3]),
        3.. => panic!("Only 3PC is supported"),
    };

    let r = coin_flip(party, context)?;
    // For large F this is very unlikely
    debug_assert!(r != F::ZERO && r != F::ONE);

    let h0_self = self_instance.2 - h1_self;
    let h0_next = next_instance.2 - h1_next;
    let h0_prev = prev_instance.2 - h1_prev;
    let ((x_self, y_self), ((x_next, y_next), (x_prev, y_prev))) = rayon::join(
        || verify_dot_product_reduce_poly(self_instance, chunk_size_1, chunk_size_2, r),
        || {
            rayon::join(
                || verify_dot_product_reduce_poly(next_instance, chunk_size_1, chunk_size_2, r),
                || verify_dot_product_reduce_poly(prev_instance, chunk_size_1, chunk_size_2, r),
            )
        },
    );

    // let poly_time = poly_time.elapsed();

    let hr_self = lagrange_deg2(&h0_self, &h1_self, &h2_self, r);
    let hr_next = lagrange_deg2(&h0_next, &h1_next, &h2_next, r);
    let hr_prev = lagrange_deg2(&h0_prev, &h1_prev, &h2_prev, r);
    // println!("[vfy-dp-opt] n={}, inner_prod_time={}s, ss_rss_time={}s, coin_flip_time={}s, poly_time={}s", n, inner_prod_time.as_secs_f32(), ss_rss_time.as_secs_f32(), coin_flip_time.as_secs_f32(), poly_time.as_secs_f32());
    verify_dot_product_batched(
        party,
        context,
        (x_self, y_self, hr_self),
        (x_next, y_next, hr_next),
        (x_prev, y_prev, hr_prev),
    )
}

fn verify_dot_product_opt_high_degree<
    const DEGREE: usize,
    F: PrimeField + Debug,
    T1: Debug + FieldLike + Mul<F, Output = T1>,
    U1: Debug + FieldLike + Mul<F, Output = U1>,
    T2: Debug + FieldLike + Mul<F, Output = T2>,
    U2: Debug + FieldLike + Mul<F, Output = U2>,
    T3: Debug + FieldLike + Mul<F, Output = T3> + Sub<F, Output = F>,
    U3: Debug + FieldLike + Mul<F, Output = U3> + Sub<F, Output = F>,
>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    ((mut x1_vec, mut x2_vec), (mut y1_vec, mut y2_vec), z): (
        (Vec<RssShareGeneral<T1, U1>>, Vec<RssShareGeneral<T2, U2>>),
        (Vec<RssShareGeneral<T2, U2>>, Vec<RssShareGeneral<T1, U1>>),
        RssShareGeneral<T3, U3>,
    ),
) -> MpcResult<bool>
where
    F: InnerProduct + LagrangeInterExtrapolate<DEGREE>,
    T1: Mul<T2>,
    T1: Add<U1>,
    T2: Add<U2>,
    <T1 as Add<U1>>::Output: Mul<<T2 as Add<U2>>::Output>,
    <<T1 as Add<U1>>::Output as Mul<<T2 as Add<U2>>::Output>>::Output: Sub<<T1 as Mul<T2>>::Output>,
    <<<T1 as Add<U1>>::Output as Mul<<T2 as Add<U2>>::Output>>::Output as Sub<
        <T1 as Mul<T2>>::Output,
    >>::Output: Add<F, Output = F>,
    RssShareGeneral<T1, U1>: Into<RssShare<F>>,
    RssShareGeneral<T2, U2>: Into<RssShare<F>>,
    RssShareGeneral<T3, U3>: Into<RssShare<F>>,
{
    let n1 = x1_vec.len();
    // println!("n = {}", n);
    debug_assert_eq!(n1, y1_vec.len());

    let n2 = x2_vec.len();
    debug_assert!(n1 >= n2);
    // debug_assert!(n & (n - 1) == 0 && n != 0);

    if n1 == 1 {
        return check_triple_double(
            party,
            context,
            x1_vec[0].into(),
            x2_vec[0].into(),
            y1_vec[0].into(),
            y2_vec[0].into(),
            z.into(),
        );
    }

    let reduced_size_1 = (x1_vec.len() + DEGREE) / (DEGREE + 1);
    if x1_vec.len() < reduced_size_1 * (DEGREE + 1) {
        x1_vec.resize(reduced_size_1 * (DEGREE + 1), RssShareGeneral::ZERO);
    }
    if y1_vec.len() < reduced_size_1 * (DEGREE + 1) {
        y1_vec.resize(reduced_size_1 * (DEGREE + 1), RssShareGeneral::ZERO);
    }

    let reduced_size_2 = (x2_vec.len() + DEGREE) / (DEGREE + 1);
    if x2_vec.len() < reduced_size_2 * (DEGREE + 1) {
        x2_vec.resize(reduced_size_2 * (DEGREE + 1), RssShareGeneral::ZERO);
    }
    if y2_vec.len() < reduced_size_2 * (DEGREE + 1) {
        y2_vec.resize(reduced_size_2 * (DEGREE + 1), RssShareGeneral::ZERO);
    }

    let mut x1_vec_high_evals = vec![RssShareGeneral::ZERO; DEGREE * reduced_size_1];
    let mut x2_vec_high_evals = vec![RssShareGeneral::ZERO; DEGREE * reduced_size_2];
    let mut y1_vec_high_evals = vec![RssShareGeneral::ZERO; DEGREE * reduced_size_1];
    let mut y2_vec_high_evals = vec![RssShareGeneral::ZERO; DEGREE * reduced_size_2];
    rayon::join(
        || {
            rayon::join(
                || {
                    x1_vec
                        .par_chunks_exact(DEGREE + 1)
                        .zip_eq(x1_vec_high_evals.par_chunks_exact_mut(DEGREE))
                        .for_each(|(x_vec, out)| F::extrapolate(x_vec, out))
                },
                || {
                    y1_vec
                        .par_chunks_exact(DEGREE + 1)
                        .zip_eq(y1_vec_high_evals.par_chunks_exact_mut(DEGREE))
                        .for_each(|(y_vec, out)| F::extrapolate(y_vec, out))
                },
            )
        },
        || {
            rayon::join(
                || {
                    x2_vec
                        .par_chunks_exact(DEGREE + 1)
                        .zip_eq(x2_vec_high_evals.par_chunks_exact_mut(DEGREE))
                        .for_each(|(x_vec, out)| F::extrapolate(x_vec, out))
                },
                || {
                    y2_vec
                        .par_chunks_exact(DEGREE + 1)
                        .zip_eq(y2_vec_high_evals.par_chunks_exact_mut(DEGREE))
                        .for_each(|(y_vec, out)| F::extrapolate(y_vec, out))
                },
            )
        },
    );

    // let inner_prod_time = Instant::now();
    let hs = (
        x1_vec.par_chunks_exact(DEGREE + 1),
        y1_vec.par_chunks_exact(DEGREE + 1),
        x1_vec_high_evals.par_chunks_exact(DEGREE),
        y1_vec_high_evals.par_chunks_exact(DEGREE),
    )
        .into_par_iter()
        .chain((
            y2_vec.par_chunks_exact(DEGREE + 1),
            x2_vec.par_chunks_exact(DEGREE + 1),
            y2_vec_high_evals.par_chunks_exact(DEGREE),
            x2_vec_high_evals.par_chunks_exact(DEGREE),
        ))
        .fold(
            || vec![F::ZERO; 2 * DEGREE],
            |mut acc, (x, y, x_high, y_high)| {
                for i in 0..2 * DEGREE {
                    let (x, y) = if i < DEGREE {
                        (x[i + 1], y[i + 1])
                    } else {
                        (x_high[i - DEGREE], y_high[i - DEGREE])
                    };
                    acc[i] = (x.si + x.sii) * (y.si + y.sii) - x.si * y.si + acc[i];
                }
                acc
            },
        )
        .reduce(
            || vec![F::ZERO; 2 * DEGREE],
            |mut sum, v| {
                for (acc, x) in zip_eq(&mut sum, &v) {
                    *acc += *x;
                }
                sum
            },
        );

    // let inner_prod_time = inner_prod_time.elapsed();
    // let ss_rss_time = Instant::now();
    let mut h = ss_to_rss_shares(party, &hs)?;
    // let ss_rss_time = ss_rss_time.elapsed();
    let h0 = z - h.iter().take(DEGREE).cloned().sum::<RssShare<F>>();
    h.insert(0, h0);
    // let coin_flip_time = Instant::now();
    // Coin flip
    let r: F = coin_flip(party, context)?;
    // For large F this is very unlikely
    debug_assert!(r != F::ZERO && r != F::ONE);
    // let coin_flip_time = coin_flip_time.elapsed();

    // let poly_time = Instant::now();
    // Compute polynomials
    let mut f1 = vec![RssShareGeneral::ZERO; reduced_size_1];
    let mut f2 = vec![RssShareGeneral::ZERO; reduced_size_2];
    let mut g1 = vec![RssShareGeneral::ZERO; reduced_size_1];
    let mut g2 = vec![RssShareGeneral::ZERO; reduced_size_2];
    rayon::join(
        || {
            rayon::join(
                || {
                    f1.par_iter_mut()
                        .zip_eq(x1_vec.par_chunks_exact(DEGREE + 1))
                        .for_each(|(dst, x)| *dst = F::interpolate(x, r))
                },
                || {
                    g1.par_iter_mut()
                        .zip_eq(y1_vec.par_chunks_exact(DEGREE + 1))
                        .for_each(|(dst, x)| *dst = F::interpolate(x, r))
                },
            )
        },
        || {
            rayon::join(
                || {
                    f2.par_iter_mut()
                        .zip_eq(x2_vec.par_chunks_exact(DEGREE + 1))
                        .for_each(|(dst, x)| *dst = F::interpolate(x, r))
                },
                || {
                    g2.par_iter_mut()
                        .zip_eq(y2_vec.par_chunks_exact(DEGREE + 1))
                        .for_each(|(dst, x)| *dst = F::interpolate(x, r))
                },
            )
        },
    );

    // let poly_time = poly_time.elapsed();
    let hr = F::interpolate_target(&h, r);
    // println!("[vfy-dp-opt] n={}, inner_prod_time={}s, ss_rss_time={}s, coin_flip_time={}s, poly_time={}s", n, inner_prod_time.as_secs_f32(), ss_rss_time.as_secs_f32(), coin_flip_time.as_secs_f32(), poly_time.as_secs_f32());
    verify_dot_product_opt_high_degree::<DEGREE, _, _, _, _, _, _, _>(
        party,
        context,
        ((f1, f2), (g1, g2), hr),
    )
}

pub trait GenericInnerProduct<T1, U1, T2, U2> {
    fn weak_inner_product(a: &[RssShareGeneral<T1, U1>], b: &[RssShareGeneral<T2, U2>]) -> Self;
    fn weak_inner_product_2(a: &[RssShareGeneral<T1, U1>], b: &[RssShareGeneral<T2, U2>]) -> Self;
    fn weak_inner_product_3(a: &[RssShareGeneral<T1, U1>], b: &[RssShareGeneral<T2, U2>]) -> Self;
}

impl<
    Fp: PrimeField,
    T1: FieldLike + Mul<Fp, Output = T1>,
    U1: FieldLike + Mul<Fp, Output = U1>,
    T2: FieldLike + Mul<Fp, Output = T2>,
    U2: FieldLike + Mul<Fp, Output = U2>,
> GenericInnerProduct<T1, U1, T2, U2> for Fp
where
    T1: Mul<T2>,
    T1: Add<U1>,
    T2: Add<U2>,
    <T1 as Add<U1>>::Output: Mul<<T2 as Add<U2>>::Output>,
    <<T1 as Add<U1>>::Output as Mul<<T2 as Add<U2>>::Output>>::Output: Sub<<T1 as Mul<T2>>::Output>,
    <<<T1 as Add<U1>>::Output as Mul<<T2 as Add<U2>>::Output>>::Output as Sub<
        <T1 as Mul<T2>>::Output,
    >>::Output: Add<Fp, Output = Fp>,
{
    fn weak_inner_product(a: &[RssShareGeneral<T1, U1>], b: &[RssShareGeneral<T2, U2>]) -> Fp {
        a.iter().zip(b).fold(Fp::ZERO, |sum, (x, y)| {
            (x.si + x.sii) * (y.si + y.sii) - x.si * y.si + sum
        })
    }

    fn weak_inner_product_2(a: &[RssShareGeneral<T1, U1>], b: &[RssShareGeneral<T2, U2>]) -> Fp {
        a.iter().zip(b).step_by(2).fold(Fp::ZERO, |sum, (x, y)| {
            (x.si + x.sii) * (y.si + y.sii) - x.si * y.si + sum
        })
    }

    fn weak_inner_product_3(a: &[RssShareGeneral<T1, U1>], b: &[RssShareGeneral<T2, U2>]) -> Fp {
        let ret = a
            .chunks_exact(2)
            .zip(b.chunks_exact(2))
            .fold(Fp::ZERO, |sum, (x, y)| {
                let x = x[0] + (x[1] - x[0]) * Fp::TWO;
                let y = y[0] + (y[1] - y[0]) * Fp::TWO;
                (x.si + x.sii) * (y.si + y.sii) - x.si * y.si + sum
            });
        if a.len() % 2 == 1 {
            let x = a.last().unwrap();
            let y = b.last().unwrap();
            ((x.si + x.sii) * (y.si + y.sii) - x.si * y.si) + ret
        } else {
            ret
        }
    }
}

/// Protocol 1 CheckTriple
fn check_triple<F: Debug + Field + DigestExt>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    x: RssShare<F>,
    y: RssShare<F>,
    z: RssShare<F>,
) -> MpcResult<bool>
where
    F: InnerProduct,
{
    // Generate RSS sharing of random value
    let x_prime = party.generate_random(1)[0];
    let z_prime = weak_mult(party, &x_prime, &y)?;
    let t: F = coin_flip(party, context)?;
    let rho = reconstruct(party, context, x + x_prime * t)?;
    reconstruct(party, context, z + z_prime * t - y * rho).map(|x| x.is_zero())
}

fn check_triple_double<F: Debug + Field + DigestExt>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    x1: RssShare<F>,
    x2: RssShare<F>,
    y1: RssShare<F>,
    y2: RssShare<F>,
    z: RssShare<F>,
) -> MpcResult<bool>
where
    F: InnerProduct,
{
    // Generate RSS sharing of random value
    let x_prime = party.generate_random(2);
    let (x1_p, x2_p) = (x_prime[0], x_prime[1]);
    let z1_p = weak_mult(party, &x1_p, &y1)?;
    let z2_p = weak_mult(party, &x2_p, &y2)?;
    let t: F = coin_flip(party, context)?;
    let rho1 = reconstruct(party, context, x1 + x1_p * t)?;
    let rho2 = reconstruct(party, context, x2 + x2_p * t)?;
    reconstruct(
        party,
        context,
        z + z1_p * t + z2_p * t - y1 * rho1 - y2 * rho2,
    )
    .map(|x| x.is_zero())
}

/// Shared lagrange evaluation of the polynomial h at position x for given (shared) points h(0), h(1), h(2)
#[inline]
fn lagrange_deg2<
    T1: Field + Mul<F, Output = T1>,
    U1: Field + Mul<F, Output = U1>,
    F: PrimeField,
>(
    h0: &RssShareGeneral<T1, U1>,
    h1: &RssShareGeneral<T1, U1>,
    h2: &RssShareGeneral<T1, U1>,
    x: F,
) -> RssShareGeneral<T1, U1> {
    // Lagrange weights
    // w0^-1 = (1-0)*(2-0) = 1*2 = 2
    let w0 = F::TWO.inverse();
    // w1^-1 = (0-1)*(2-1) = 1*(2-1) = (2-1) = 2+1
    let w1 = -F::ONE;
    // w2^-1 = (0-2)*(1-2) = 2*(1+2) = 2 * (2+1)
    let w2 = w0;
    let l0 = w0 * (x - F::ONE) * (x - F::TWO);
    let l1 = w1 * x * (x - F::TWO);
    let l2 = w2 * x * (x - F::ONE);
    // Lagrange interpolation
    (*h0) * l0 + (*h1) * l1 + (*h2) * l2
}

pub fn reconstruct<F: Field + DigestExt>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    rho: RssShare<F>,
) -> MpcResult<F> {
    party
        .open_rss(context, slice::from_ref(&rho.si), slice::from_ref(&rho.sii))
        .map(|v| v[0])
}

pub fn reconstruct_slice<F: Field + DigestExt>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    rho: &[RssShare<F>],
) -> MpcResult<Vec<F>> {
    let share_i = rho.iter().map(|x| x.si).collect::<Vec<_>>();
    let share_ii = rho.iter().map(|x| x.sii).collect::<Vec<_>>();
    party.open_rss(context, &share_i, &share_ii)
}

/// Coin flip protocol returns a random value in F
///
/// Generates a sharing of a random value that is then reconstructed globally.
fn coin_flip<F: Field + DigestExt>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
) -> MpcResult<F> {
    let r: RssShare<F> = party.generate_random(1)[0];
    reconstruct(party, context, r)
}

/// Coin flip protocol returns a n random values in F
///
/// Generates a sharing of a n random values that is then reconstructed globally.
fn coin_flip_n<F: Field + DigestExt>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    n: usize,
) -> MpcResult<Vec<F>> {
    let (r_i, r_ii): (Vec<_>, Vec<_>) = party
        .generate_random::<F>(n)
        .into_iter()
        .map(|rss| (rss.si, rss.sii))
        .unzip();
    party.open_rss(context, &r_i, &r_ii)
}

/// Computes the components wise multiplication of replicated shared x and y.
fn weak_mult<F: Field + Copy + Sized>(
    party: &mut MainParty,
    x: &RssShare<F>,
    y: &RssShare<F>,
) -> MpcResult<RssShare<F>>
where
    F: InnerProduct,
{
    // Compute a sum sharing of x*y
    let zs = F::weak_inner_product(&[*x], &[*y]);
    single_ss_to_rss_shares(party, zs)
}

/// Converts a vector of sum sharings into a replicated sharing
#[inline]
pub fn ss_to_rss_shares<F: Field + Copy + Sized>(
    party: &mut MainParty,
    sum_shares: &[F],
) -> MpcResult<RssShareVec<F>> {
    let n = sum_shares.len();
    let alphas = party.generate_alpha::<F>(n);
    let s_i: Vec<F> = sum_shares.iter().zip(alphas).map(|(s, a)| *s + a).collect();
    let mut s_ii = vec![F::ZERO; n];
    party.send_field_slice(Direction::Previous, &s_i);
    party
        .receive_field_slice(Direction::Next, &mut s_ii)
        .rcv()?;
    let res: RssShareVec<F> = s_ii
        .iter()
        .zip(s_i)
        .map(|(sii, si)| RssShare::from(si, *sii))
        .collect();
    Ok(res)
}

/// Converts a sum sharing into a replicated sharing
#[inline]
fn single_ss_to_rss_shares<F: Field + Copy + Sized>(
    party: &mut MainParty,
    sum_share: F,
) -> MpcResult<RssShare<F>> {
    // Convert zs to RSS sharing
    let s_i = [sum_share + party.generate_alpha::<F>(1).next().unwrap()];
    let mut s_ii = [F::ZERO; 1];
    party.send_field_slice(Direction::Previous, &s_i);
    party
        .receive_field_slice(Direction::Next, &mut s_ii)
        .rcv()?;
    Ok(RssShare::from(s_i[0], s_ii[0]))
}

#[cfg(test)]
mod test {
    use std::fmt::Debug;
    use std::vec;

    use crate::lut_sp::our_online::LUT256SPMalTable;
    use crate::lut_sp_malsec::mult_verification::secret_share_two_party;
    use crate::rep3_core::party::test_export::TestSetup;
    use crate::rep3_core::party::{DigestExt, RngExt};
    use crate::rep3_core::share::{HasZero, RssShareGeneral};
    use crate::share::Empty;
    use crate::share::unsigned_ring::UR8;
    use crate::util::aligned_vec::{AlignedAllocator, AlignedVec};
    use crate::util::mul_triple_vec::ManyToOneMulTriple;
    use crate::{
        lut_sp_malsec::mult_verification::verify_multiplication_triples,
        rep3_core::{
            party::{
                MainParty,
                broadcast::{Broadcast, BroadcastContext},
            },
            share::RssShare,
            test::PartySetup,
        },
        share::{Field, mersenne61::Mersenne61, test::secret_share_vector, unsigned_ring::UR16},
        util::mul_triple_vec::InnerProductTriple,
    };
    use itertools::{izip, multiunzip};
    use rand::{CryptoRng, Rng, thread_rng};

    //     use super::{lagrange_deg2, ss_to_rss_shares, verify_dot_product, weak_mult};

    fn gen_rand_vec<R: Rng + CryptoRng, F: Field>(rng: &mut R, n: usize) -> Vec<F> {
        F::generate(rng, n)
    }

    fn weak_inner_product<F: Field>(
        a: &[RssShare<F>],
        b: &[RssShare<F>],
        ri: &[F],
        rii: &[F],
        num_lookups: usize,
        elems_per_block: usize,
        elems_per_lookup: usize,
    ) -> Vec<F> {
        izip!(
            a.chunks_exact(elems_per_lookup),
            ri.chunks_exact(elems_per_block),
            rii.chunks_exact(elems_per_block)
        )
        .enumerate()
        .flat_map(|(i, (a, ri, rii))| {
            let mut out = rii.to_vec();
            for j in 0..elems_per_block {
                out[j] -= ri[j];
            }
            for j in 0..elems_per_lookup {
                let y = b[(j / elems_per_block) * num_lookups + i];
                out[j % elems_per_block] += a[j].si * (y.si + y.sii) + a[j].sii * y.si;
            }
            out
        })
        .collect()
    }

    fn convert_vec<F: Field>(x: &Vec<F>) -> AlignedVec<F, 64> {
        let mut y = vec::from_elem_in(F::ZERO, x.len(), AlignedAllocator::<64>);
        y.copy_from_slice(&x);
        y
    }

    fn generate_inner_product_triples<F: Field>(
        rng: &mut (impl Rng + CryptoRng),
        n: usize,
        num_lookups: usize,
        elems_per_block: usize,
        elems_per_lookup: usize,
        add_error: bool,
    ) -> (
        Vec<InnerProductTriple<F>>,
        Vec<InnerProductTriple<F>>,
        Vec<InnerProductTriple<F>>,
    ) {
        let a_vec: Vec<Vec<F>> = (0..n)
            .map(|_| gen_rand_vec::<_, F>(rng, num_lookups * elems_per_lookup))
            .collect();
        let b_vec: Vec<Vec<F>> = (0..n)
            .map(|_| gen_rand_vec::<_, F>(rng, num_lookups * elems_per_lookup / elems_per_block))
            .collect();
        let (r1, r2, r3) = (
            (0..n)
                .map(|_| gen_rand_vec::<_, F>(rng, num_lookups * elems_per_block))
                .collect::<Vec<_>>(),
            (0..n)
                .map(|_| gen_rand_vec::<_, F>(rng, num_lookups * elems_per_block))
                .collect::<Vec<_>>(),
            (0..n)
                .map(|_| gen_rand_vec::<_, F>(rng, num_lookups * elems_per_block))
                .collect::<Vec<_>>(),
        );

        let (a1, a2, a3): (Vec<_>, Vec<_>, Vec<_>) =
            multiunzip(a_vec.iter().map(|v| secret_share_vector(rng, v)));
        let (b1, b2, b3): (Vec<_>, Vec<_>, Vec<_>) =
            multiunzip(b_vec.iter().map(|v| secret_share_vector(rng, v)));
        let mut c1 = izip!(&a1, &b1, &r1, &r2)
            .map(|(a, b, ri, rii)| {
                weak_inner_product(
                    a,
                    b,
                    ri,
                    rii,
                    num_lookups,
                    elems_per_block,
                    elems_per_lookup,
                )
            })
            .collect::<Vec<_>>();
        let mut c2 = izip!(&a2, &b2, &r2, &r3)
            .map(|(a, b, ri, rii)| {
                weak_inner_product(
                    a,
                    b,
                    ri,
                    rii,
                    num_lookups,
                    elems_per_block,
                    elems_per_lookup,
                )
            })
            .collect::<Vec<_>>();
        let mut c3 = izip!(&a3, &b3, &r3, &r1)
            .map(|(a, b, ri, rii)| {
                weak_inner_product(
                    a,
                    b,
                    ri,
                    rii,
                    num_lookups,
                    elems_per_block,
                    elems_per_lookup,
                )
            })
            .collect::<Vec<_>>();

        if add_error {
            let mut r = gen_rand_vec::<_, F>(rng, 1)[0];
            if r.is_zero() {
                r = F::ONE;
            }
            match rng.gen_range(0..3) {
                0 => c1[rng.gen_range(0..n)][rng.gen_range(0..num_lookups * elems_per_block)] += r,
                1 => c2[rng.gen_range(0..n)][rng.gen_range(0..num_lookups * elems_per_block)] += r,
                2 => c3[rng.gen_range(0..n)][rng.gen_range(0..num_lookups * elems_per_block)] += r,
                _ => unreachable!(),
            };
        }

        // izip!(&c_vec, &c1, &c2, &c3).for_each(|(c, c1, c2, c3)| assert_eq!(*c, *c1 + *c2 + *c3));

        let make_triple = |a1: &Vec<Vec<RssShare<F>>>,
                           b1: &Vec<Vec<RssShare<F>>>,
                           c1: &Vec<Vec<F>>,
                           c2: &Vec<Vec<F>>,
                           r1: &Vec<Vec<F>>,
                           r2: &Vec<Vec<F>>| {
            izip!(a1, b1, c1, c2, r1, r2)
                .map(|(a, b, ci, cii, ri, rii)| InnerProductTriple {
                    ai: convert_vec(&a.iter().map(|x| x.si).collect()),
                    aii: convert_vec(&a.iter().map(|x| x.sii).collect()),
                    bi: convert_vec(&b.iter().map(|x| x.si).collect()),
                    bii: convert_vec(&b.iter().map(|x| x.sii).collect()),
                    ci: convert_vec(ci),
                    cii: convert_vec(cii),
                    ri: convert_vec(ri),
                    rii: convert_vec(rii),
                    elems_per_block,
                    elems_per_lookup,
                })
                .collect::<Vec<_>>()
        };

        let triple1 = make_triple(&a1, &b1, &c1, &c2, &r1, &r2);
        let triple2 = make_triple(&a2, &b2, &c2, &c3, &r2, &r3);
        let triple3 = make_triple(&a3, &b3, &c3, &c1, &r3, &r1);
        (triple1, triple2, triple3)
    }

    fn weak_mult<F: Field>(a: &[RssShare<F>], b: &[RssShare<F>], ri: &[F], rii: &[F]) -> Vec<F> {
        (0..a.len())
            .map(|i| {
                a[i].si * (b[i % b.len()].si + b[i % b.len()].sii)
                    + a[i].sii * b[i % b.len()].si
                    + rii[i]
                    - ri[i]
            })
            .collect()
    }

    fn generate_mul_triples<F: Field>(
        rng: &mut (impl Rng + CryptoRng),
        n: usize,
        m: usize,
        k: usize,
        add_error: bool,
    ) -> (
        Vec<ManyToOneMulTriple<F>>,
        Vec<ManyToOneMulTriple<F>>,
        Vec<ManyToOneMulTriple<F>>,
    ) {
        let a_vec: Vec<Vec<F>> = (0..n).map(|_| gen_rand_vec::<_, F>(rng, m)).collect();
        let b_vec: Vec<Vec<F>> = (0..n).map(|_| gen_rand_vec::<_, F>(rng, k)).collect();
        let (r1, r2, r3) = (
            (0..n)
                .map(|_| gen_rand_vec::<_, F>(rng, m))
                .collect::<Vec<_>>(),
            (0..n)
                .map(|_| gen_rand_vec::<_, F>(rng, m))
                .collect::<Vec<_>>(),
            (0..n)
                .map(|_| gen_rand_vec::<_, F>(rng, m))
                .collect::<Vec<_>>(),
        );

        let (a1, a2, a3): (Vec<_>, Vec<_>, Vec<_>) =
            multiunzip(a_vec.iter().map(|v| secret_share_vector(rng, v)));
        let (b1, b2, b3): (Vec<_>, Vec<_>, Vec<_>) =
            multiunzip(b_vec.iter().map(|v| secret_share_vector(rng, v)));
        let mut c1 = izip!(&a1, &b1, &r1, &r2)
            .map(|(a, b, ri, rii)| weak_mult(a, b, ri, rii))
            .collect::<Vec<_>>();
        let mut c2 = izip!(&a2, &b2, &r2, &r3)
            .map(|(a, b, ri, rii)| weak_mult(a, b, ri, rii))
            .collect::<Vec<_>>();
        let mut c3 = izip!(&a3, &b3, &r3, &r1)
            .map(|(a, b, ri, rii)| weak_mult(a, b, ri, rii))
            .collect::<Vec<_>>();

        if add_error {
            let mut r = gen_rand_vec::<_, F>(rng, 1)[0];
            if r.is_zero() {
                r = F::ONE;
            }
            match rng.gen_range(0..3) {
                0 => c1[rng.gen_range(0..n)][rng.gen_range(0..m)] += r,
                1 => c2[rng.gen_range(0..n)][rng.gen_range(0..m)] += r,
                2 => c3[rng.gen_range(0..n)][rng.gen_range(0..m)] += r,
                _ => unreachable!(),
            };
        }

        // izip!(&c_vec, &c1, &c2, &c3).for_each(|(c, c1, c2, c3)| assert_eq!(*c, *c1 + *c2 + *c3));

        let make_triple = |a1: &Vec<Vec<RssShare<F>>>,
                           b1: &Vec<Vec<RssShare<F>>>,
                           c1: &Vec<Vec<F>>,
                           c2: &Vec<Vec<F>>,
                           r1: &Vec<Vec<F>>,
                           r2: &Vec<Vec<F>>| {
            izip!(a1, b1, c1, c2, r1, r2)
                .map(|(a, b, ci, cii, ri, rii)| ManyToOneMulTriple {
                    ai: convert_vec(&a.iter().map(|x| x.si).collect()),
                    aii: convert_vec(&a.iter().map(|x| x.sii).collect()),
                    bi: convert_vec(&b.iter().map(|x| x.si).collect()),
                    bii: convert_vec(&b.iter().map(|x| x.sii).collect()),
                    ci: convert_vec(ci),
                    cii: convert_vec(cii),
                    ri: convert_vec(ri),
                    rii: convert_vec(rii),
                })
                .collect::<Vec<_>>()
        };

        let triple1 = make_triple(&a1, &b1, &c1, &c2, &r1, &r2);
        let triple2 = make_triple(&a2, &b2, &c2, &c3, &r2, &r3);
        let triple3 = make_triple(&a3, &b3, &c3, &c1, &r3, &r1);
        (triple1, triple2, triple3)
    }

    fn generate_binary_shares_single_party<F: Field>(
        rng: &mut (impl Rng + CryptoRng),
        n: usize,
        add_error: bool,
    ) -> (
        Vec<RssShare<F>>,
        Vec<RssShareGeneral<F, Empty>>,
        Vec<RssShareGeneral<Empty, F>>,
    ) {
        let mut sums: Vec<F> = (0..n)
            .map(|_| if rng.gen_bool(0.5) { F::ZERO } else { F::ONE })
            .collect();
        if add_error {
            sums[rng.gen_range(0..n)] += F::ONE + F::ONE;
        }
        secret_share_two_party(rng, &sums)
    }

    /// TODO: These tests are not working yet after a refactor. They need to be fixed.

    fn test_ur16_mul_verify_helper<F: Field + DigestExt + Debug>(
        add_error_ip: bool,
        add_error_mul: bool,
    ) {
        let m = 12;
        let n = 9;
        let k = 3;
        const N_THREADS: usize = 3;

        // let mut rng = thread_rng();
        // let ip_triples = generate_inner_product_triples(&mut rng, n, 9, 3, 12, add_error_ip);
        // let mul_triples = generate_mul_triples(&mut rng, n, 192, 192, add_error_mul);

        // let program = |ip_triples: Vec<InnerProductTriple<F>>,
        //                mul_triples: Vec<ManyToOneMulTriple<F>>| {
        //     move |p: &mut MainParty| {
        //         let mut context = BroadcastContext::new();
        //         let res = verify_multiplication_triples::<7, F, Mersenne61>(
        //             p,
        //             &mut context,
        //             &ip_triples,
        //             &mul_triples,
        //             5,
        //         )
        //         .unwrap();
        //         p.compare_view(context).unwrap();
        //         res
        //     }
        // };
        // let ((r1, _), (r2, _), (r3, _)) = PartySetup::localhost_setup_multithreads(
        //     N_THREADS,
        //     program(ip_triples.0, mul_triples.0),
        //     program(ip_triples.1, mul_triples.1),
        //     program(ip_triples.2, mul_triples.2),
        // );
        // assert_eq!(r1, !add_error_ip && !add_error_mul);
        // assert_eq!(r1, r2);
        // assert_eq!(r1, r3);
    }

    #[test]
    fn test_ur8_mul_verify_correctness() {
        test_ur16_mul_verify_helper::<UR8>(false, false);
    }

    #[test]
    fn test_ur16_mul_verify_correctness() {
        test_ur16_mul_verify_helper::<UR16>(false, false);
    }

    #[test]
    fn test_ur16_mul_verify_soundness() {
        test_ur16_mul_verify_helper::<UR16>(true, false);
        test_ur16_mul_verify_helper::<UR16>(false, true);
    }

    // using crate::lut_sp_malsec::mult_verification::{batch_and_lift_x, batch_and_lift_x_8_mersenne, compute_poly_dst,
    // compute_poly_dst_mersenne61};
    // #[test]
    // fn test_compute_poly_dst() {
    //     let mut rng = thread_rng();
    //     for _ in 0..100000 {
    //         let mut dst = std::vec::from_elem_in(Mersenne61::ZERO, 24, AlignedAllocator::<64>);
    //         let mut x = std::vec::from_elem_in(Mersenne61::ZERO, 48, AlignedAllocator::<64>);
    //         Mersenne61::fill(&mut rng, &mut x);
    //         let r = Mersenne61::generate(&mut rng, 1)[0];
    //         compute_poly_dst_mersenne61(&mut dst, &x, r);
    //         let mut dst2 = std::vec::from_elem_in(Mersenne61::ZERO, 24, AlignedAllocator::<64>);

    //         unsafe {
    //             compute_poly_dst(
    //                 &mut *(&mut dst2[..] as *mut [_]
    //                     as *mut [RssShareGeneral<Mersenne61, Empty>]),
    //                 &*(&x[..] as *const [_] as *const [RssShareGeneral<Mersenne61, Empty>]),
    //                 r,
    //             );
    //         }
    //         assert_eq!(dst, dst2);
    //     }
    // }

    // #[test]
    // fn test_batch_and_lift_x() {
    //     let mut rng = thread_rng();
    //     for _ in 0..1000 {
    //         let mut dst = std::vec::from_elem_in(RssShareGeneral::ZERO, 24, AlignedAllocator::<64>);
    //         let x = (0..13)
    //             .map(|_| {
    //                 let mut vec = std::vec::from_elem_in(
    //                     RssShareGeneral::<Empty, UR8>::ZERO,
    //                     24,
    //                     AlignedAllocator::<64>,
    //                 );
    //                 unsafe {
    //                     UR8::fill(&mut rng, &mut *(&mut vec[..] as *mut [_] as *mut [UR8]));
    //                 }
    //                 vec
    //             })
    //             .collect::<Vec<_>>();
    //         let powers = Mersenne61::generate(&mut rng, 13);
    //         batch_and_lift_x_par(&mut dst, &x, &powers);
    //         let mut dst2 = std::vec::from_elem_in(RssShareGeneral::ZERO, 24, AlignedAllocator::<64>);
    //         batch_and_lift_x(&mut dst2, &x, 0, &powers);
    //         assert_eq!(dst, dst2);
    //     }
    // }
}
