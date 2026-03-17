use std::fmt::Debug;
use std::ops::{Add, BitXor, Mul, MulAssign, Sub};
use std::time::Instant;
use std::{slice, vec};

use crate::rep3_core::party::correlated_randomness::GlobalRng;
use crate::rep3_core::party::RngExt;
use crate::share::bs_bool8::BsBool8;
use crate::share::{BitDecompose, CountOnes, CountOnesParity, Field, InnerProduct};
use crate::share::{FieldLike, LagrangeInterExtrapolate, PrimeField};
use crate::util::aligned_vec::{AlignedAllocator, AlignedVec};
use crate::util::mul_triple_vec::ManyToOneMulTripleBoolean;
use crate::{
    rep3_core::{
        network::task::Direction,
        party::{
            broadcast::{Broadcast, BroadcastContext},
            error::MpcResult,
            DigestExt, MainParty, Party,
        },
        share::{EmptyRssShare, HasZero, RssShare, RssShareGeneral, RssShareVec},
    },
    share::Empty,
    util::mul_triple_vec::InnerProductTripleBoolean,
};
use itertools::{izip, multiunzip, zip_eq, Itertools};
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

/// This protocol checks that
/// (1) the many-to-one multiplication tuples are correctly computed;
/// (2) the inner product tuples are correctly computed;
/// (3) the shared values claimed to be binary (0 or 1) are actually binary.
pub fn verify_multiplication_triples<
    const DEGREE: usize,
    Fp: PrimeField + InnerProduct + Debug + LagrangeInterExtrapolate<DEGREE>,
>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    triples: &[InnerProductTripleBoolean<BsBool8>],
    mul_triples: &[ManyToOneMulTripleBoolean<BsBool8>],
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

    let t_len = kappa * (Fp::NBITS - 1);
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
            let self_instance = get_self_inner_product_tuples(triples, &mut self_global_rng, kappa);
            let self_instance_mul = get_self_mul_tuples(mul_triples, &mut self_global_rng, kappa);

            let self_t = prove_lift_instances::<BsBool8, Fp>(&self_instance_mul, &self_instance);
            let (t_s1, t_s2, t_s3) = secret_share_two_party(&mut party.random_local, &self_t);
            party.io().send_field_slice(Direction::Next, &t_s2);
            party.io().send_field_slice(Direction::Previous, &t_s3);

            let t_instance = binary_shares_to_self_instance(&t_s1);
            let self_t = t_s1
                .par_chunks_exact(Fp::NBITS - 1)
                .map(|t_shares| {
                    reconstruct_t::<Fp, _, _>(t_shares) - RssShare::from(Fp::ONE, Fp::ONE)
                })
                .collect::<Vec<_>>();

            (self_instance_mul, self_instance, self_t, t_instance)
        },
        || {
            rayon::join(
                || {
                    let next_instance =
                        get_next_inner_product_tuples(triples, &mut next_global_rng, kappa);
                    let next_instance_mul =
                        get_next_mul_tuples(mul_triples, &mut next_global_rng, kappa);
                    let next_t = next_t_share.rcv().unwrap();

                    let t_instance = binary_shares_to_next_instance(&next_t);
                    let next_t = next_t
                        .par_chunks_exact(Fp::NBITS - 1)
                        .map(|t_shares| {
                            reconstruct_t::<Fp, _, _>(t_shares)
                                - RssShareGeneral::from(Empty, Fp::ONE)
                        })
                        .collect::<Vec<_>>();

                    (next_instance_mul, next_instance, next_t, t_instance)
                },
                || {
                    let prev_instance =
                        get_prev_inner_product_tuples(triples, &mut prev_global_rng, kappa);
                    let prev_instance_mul =
                        get_prev_mul_tuples(mul_triples, &mut prev_global_rng, kappa);
                    let prev_t = prev_t_share.rcv().unwrap();

                    let t_instance = binary_shares_to_prev_instance(&prev_t);

                    let prev_t = prev_t
                        .par_chunks_exact(Fp::NBITS - 1)
                        .map(|t_shares| {
                            reconstruct_t::<Fp, _, _>(t_shares)
                                - RssShareGeneral::from(Fp::ONE, Empty)
                        })
                        .collect::<Vec<_>>();

                    (prev_instance_mul, prev_instance, prev_t, t_instance)
                },
            )
        },
    );

    println!("{} prepare instances: {:?}", party.i, start.elapsed());

    // Lift the instances

    let r: Fp = coin_flip(party, context)?;

    let start = Instant::now();
    let (self_instance, (next_instance, prev_instance)) = rayon::join(
        || batch_fp_instances::<BsBool8, Fp, _, _, _, _, _, _, _, _, _, _, _, _>(&self_instance, r),
        || {
            rayon::join(
                || {
                    batch_fp_instances::<BsBool8, Fp, _, _, _, _, _, _, _, _, _, _, _, _>(
                        &next_instance,
                        r,
                    )
                },
                || {
                    batch_fp_instances::<BsBool8, Fp, _, _, _, _, _, _, _, _, _, _, _, _>(
                        &prev_instance,
                        r,
                    )
                },
            )
        },
    );
    println!("{} batch instances: {:?}", party.i, start.elapsed());

    let start = Instant::now();
    println!(
        "Instance size: {} {}",
        self_instance.0 .0.len(),
        self_instance.0 .1.len()
    );

    let next_instance = (next_instance.0, next_instance.1, next_instance.2.into());
    let prev_instance = (prev_instance.0, prev_instance.1, prev_instance.2.into());
    let result =
        verify_dot_product_batched(party, context, self_instance, next_instance, prev_instance)?;
    println!("{} dot products: {:?}", party.i, start.elapsed());

    Ok(result)
}

fn linear_combination<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    let stride = a.len() / b.len();
    let mut out = vec![F::ZERO; stride];
    for i in 0..a.len() {
        out[i % stride] += a[i] * b[i / stride];
    }
    out
}

fn linear_combination_alt(a: &[BsBool8], b: &[BsBool8]) -> Vec<bool> {
    let mut out = vec![false; a.len() / b.len()];
    for i in 0..a.len() {
        out[i / b.len()] ^= ((a[i] * b[i % b.len()]).count_ones() % 2) != 0;
    }
    out
}

fn linear_combination_single(a: &[BsBool8], b: &[BsBool8]) -> bool {
    let mut out = false;
    for i in 0..a.len() {
        out ^= ((a[i] * b[i]).count_ones() % 2) != 0;
    }
    out
}
fn get_self_mul_tuples<F: Field + Debug>(
    triples: &[ManyToOneMulTripleBoolean<F>],
    global_rng: &mut (impl Rng + CryptoRng),
    kappa: usize,
) -> (
    (
        Vec<Vec<RssShareGeneral<Empty, F>>>,
        Vec<Vec<RssShareGeneral<F, Empty>>>,
    ),
    (
        Vec<RssShareGeneral<F, Empty>>,
        Vec<RssShareGeneral<Empty, F>>,
    ),
    Vec<RssShare<F>>,
) {
    let y1 = triples
        .iter()
        .flat_map(|triple| triple.bi.iter().map(|bi| RssShareGeneral::from(*bi, Empty)))
        .collect::<Vec<_>>();
    let y2 = triples
        .iter()
        .flat_map(|triple| {
            triple
                .bii
                .iter()
                .map(|bii| RssShareGeneral::from(Empty, *bii))
        })
        .collect::<Vec<_>>();

    let total_len = y1.len();
    let (x1, x2, z) = multiunzip((0..kappa).map(|_| {
        let mut x1 = Vec::with_capacity(total_len);
        let mut x2 = Vec::with_capacity(total_len);
        let mut z = RssShare::ZERO;
        let gamma = F::generate(global_rng, total_len);
        let mut offset = 0;
        for triple in triples {
            let gammas = &gamma[offset..offset + triple.bi.len()];
            offset += triple.bi.len();

            if triple.ai.len() == triple.bi.len() {
                x1.extend(
                    zip_eq(&triple.aii, gammas)
                        .map(|(xii, gamma)| RssShareGeneral::from(Empty, *xii * *gamma)),
                );
                x2.extend(
                    zip_eq(&triple.ai, gammas)
                        .map(|(xi, gamma)| RssShareGeneral::from(*xi * *gamma, Empty)),
                );
                z += izip!(
                    &triple.ci,
                    &triple.ai,
                    &triple.bi,
                    &triple.ri,
                    &triple.rii,
                    gammas
                )
                .map(|(ci, xi, bi, ri, rii, gamma)| {
                    RssShare::from(*ci - *xi * *bi + *ri, -*rii) * *gamma
                })
                .sum();
            } else {
                let coeffs = F::generate(global_rng, triple.ai.len() / triple.bi.len());
                let xi = linear_combination(&triple.ai, &coeffs);
                let xii = linear_combination(&triple.aii, &coeffs);
                let ri = linear_combination(&triple.ri, &coeffs);
                let rii = linear_combination(&triple.rii, &coeffs);
                let ci = linear_combination(&triple.ci, &coeffs);
                x1.extend(
                    zip_eq(&xii, gammas)
                        .map(|(xii, gamma)| RssShareGeneral::from(Empty, *xii * *gamma)),
                );
                x2.extend(
                    zip_eq(&xi, gammas)
                        .map(|(xi, gamma)| RssShareGeneral::from(*xi * *gamma, Empty)),
                );
                z += izip!(&ci, &xi, &triple.bi, &ri, &rii, gammas)
                    .map(|(ci, xi, bi, ri, rii, gamma)| {
                        RssShare::from(*ci - *xi * *bi + *ri, -*rii) * *gamma
                    })
                    .sum();
            }
        }
        (x1, x2, z)
    }));
    ((x1, x2), (y1, y2), z)
}

fn get_next_mul_tuples<F: Field + Debug>(
    triples: &[ManyToOneMulTripleBoolean<F>],
    global_rng: &mut (impl Rng + CryptoRng),
    kappa: usize,
) -> (
    (
        Vec<Vec<RssShare<Empty>>>,
        Vec<Vec<RssShareGeneral<Empty, F>>>,
    ),
    (Vec<RssShareGeneral<Empty, F>>, Vec<RssShare<Empty>>),
    Vec<RssShareGeneral<Empty, F>>,
) {
    let y1 = triples
        .iter()
        .flat_map(|triple| {
            triple
                .bii
                .iter()
                .map(|bii| RssShareGeneral::from(Empty, *bii))
        })
        .collect::<Vec<_>>();
    let y2 = vec![EmptyRssShare; y1.len()];
    let total_len = y1.len();
    let (x1, x2, z) = multiunzip((0..kappa).map(|_| {
        let x1 = vec![EmptyRssShare; total_len];
        let mut x2 = Vec::with_capacity(total_len);
        let mut z = RssShareGeneral::ZERO;
        let gamma = F::generate(global_rng, total_len);
        let mut offset = 0;
        for triple in triples {
            let gammas = &gamma[offset..offset + triple.bi.len()];
            offset += triple.bi.len();
            if triple.ai.len() == triple.bi.len() {
                x2.extend(
                    zip_eq(&triple.aii, gammas)
                        .map(|(xii, gamma)| RssShareGeneral::from(Empty, *xii * *gamma)),
                );
                z += izip!(&triple.cii, &triple.aii, &triple.bii, &triple.rii, gammas)
                    .map(|(cii, xii, bii, rii, gamma)| {
                        RssShareGeneral::from(Empty, *cii - *xii * *bii + *rii) * *gamma
                    })
                    .sum();
            } else {
                let coeffs = F::generate(global_rng, triple.ai.len() / triple.bi.len());
                let xii = linear_combination(&triple.aii, &coeffs);
                let rii = linear_combination(&triple.rii, &coeffs);
                let cii = linear_combination(&triple.cii, &coeffs);
                x2.extend(
                    zip_eq(&xii, gammas)
                        .map(|(xii, gamma)| RssShareGeneral::from(Empty, *xii * *gamma)),
                );
                z += izip!(&cii, &xii, &triple.bii, &rii, gammas)
                    .map(|(cii, xii, bii, rii, gamma)| {
                        RssShareGeneral::from(Empty, *cii - *xii * *bii + *rii) * *gamma
                    })
                    .sum();
            }
        }
        (x1, x2, z)
    }));
    ((x1, x2), (y1, y2), z)
}

fn get_prev_mul_tuples<F: Field + Debug>(
    triples: &[ManyToOneMulTripleBoolean<F>],
    global_rng: &mut (impl Rng + CryptoRng),
    kappa: usize,
) -> (
    (
        Vec<Vec<RssShareGeneral<F, Empty>>>,
        Vec<Vec<RssShare<Empty>>>,
    ),
    (Vec<RssShare<Empty>>, Vec<RssShareGeneral<F, Empty>>),
    Vec<RssShareGeneral<F, Empty>>,
) {
    let y2 = triples
        .iter()
        .flat_map(|triple| triple.bi.iter().map(|bi| RssShareGeneral::from(*bi, Empty)))
        .collect::<Vec<_>>();
    let y1 = vec![EmptyRssShare; y2.len()];
    let total_len = y1.len();
    let (x1, x2, z) = multiunzip((0..kappa).map(|_| {
        let mut x1 = Vec::with_capacity(total_len);
        let x2 = vec![EmptyRssShare; total_len];
        let mut z = RssShareGeneral::ZERO;
        let gamma = F::generate(global_rng, total_len);
        let mut offset = 0;
        for triple in triples {
            let gammas = &gamma[offset..offset + triple.bi.len()];
            offset += triple.bi.len();

            if triple.ai.len() == triple.bi.len() {
                x1.extend(
                    zip_eq(&triple.ai, gammas)
                        .map(|(xi, gamma)| RssShareGeneral::from(*xi * *gamma, Empty)),
                );
                z += zip_eq(&triple.ri, gammas)
                    .map(|(ri, gamma)| RssShareGeneral::from(-*ri, Empty) * *gamma)
                    .sum();
            } else {
                let coeffs = F::generate(global_rng, triple.ai.len() / triple.bi.len());
                let xi = linear_combination(&triple.ai, &coeffs);
                let ri = linear_combination(&triple.ri, &coeffs);
                x1.extend(
                    izip!(&xi, gammas)
                        .map(|(xi, gamma)| RssShareGeneral::from(*xi * *gamma, Empty)),
                );
                z += zip_eq(&ri, gammas)
                    .map(|(ri, gamma)| RssShareGeneral::from(-*ri, Empty) * *gamma)
                    .sum();
            }
        }
        (x1, x2, z)
    }));
    ((x1, x2), (y1, y2), z)
}

fn get_self_inner_product_tuples(
    triples: &[InnerProductTripleBoolean<BsBool8>],
    global_rng: &mut (impl Rng + CryptoRng),
    kappa: usize,
) -> (
    (
        Vec<Vec<RssShareGeneral<Empty, bool>>>,
        Vec<Vec<RssShareGeneral<bool, Empty>>>,
    ),
    (
        Vec<RssShareGeneral<bool, Empty>>,
        Vec<RssShareGeneral<Empty, bool>>,
    ),
    Vec<RssShare<bool>>,
) {
    // Self's instance (functioning as P_i)
    let total_len = triples
        .iter()
        .map(|triple| {
            triple.ci.len() * 8 / triple.bits_per_block * triple.bits_per_lookup
                / triple.bits_per_block
        })
        .sum::<usize>();

    let mut y1 = Vec::with_capacity(total_len);
    let mut y2 = Vec::with_capacity(total_len);
    for triple in triples {
        let bytes_per_block = triple.bits_per_block / 8;
        let num_lookups = triple.ci.len() / bytes_per_block;
        let ohv_block_size = (num_lookups + 7) / 8;
        let b_len = triple.bits_per_lookup / triple.bits_per_block;
        let is_preshifted = triple.shifts.is_empty();
        for i in 0..num_lookups {
            let byte_index = i / 8;
            let mask = 1 << (i % 8);
            let get_b = |b_array: &[BsBool8], mut j: usize| {
                if !is_preshifted {
                    j ^= triple.shifts[i];
                }
                (b_array[j * ohv_block_size + byte_index].0 & mask) != 0
            };
            for j in 0..b_len {
                y1.push(RssShareGeneral::from(get_b(&triple.bi, j), Empty));
                y2.push(RssShareGeneral::from(Empty, get_b(&triple.bii, j)));
            }
        }
    }
    debug_assert_eq!(y1.len(), total_len);
    debug_assert_eq!(y2.len(), total_len);

    let (x1, x2, z) = multiunzip((0..kappa).map(|_| {
        let mut x1 = Vec::with_capacity(total_len);
        let mut x2 = Vec::with_capacity(total_len);
        let mut z = RssShare::from(false, false);
        for triple in triples {
            debug_assert!(triple.bits_per_block % 8 == 0);
            let bytes_per_block = triple.bits_per_block / 8;
            let bytes_per_lookup = triple.bits_per_lookup / 8;
            let num_lookups = triple.ci.len() / bytes_per_block;
            let ohv_block_size = (num_lookups + 7) / 8;
            let coeffs = BsBool8::generate(global_rng, triple.ci.len());
            let gamma = BsBool8::generate(global_rng, ohv_block_size);
            let b_len = triple.bits_per_lookup / triple.bits_per_block;
            let is_preshifted = triple.shifts.is_empty();
            for (i, (ai, aii, ci, ri, rii, coeffs)) in izip!(
                triple.ai.chunks_exact(bytes_per_lookup),
                triple.aii.chunks_exact(bytes_per_lookup),
                triple.ci.chunks_exact(bytes_per_block),
                triple.ri.chunks_exact(bytes_per_block),
                triple.rii.chunks_exact(bytes_per_block),
                coeffs.chunks_exact(bytes_per_block),
            )
            .enumerate()
            {
                let byte_index = i / 8;
                let mask = 1 << (i % 8);
                let gamma = (gamma[byte_index].0 & mask) != 0;
                if !gamma {
                    x1.extend((0..b_len).map(|_| RssShareGeneral::from(Empty, false)));
                    x2.extend((0..b_len).map(|_| RssShareGeneral::from(false, Empty)));
                    continue;
                }

                let get_b = |b_array: &[BsBool8], mut j: usize| {
                    if !is_preshifted {
                        j ^= triple.shifts[i];
                    }
                    (b_array[j * ohv_block_size + byte_index].0 & mask) != 0
                };

                let xi = linear_combination_alt(ai, coeffs);
                let xii = linear_combination_alt(aii, coeffs);
                let ri = linear_combination_single(ri, coeffs);
                let rii = linear_combination_single(rii, coeffs);
                let ci = linear_combination_single(ci, coeffs);
                x1.extend(
                    xii.iter()
                        .map(|xii| RssShareGeneral::from(Empty, *xii & gamma)),
                );
                x2.extend(
                    xi.iter()
                        .map(|xi| RssShareGeneral::from(*xi & gamma, Empty)),
                );
                z ^= RssShare::from(
                    ci ^ xi
                        .iter()
                        .enumerate()
                        .map(|(j, xi)| *xi & get_b(&triple.bi, j))
                        .reduce(|acc, x| acc ^ x)
                        .unwrap()
                        ^ ri,
                    rii,
                ) & gamma;
            }
        }
        debug_assert_eq!(x1.len(), total_len);
        debug_assert_eq!(x2.len(), total_len);
        (x1, x2, z)
    }));
    ((x1, x2), (y1, y2), z)
}

fn get_next_inner_product_tuples(
    triples: &[InnerProductTripleBoolean<BsBool8>],
    global_rng: &mut (impl Rng + CryptoRng),
    kappa: usize,
) -> (
    (
        Vec<Vec<RssShare<Empty>>>,
        Vec<Vec<RssShareGeneral<Empty, bool>>>,
    ),
    (Vec<RssShareGeneral<Empty, bool>>, Vec<RssShare<Empty>>),
    Vec<RssShareGeneral<Empty, bool>>,
) {
    // Next's instance (self as P_{i-1})
    let total_len = triples
        .iter()
        .map(|triple| {
            triple.ci.len() * 8 / triple.bits_per_block * triple.bits_per_lookup
                / triple.bits_per_block
        })
        .sum::<usize>();
    let mut y1 = Vec::with_capacity(total_len);
    let y2 = vec![EmptyRssShare; total_len];
    for triple in triples {
        let bytes_per_block = triple.bits_per_block / 8;
        let num_lookups = triple.ci.len() / bytes_per_block;
        let ohv_block_size = (num_lookups + 7) / 8;
        let b_len = triple.bits_per_lookup / triple.bits_per_block;
        let is_preshifted = triple.shifts.is_empty();
        for i in 0..num_lookups {
            let byte_index = i / 8;
            let mask = 1 << (i % 8);
            let get_b = |b_array: &[BsBool8], mut j: usize| {
                if !is_preshifted {
                    j ^= triple.shifts[i];
                }
                (b_array[j * ohv_block_size + byte_index].0 & mask) != 0
            };
            for j in 0..b_len {
                y1.push(RssShareGeneral::from(Empty, get_b(&triple.bii, j)));
            }
        }
    }

    let (x1, x2, z) = multiunzip((0..kappa).map(|_| {
        let x1 = vec![EmptyRssShare; total_len];
        let mut x2 = Vec::with_capacity(total_len);
        let mut z = RssShareGeneral::from(Empty, false);
        for triple in triples {
            let bytes_per_block = triple.bits_per_block / 8;
            let bytes_per_lookup = triple.bits_per_lookup / 8;
            let num_lookups = triple.cii.len() / bytes_per_block;
            let ohv_block_size = (num_lookups + 7) / 8;
            let coeffs = BsBool8::generate(global_rng, triple.cii.len());
            let gamma = BsBool8::generate(global_rng, ohv_block_size);
            let b_len = triple.bits_per_lookup / triple.bits_per_block;
            let is_preshifted = triple.shifts.is_empty();
            for (i, (aii, cii, rii, coeffs)) in izip!(
                triple.aii.chunks_exact(bytes_per_lookup),
                triple.cii.chunks_exact(bytes_per_block),
                triple.rii.chunks_exact(bytes_per_block),
                coeffs.chunks_exact(bytes_per_block),
            )
            .enumerate()
            {
                let byte_index = i / 8;
                let mask = 1 << (i % 8);
                let gamma = (gamma[byte_index].0 & mask) != 0;
                if !gamma {
                    x2.extend((0..b_len).map(|_| RssShareGeneral::from(Empty, false)));
                    continue;
                }

                let get_b = |b_array: &[BsBool8], mut j: usize| {
                    if !is_preshifted {
                        j ^= triple.shifts[i];
                    }
                    (b_array[j * ohv_block_size + byte_index].0 & mask) != 0
                };

                let xii = linear_combination_alt(aii, coeffs);
                let rii = linear_combination_single(rii, coeffs);
                let cii = linear_combination_single(cii, coeffs);
                x2.extend(
                    xii.iter()
                        .map(|xii| RssShareGeneral::from(Empty, *xii & gamma)),
                );
                z ^= RssShareGeneral::from(
                    Empty,
                    cii ^ xii
                        .iter()
                        .enumerate()
                        .map(|(j, xii)| *xii & get_b(&triple.bii, j))
                        .reduce(|acc, x| acc ^ x)
                        .unwrap()
                        ^ rii,
                ) & gamma;
            }
        }
        (x1, x2, z)
    }));
    ((x1, x2), (y1, y2), z)
}

fn get_prev_inner_product_tuples(
    triples: &[InnerProductTripleBoolean<BsBool8>],
    global_rng: &mut (impl Rng + CryptoRng),
    kappa: usize,
) -> (
    (
        Vec<Vec<RssShareGeneral<bool, Empty>>>,
        Vec<Vec<RssShare<Empty>>>,
    ),
    (Vec<RssShare<Empty>>, Vec<RssShareGeneral<bool, Empty>>),
    Vec<RssShareGeneral<bool, Empty>>,
) {
    // Prev's instance (self as P_{i+1})
    let total_len = triples
        .iter()
        .map(|triple| {
            triple.ci.len() * 8 / triple.bits_per_block * triple.bits_per_lookup
                / triple.bits_per_block
        })
        .sum::<usize>();
    let y1 = vec![EmptyRssShare; total_len];
    let mut y2 = Vec::with_capacity(total_len);
    for triple in triples {
        let bytes_per_block = triple.bits_per_block / 8;
        let num_lookups = triple.ci.len() / bytes_per_block;
        let ohv_block_size = (num_lookups + 7) / 8;
        let b_len = triple.bits_per_lookup / triple.bits_per_block;
        let is_preshifted = triple.shifts.is_empty();
        for i in 0..num_lookups {
            let byte_index = i / 8;
            let mask = 1 << (i % 8);

            let get_b = |b_array: &[BsBool8], mut j: usize| {
                if !is_preshifted {
                    j ^= triple.shifts[i];
                }
                (b_array[j * ohv_block_size + byte_index].0 & mask) != 0
            };
            for j in 0..b_len {
                y2.push(RssShareGeneral::from(get_b(&triple.bi, j), Empty));
            }
        }
    }

    let (x1, x2, z) = multiunzip((0..kappa).map(|_| {
        let mut x1 = Vec::with_capacity(total_len);
        let x2 = vec![EmptyRssShare; total_len];
        let mut z = RssShareGeneral::from(false, Empty);
        for triple in triples {
            let bytes_per_block = triple.bits_per_block / 8;
            let bytes_per_lookup = triple.bits_per_lookup / 8;
            let num_lookups = triple.ci.len() / bytes_per_block;
            let ohv_block_size = (num_lookups + 7) / 8;
            let coeffs = BsBool8::generate(global_rng, triple.ci.len());
            let gamma = BsBool8::generate(global_rng, ohv_block_size);
            let b_len = triple.bits_per_lookup / triple.bits_per_block;

            for (i, (ai, ri, coeffs)) in izip!(
                triple.ai.chunks_exact(bytes_per_lookup),
                triple.ri.chunks_exact(bytes_per_block),
                coeffs.chunks_exact(bytes_per_block),
            )
            .enumerate()
            {
                let byte_index = i / 8;
                let mask = 1 << (i % 8);
                let gamma = (gamma[byte_index].0 & mask) != 0;
                if !gamma {
                    x1.extend((0..b_len).map(|_| RssShareGeneral::from(false, Empty)));
                    continue;
                }

                let xi = linear_combination_alt(ai, coeffs);
                let ri = linear_combination_single(ri, coeffs);
                x1.extend(
                    xi.iter()
                        .map(|xi| RssShareGeneral::from(*xi & gamma, Empty)),
                );

                z ^= RssShareGeneral::from(ri, Empty) & gamma;
            }
        }
        (x1, x2, z)
    }));
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

trait BooleanLift<Fp: PrimeField> {
    type Output;

    fn lift_boolean(&self) -> Self::Output;
}

impl<Fp: PrimeField> BooleanLift<Fp> for bool {
    type Output = Fp;
    fn lift_boolean(&self) -> Self::Output {
        if *self {
            Fp::ONE
        } else {
            Fp::ZERO
        }
    }
}

impl<Fp: PrimeField> BooleanLift<Fp> for Empty {
    type Output = Empty;
    fn lift_boolean(&self) -> Self::Output {
        Empty
    }
}

impl<Fp: PrimeField, T: BooleanLift<Fp>, U: BooleanLift<Fp>> BooleanLift<Fp>
    for RssShareGeneral<T, U>
{
    type Output = RssShareGeneral<<T as BooleanLift<Fp>>::Output, <U as BooleanLift<Fp>>::Output>;

    fn lift_boolean(&self) -> Self::Output {
        RssShareGeneral::from(self.si.lift_boolean(), self.sii.lift_boolean())
    }
}

fn batch_fp_instances<
    F: Field,
    Fp: PrimeField,
    T1: FieldLike + BitDecompose<Fp>,
    U1: FieldLike + BitDecompose<Fp>,
    T2: FieldLike + BitDecompose<Fp>,
    U2: FieldLike + BitDecompose<Fp>,
    T3: FieldLike + BitDecompose<Fp> + CountOnesParity<Output = TB3>,
    U3: FieldLike + BitDecompose<Fp> + CountOnesParity<Output = UB3>,
    TB1: Send
        + Sync
        + Copy
        + BitXor<TB1, Output = TB1>
        + BooleanLift<Fp, Output = <T1 as BitDecompose<Fp>>::Output>,
    UB1: Send
        + Sync
        + Copy
        + BitXor<UB1, Output = UB1>
        + BooleanLift<Fp, Output = <U1 as BitDecompose<Fp>>::Output>,
    TB2: Send
        + Sync
        + Copy
        + BitXor<TB2, Output = TB2>
        + BooleanLift<Fp, Output = <T2 as BitDecompose<Fp>>::Output>,
    UB2: Send
        + Sync
        + Copy
        + BitXor<UB2, Output = UB2>
        + BooleanLift<Fp, Output = <U2 as BitDecompose<Fp>>::Output>,
    TB3: Send
        + Sync
        + Copy
        + BitXor<TB3, Output = TB3>
        + BooleanLift<Fp, Output = <T3 as BitDecompose<Fp>>::Output>,
    UB3: Send
        + Sync
        + Copy
        + BitXor<UB3, Output = UB3>
        + BooleanLift<Fp, Output = <U3 as BitDecompose<Fp>>::Output>,
>(
    (((x1, x2), (y1, y2), z), ((x3, x4), (y3, y4), z2), t, (t_x, t_y, t_z)): &(
        (
            (
                Vec<Vec<RssShareGeneral<T1, U1>>>,
                Vec<Vec<RssShareGeneral<T2, U2>>>,
            ),
            (Vec<RssShareGeneral<T2, U2>>, Vec<RssShareGeneral<T1, U1>>),
            Vec<RssShareGeneral<T3, U3>>,
        ),
        (
            (
                Vec<Vec<RssShareGeneral<TB1, UB1>>>,
                Vec<Vec<RssShareGeneral<TB2, UB2>>>,
            ),
            (
                Vec<RssShareGeneral<TB2, UB2>>,
                Vec<RssShareGeneral<TB1, UB1>>,
            ),
            Vec<RssShareGeneral<TB3, UB3>>,
        ),
        Vec<<RssShareGeneral<TB3, UB3> as BooleanLift<Fp>>::Output>,
        (
            Vec<
                RssShareGeneral<<T1 as BitDecompose<Fp>>::Output, <U1 as BitDecompose<Fp>>::Output>,
            >,
            Vec<
                RssShareGeneral<<T2 as BitDecompose<Fp>>::Output, <U2 as BitDecompose<Fp>>::Output>,
            >,
            Vec<
                RssShareGeneral<<T3 as BitDecompose<Fp>>::Output, <U3 as BitDecompose<Fp>>::Output>,
            >,
        ),
    ),
    r: Fp,
) -> (
    (
        AlignedVec<
            RssShareGeneral<<T1 as BitDecompose<Fp>>::Output, <U1 as BitDecompose<Fp>>::Output>,
            64,
        >,
        AlignedVec<
            RssShareGeneral<<T2 as BitDecompose<Fp>>::Output, <U2 as BitDecompose<Fp>>::Output>,
            64,
        >,
    ),
    (
        AlignedVec<
            RssShareGeneral<<T2 as BitDecompose<Fp>>::Output, <U2 as BitDecompose<Fp>>::Output>,
            64,
        >,
        AlignedVec<
            RssShareGeneral<<T1 as BitDecompose<Fp>>::Output, <U1 as BitDecompose<Fp>>::Output>,
            64,
        >,
    ),
    RssShareGeneral<<T3 as BitDecompose<Fp>>::Output, <U3 as BitDecompose<Fp>>::Output>,
) {
    let powers = generate_powers(r, x1.len() + t_x.len());
    let ((x1, x2), ((y1, y2), z)) = rayon::join(
        || {
            rayon::join(
                || {
                    let mut out_x1 = vec::from_elem_in(
                        RssShareGeneral::ZERO,
                        x1[0].len() * F::NBITS + x3[0].len() + t_x.len(),
                        AlignedAllocator::<64>,
                    );
                    for (xs, power) in zip_eq(x1, &powers[..x1.len()]) {
                        (out_x1.par_chunks_exact_mut(F::NBITS), xs)
                            .into_par_iter()
                            .for_each(|(out, x)| {
                                for (y, x) in zip_eq(out, x.bit_decompose(F::NBITS)) {
                                    *y += x * *power;
                                }
                            });
                    }
                    for (xs, power) in zip_eq(x3, &powers[..x1.len()]) {
                        (&mut out_x1[x1[0].len() * F::NBITS..], xs)
                            .into_par_iter()
                            .for_each(|(out, x)| {
                                *out += x.lift_boolean() * *power;
                            });
                    }
                    izip!(
                        &mut out_x1[x1[0].len() * F::NBITS + x3[0].len()..],
                        t_x,
                        &powers[x1.len()..]
                    )
                    .for_each(|(out, xs, gamma)| *out = *xs * *gamma);
                    out_x1
                },
                || {
                    let mut out_x2 = vec::from_elem_in(
                        RssShareGeneral::ZERO,
                        x2[0].len() * F::NBITS + x4[0].len(),
                        AlignedAllocator::<64>,
                    );
                    for (xs, power) in zip_eq(x2, &powers[..x1.len()]) {
                        (out_x2.par_chunks_exact_mut(F::NBITS), xs)
                            .into_par_iter()
                            .for_each(|(out, x)| {
                                for (y, x) in zip_eq(out, x.bit_decompose(F::NBITS)) {
                                    *y += x * *power;
                                }
                            });
                    }
                    for (xs, power) in zip_eq(x4, &powers[..x1.len()]) {
                        (&mut out_x2[x1[0].len() * F::NBITS..], xs)
                            .into_par_iter()
                            .for_each(|(out, x)| {
                                *out += x.lift_boolean() * *power;
                            });
                    }
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
                                y1.len() * F::NBITS + y3.len() + t_y.len(),
                                AlignedAllocator::<64>,
                            );
                            (out_y1.par_chunks_exact_mut(F::NBITS), y1)
                                .into_par_iter()
                                .for_each(|(out, x)| {
                                    for (y, x) in zip_eq(out, x.bit_decompose(F::NBITS)) {
                                        *y = x;
                                    }
                                });
                            for (y, x) in zip_eq(
                                &mut out_y1[y1.len() * F::NBITS..y1.len() * F::NBITS + y3.len()],
                                y3,
                            ) {
                                *y = x.lift_boolean();
                            }
                            out_y1[y1.len() * F::NBITS + y3.len()..].copy_from_slice(t_y);
                            out_y1
                        },
                        || {
                            let mut out_y2 = vec::from_elem_in(
                                RssShareGeneral::ZERO,
                                y2.len() * F::NBITS + y4.len(),
                                AlignedAllocator::<64>,
                            );
                            (out_y2.par_chunks_exact_mut(F::NBITS), y2)
                                .into_par_iter()
                                .for_each(|(out, x)| {
                                    for (y, x) in zip_eq(out, x.bit_decompose(F::NBITS)) {
                                        *y = x;
                                    }
                                });
                            for (y, x) in zip_eq(&mut out_y2[y2.len() * F::NBITS..], y4) {
                                *y = x.lift_boolean();
                            }
                            out_y2
                        },
                    )
                },
                || {
                    (z, z2, &powers, t)
                        .into_par_iter()
                        .map(|(z, z2, power, t)| {
                            ((z.count_ones_parity() ^ *z2).lift_boolean() + *t) * *power
                        })
                        .chain(
                            t_z.par_iter()
                                .zip(powers[z.len()..].par_iter())
                                .map(|(z, power)| *z * *power),
                        )
                        .sum()
                },
            )
        },
    );
    ((x1, x2), (y1, y2), z)
}

fn bit_decompose<Fp: From<u64>>(input: u64, num_var: usize) -> Vec<Fp> {
    let mut res = Vec::with_capacity(num_var);
    let mut i = input;
    for _ in 0..num_var {
        res.push(Fp::from((i & 1 == 1) as u64));
        i >>= 1;
    }
    res
}

fn reconstruct_t<
    Fp: PrimeField,
    T1: FieldLike + Mul<Fp, Output = T1>,
    U1: FieldLike + Mul<Fp, Output = U1>,
>(
    t_shares: &[RssShareGeneral<T1, U1>],
) -> RssShareGeneral<T1, U1> {
    let mut res = RssShareGeneral::ZERO;
    let mut multiplier = Fp::TWO;
    for t_share in t_shares {
        res += *t_share * multiplier;
        multiplier += multiplier;
    }
    res
}

fn prove_lift_instances<F: Field + CountOnes<Output = u32>, Fp: PrimeField>(
    ((x1, x2), (y1, y2), z): &(
        (
            Vec<Vec<RssShareGeneral<Empty, F>>>,
            Vec<Vec<RssShareGeneral<F, Empty>>>,
        ),
        (
            Vec<RssShareGeneral<F, Empty>>,
            Vec<RssShareGeneral<Empty, F>>,
        ),
        Vec<RssShare<F>>,
    ),
    ((x3, x4), (y3, y4), z2): &(
        (
            Vec<Vec<RssShareGeneral<Empty, bool>>>,
            Vec<Vec<RssShareGeneral<bool, Empty>>>,
        ),
        (
            Vec<RssShareGeneral<bool, Empty>>,
            Vec<RssShareGeneral<Empty, bool>>,
        ),
        Vec<RssShare<bool>>,
    ),
) -> Vec<Fp> {
    let h = (x1, x2, x3, x4)
        .into_par_iter()
        .map(|(x1, x2, x3, x4)| {
            x1.iter()
                .zip_eq(y1.iter())
                .map(|(x, y)| (x.sii * y.si).count_ones() as u64)
                .sum::<u64>()
                + x2.iter()
                    .zip_eq(y2.iter())
                    .map(|(x, y)| (x.si * y.sii).count_ones() as u64)
                    .sum::<u64>()
                + x3.iter()
                    .zip_eq(y3.iter())
                    .map(|(x, y)| (x.sii & y.si) as u64)
                    .sum::<u64>()
                + x4.iter()
                    .zip_eq(y4.iter())
                    .map(|(x, y)| (x.si & y.sii) as u64)
                    .sum::<u64>()
        })
        .collect::<Vec<_>>();
    let t = h
        .iter()
        .zip_eq(z.iter())
        .zip_eq(z2.iter())
        .map(|((h, z), z2)| {
            let z_si = ((z.si.count_ones() % 2) != 0) ^ z2.si;
            let z_sii = ((z.sii.count_ones() % 2) != 0) ^ z2.sii;

            let val = h + 2 - (z_si as u64 + z_sii as u64);
            // debug_assert!(val % 2 == 0);
            val / 2
        })
        .collect::<Vec<_>>();
    t.iter()
        .flat_map(|t| bit_decompose::<Fp>(*t, Fp::NBITS - 1))
        .collect()
}

fn generate_powers<Fp: PrimeField>(r: Fp, len: usize) -> Vec<Fp> {
    let mut result = vec![Fp::ONE; len];
    for i in 1..result.len() {
        result[i] = result[i - 1] * r;
    }
    result
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
        _mm512_mask_sub_epi64, _mm512_mul_epu32, _mm512_permutex2var_epi64, _mm512_set1_epi64,
        _mm512_set_epi64, _mm512_shrdi_epi64, _mm512_slli_epi64, _mm512_srli_epi64,
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
    let mut n1 = self_instance.0 .0.len();
    let mut n2 = self_instance.0 .1.len();
    debug_assert!(n1 >= n2);

    if n1 == 1 {
        let self_instance = (
            (self_instance.0 .0[0].into(), self_instance.0 .1[0].into()),
            (self_instance.1 .0[0].into(), self_instance.1 .1[0].into()),
            self_instance.2,
        );
        let next_instance = (
            (next_instance.0 .0[0].into(), next_instance.0 .1[0].into()),
            (next_instance.1 .0[0].into(), next_instance.1 .1[0].into()),
            next_instance.2,
        );
        let prev_instance = (
            (prev_instance.0 .0[0].into(), prev_instance.0 .1[0].into()),
            (prev_instance.1 .0[0].into(), prev_instance.1 .1[0].into()),
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
    use crate::lut_sp_boolean_malsec::mult_verification::secret_share_two_party;
    use crate::rep3_core::party::test_export::TestSetup;
    use crate::rep3_core::share::RssShareGeneral;
    use crate::share::bs_bool8::BsBool8;
    use crate::share::Empty;
    use crate::util::mul_triple_vec::ManyToOneMulTripleBoolean;
    use crate::{
        lut_sp_boolean_malsec::mult_verification::verify_multiplication_triples,
        rep3_core::{
            party::{
                broadcast::{Broadcast, BroadcastContext},
                MainParty,
            },
            share::RssShare,
            test::PartySetup,
        },
        share::{mersenne61::Mersenne61, test::secret_share_vector, Field},
        util::mul_triple_vec::InnerProductTripleBoolean,
    };
    use itertools::{izip, multiunzip};
    use rand::{thread_rng, CryptoRng, Rng};

    //     use super::{lagrange_deg2, ss_to_rss_shares, verify_dot_product, weak_mult};

    fn gen_rand_vec<R: Rng + CryptoRng, F: Field>(rng: &mut R, n: usize) -> Vec<F> {
        F::generate(rng, n)
    }

    fn weak_inner_product(
        a: &[RssShare<BsBool8>],
        b: &[RssShare<BsBool8>],
        ri: &[BsBool8],
        rii: &[BsBool8],
        num_lookups: usize,
        bits_per_block: usize,
        bits_per_lookup: usize,
    ) -> Vec<BsBool8> {
        let bytes_per_block = bits_per_block / 8;
        let bytes_per_lookup = bits_per_lookup / 8;
        let ohv_block_size = (num_lookups + 7) / 8;
        izip!(
            a.chunks_exact(bytes_per_lookup),
            ri.chunks_exact(bytes_per_block),
            rii.chunks_exact(bytes_per_block)
        )
        .enumerate()
        .flat_map(|(i, (a, ri, rii))| {
            let byte_index = i / 8;
            let mask = 1 << (i % 8);

            let mut out = rii.to_vec();
            for j in 0..bytes_per_block {
                out[j] -= ri[j];
            }
            for j in 0..bytes_per_lookup {
                let k = j / bytes_per_block;
                let y_si = (b[k * ohv_block_size + byte_index].si.0 & mask) != 0;
                let y_sii = (b[k * ohv_block_size + byte_index].sii.0 & mask) != 0;
                match (y_si, y_sii) {
                    (true, true) => {
                        out[j % bytes_per_block] += a[j].sii;
                    }
                    (true, false) => {
                        out[j % bytes_per_block] += a[j].sii + a[j].si;
                    }
                    (false, true) => {
                        out[j % bytes_per_block] += a[j].si;
                    }
                    (false, false) => {
                        // do nothing
                    }
                }
            }
            out
        })
        .collect()
    }

    fn generate_inner_product_triples(
        rng: &mut (impl Rng + CryptoRng),
        n: usize,
        num_lookups: usize,
        bits_per_block: usize,
        bits_per_lookup: usize,
        add_error: bool,
    ) -> (
        Vec<InnerProductTripleBoolean<BsBool8>>,
        Vec<InnerProductTripleBoolean<BsBool8>>,
        Vec<InnerProductTripleBoolean<BsBool8>>,
    ) {
        let a_vec: Vec<Vec<BsBool8>> = (0..n)
            .map(|_| gen_rand_vec::<_, BsBool8>(rng, num_lookups * bits_per_lookup / 8))
            .collect();
        let b_vec: Vec<Vec<BsBool8>> = (0..n)
            .map(|_| {
                gen_rand_vec::<_, BsBool8>(
                    rng,
                    (num_lookups + 7) / 8 * bits_per_lookup / bits_per_block,
                )
            })
            .collect();
        let (r1, r2, r3) = (
            (0..n)
                .map(|_| gen_rand_vec::<_, BsBool8>(rng, num_lookups * bits_per_block / 8))
                .collect::<Vec<_>>(),
            (0..n)
                .map(|_| gen_rand_vec::<_, BsBool8>(rng, num_lookups * bits_per_block / 8))
                .collect::<Vec<_>>(),
            (0..n)
                .map(|_| gen_rand_vec::<_, BsBool8>(rng, num_lookups * bits_per_block / 8))
                .collect::<Vec<_>>(),
        );

        let (a1, a2, a3): (Vec<_>, Vec<_>, Vec<_>) =
            multiunzip(a_vec.iter().map(|v| secret_share_vector(rng, v)));
        let (b1, b2, b3): (Vec<_>, Vec<_>, Vec<_>) =
            multiunzip(b_vec.iter().map(|v| secret_share_vector(rng, v)));
        let mut c1 = izip!(&a1, &b1, &r1, &r2)
            .map(|(a, b, ri, rii)| {
                weak_inner_product(a, b, ri, rii, num_lookups, bits_per_block, bits_per_lookup)
            })
            .collect::<Vec<_>>();
        let mut c2 = izip!(&a2, &b2, &r2, &r3)
            .map(|(a, b, ri, rii)| {
                weak_inner_product(a, b, ri, rii, num_lookups, bits_per_block, bits_per_lookup)
            })
            .collect::<Vec<_>>();
        let mut c3 = izip!(&a3, &b3, &r3, &r1)
            .map(|(a, b, ri, rii)| {
                weak_inner_product(a, b, ri, rii, num_lookups, bits_per_block, bits_per_lookup)
            })
            .collect::<Vec<_>>();

        if add_error {
            let mut r = gen_rand_vec::<_, BsBool8>(rng, 1)[0];
            if r.is_zero() {
                r = BsBool8::ONE;
            }
            match rng.gen_range(0..3) {
                0 => {
                    c1[rng.gen_range(0..n)][rng.gen_range(0..num_lookups * bits_per_block / 8)] += r
                }
                1 => {
                    c2[rng.gen_range(0..n)][rng.gen_range(0..num_lookups * bits_per_block / 8)] += r
                }
                2 => {
                    c3[rng.gen_range(0..n)][rng.gen_range(0..num_lookups * bits_per_block / 8)] += r
                }
                _ => unreachable!(),
            };
        }

        // izip!(&c_vec, &c1, &c2, &c3).for_each(|(c, c1, c2, c3)| assert_eq!(*c, *c1 + *c2 + *c3));

        let make_triple = |a1: &Vec<Vec<RssShare<BsBool8>>>,
                           b1: &Vec<Vec<RssShare<BsBool8>>>,
                           c1: &Vec<Vec<BsBool8>>,
                           c2: &Vec<Vec<BsBool8>>,
                           r1: &Vec<Vec<BsBool8>>,
                           r2: &Vec<Vec<BsBool8>>| {
            izip!(a1, b1, c1, c2, r1, r2)
                .map(|(a, b, ci, cii, ri, rii)| InnerProductTripleBoolean {
                    ai: a.iter().map(|x| x.si).collect(),
                    aii: a.iter().map(|x| x.sii).collect(),
                    bi: b.iter().map(|x| x.si).collect(),
                    bii: b.iter().map(|x| x.sii).collect(),
                    ci: ci.clone(),
                    cii: cii.clone(),
                    ri: ri.clone(),
                    rii: rii.clone(),
                    shifts: vec![],
                    bits_per_block,
                    bits_per_lookup,
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
        Vec<ManyToOneMulTripleBoolean<F>>,
        Vec<ManyToOneMulTripleBoolean<F>>,
        Vec<ManyToOneMulTripleBoolean<F>>,
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
                .map(|(a, b, ci, cii, ri, rii)| ManyToOneMulTripleBoolean {
                    ai: a.iter().map(|x| x.si).collect(),
                    aii: a.iter().map(|x| x.sii).collect(),
                    bi: b.iter().map(|x| x.si).collect(),
                    bii: b.iter().map(|x| x.sii).collect(),
                    ci: ci.clone(),
                    cii: cii.clone(),
                    ri: ri.clone(),
                    rii: rii.clone(),
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

    fn test_bsbool8_mul_verify_helper(add_error_ip: bool, add_error_mul: bool) {
        let m = 12;
        let n = 9;
        let k = 3;
        const N_THREADS: usize = 3;

        let mut rng = thread_rng();
        let ip_triples = generate_inner_product_triples(&mut rng, n, 13, 24, 72, add_error_ip);
        let mul_triples = generate_mul_triples(&mut rng, n, m, k, add_error_mul);

        let program = |ip_triples: Vec<InnerProductTripleBoolean<BsBool8>>,
                       mul_triples: Vec<ManyToOneMulTripleBoolean<BsBool8>>| {
            move |p: &mut MainParty| {
                let mut context = BroadcastContext::new();
                let res = verify_multiplication_triples::<3, Mersenne61>(
                    p,
                    &mut context,
                    &ip_triples,
                    &mul_triples,
                    5,
                )
                .unwrap();
                p.compare_view(context).unwrap();
                res
            }
        };
        let ((r1, _), (r2, _), (r3, _)) = PartySetup::localhost_setup_multithreads(
            N_THREADS,
            program(ip_triples.0, mul_triples.0),
            program(ip_triples.1, mul_triples.1),
            program(ip_triples.2, mul_triples.2),
        );
        assert_eq!(r1, !add_error_ip && !add_error_mul);
        assert_eq!(r1, r2);
        assert_eq!(r1, r3);
    }

    #[test]
    fn test_bsbool8_mul_verify_correctness() {
        test_bsbool8_mul_verify_helper(false, false);
    }

    #[test]
    fn test_bsbool8_mul_verify_soundness() {
        test_bsbool8_mul_verify_helper(true, false);
        test_bsbool8_mul_verify_helper(false, true);
    }
}
