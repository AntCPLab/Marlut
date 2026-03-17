use crate::lut_sp::VerificationRecorder;
use crate::lut_sp_malsec::mult_verification::{reconstruct_slice, ss_to_rss_shares};
use crate::rep3_core::{
    network::task::Direction,
    party::{CombinedCommStats, MainParty, Party, broadcast::BroadcastContext, error::MpcResult},
    share::{RssShare, RssShareGeneral},
};
use crate::share::{Field, bs_bool8::BsBool8, gf8::GF8};
use crate::util::aligned_vec::AlignedAllocator;
use rayon::prelude::*;
use std::ops::{Add, Mul};
use std::time::{Duration, Instant};
use std::vec;

use crate::lut_sp::our_offline::LUT256SPOffline; //generate_rndohv
use crate::rep3_core::share::HasZero;

type F = GF8;
// type F = UR8;

pub fn run_pro_10(
    table: &[F],
    index: &RssShare<F>,
    k: usize,
    party: &mut MainParty,
    context: &mut BroadcastContext,
    recorder: &mut impl VerificationRecorder<F>,
    simd: usize,
    // ) -> RssShare<F> {
) -> (Duration, CombinedCommStats, Duration, CombinedCommStats) {
    let n = 1 << k;
    let sqrt_n = (n as f64).sqrt() as usize;

    // generate r ,r' ?
    let start = Instant::now();
    let r0 = gen_random_bits(party, (k / 2) * simd, recorder);
    let r1 = gen_random_bits(party, (k / 2) * simd, recorder);

    // invoke twice randohv to get r er, r' er'
    let (ohv0, pr, prr) = LUT256SPOffline::generate_rndohv(
        party,
        (&r0.si, &r0.sii),
        k / 2, // sure?
        simd,
        recorder,
    )
    .unwrap();

    let (ohv1, pr, prr) = LUT256SPOffline::generate_rndohv(
        party,
        (&r1.si, &r1.sii),
        k / 2, // sure?
        simd,
        recorder,
    )
    .unwrap();

    let ohv1_share: Vec<RssShare<F>> = ohv1
        .e_si
        .iter()
        .zip(ohv1.e_sii.iter())
        .map(|(&si, &sii)| RssShareGeneral { si, sii })
        .collect();

    // compute new r
    let mut new_r = RssShare {
        si: Vec::with_capacity(r0.si.len() + r1.si.len()),
        sii: Vec::with_capacity(r0.sii.len() + r1.sii.len()),
    };
    new_r.si.extend_from_slice(&r0.si); // high
    new_r.si.extend_from_slice(&r1.si); // low

    new_r.sii.extend_from_slice(&r0.sii);
    new_r.sii.extend_from_slice(&r1.sii);

    let new_r: RssShare<F> = pack_bits_to_field_share(&new_r.si, &new_r.sii);

    // compute v + new_r = c and reconstruct

    let c_share = index.add(new_r);

    let pre_duration = start.elapsed();
    let pre_com: CombinedCommStats = party.io().reset_comm_stats();

    // online
    let cs_share = vec![c_share; simd];
    let start = Instant::now();
    let cs = reconstruct_slice(party, context, cs_share.as_slice()).unwrap();
    let c = cs[0];

    // compute g0 with c, T, f0

    let mut bar_g0_si = vec![F::ZERO; n];
    let mut bar_g0_sii = vec![F::ZERO; n];

    for i in 0..sqrt_n {
        for j in 0..sqrt_n {
            let idx = c.as_u8() as usize ^ (i * sqrt_n + j);
            bar_g0_si[i * sqrt_n + j] = ohv0.e_si[i] * table[idx];
            bar_g0_sii[i * sqrt_n + j] = ohv0.e_sii[i] * table[idx];
        }
    }

    let bar_g0: Vec<RssShare<F>> = bar_g0_si
        .iter()
        .zip(bar_g0_sii.iter())
        .map(|(si, sii)| RssShareGeneral { si: *si, sii: *sii })
        .collect();

    // weak dot prod g0 f1 get tv share

    let tvs_l: Vec<F> = (0..simd)
        .into_par_iter()
        .map(|i| extended_weak_inner_product(&bar_g0, &ohv1_share[i * sqrt_n..(i + 1) * sqrt_n]))
        .collect();

    let tvs = ss_to_rss_shares(party, &tvs_l);

    let online_duration = start.elapsed();
    println!("Online duration: {:?}", online_duration);
    let online_com = party.io().reset_comm_stats();

    (pre_duration, pre_com, online_duration, online_com)
}

fn extended_weak_inner_product<F: Field>(a: &[RssShare<F>], b: &[RssShare<F>]) -> F {
    let n = a.len();
    let m = (n as f64).sqrt() as usize; // m = sqrt(n)
    assert_eq!(b.len(), m, "b must have length sqrt(n)");

    //
    (0..m).fold(F::ZERO, |sum, i| {
        let block = &a[i * m..(i + 1) * m]; // 
        let block_product = block.iter().zip(b).fold(F::ZERO, |inner_sum, (x, y)| {
            (x.si + x.sii) * (y.si + y.sii) - x.si * y.si + inner_sum
        });
        sum + block_product
    })
}

pub fn gen_random_bits<T: Field>(
    party: &mut impl Party,
    n: usize,
    recorder: &mut impl VerificationRecorder<T>,
) -> RssShare<Vec<T>> {
    let i = party.party_id();
    let mut ai = vec![T::ZERO; n];
    let mut aii = vec![T::ZERO; n];
    let first_master_direction = if i == 1 {
        Direction::Previous
    } else {
        Direction::Next
    };
    let mut rcvs = vec![];
    if i == 0 {
        ai = party.generate_random_local::<T>(n);
        aii = party.generate_random_local::<T>(n);
        let mut a2 = vec![T::ZERO; n];
        let bits = party.generate_random_local::<BsBool8>((n + 7) / 8);
        for i in 0..n {
            let byte = i / 8;
            let bit = i % 8;
            if (bits[byte].0 & (1 << bit)) == 0 {
                a2[i] = -ai[i] - aii[i];
            } else {
                a2[i] = T::ONE - ai[i] - aii[i];
            }
        }

        party.send_field_slice(Direction::Next, &aii);
        party.send_field_slice(Direction::Next, &a2);
        party.send_field_slice(Direction::Previous, &a2);
        party.send_field_slice(Direction::Previous, &ai);
    } else {
        rcvs.push(party.receive_field_slice(first_master_direction, &mut ai));
        rcvs.push(party.receive_field_slice(first_master_direction, &mut aii));
    }

    let mut bi = vec![T::ZERO; n];
    let mut bii = vec![T::ZERO; n];
    let second_master_direction = if i == 0 {
        Direction::Next
    } else {
        Direction::Previous
    };

    if i == 1 {
        bi = party.generate_random_local::<T>(n);
        bii = party.generate_random_local::<T>(n);
        let mut b0 = vec![T::ZERO; n];
        let bits = party.generate_random_local::<BsBool8>((n + 7) / 8);
        for i in 0..n {
            let byte = i / 8;
            let bit = i % 8;
            if (bits[byte].0 & (1 << bit)) == 0 {
                b0[i] = -bi[i] - bii[i];
            } else {
                b0[i] = T::ONE - bi[i] - bii[i];
            }
        }

        party.send_field_slice(Direction::Previous, &b0);
        party.send_field_slice(Direction::Previous, &bi);
        party.send_field_slice(Direction::Next, &bii);
        party.send_field_slice(Direction::Next, &b0);
    } else {
        rcvs.push(party.receive_field_slice(second_master_direction, &mut bi));
        rcvs.push(party.receive_field_slice(second_master_direction, &mut bii));
    }

    for rcv in rcvs {
        rcv.rcv().unwrap();
    }

    let mut ci = vec::from_elem_in(T::ZERO, n, AlignedAllocator::<64>);
    let mut cii = vec::from_elem_in(T::ZERO, n, AlignedAllocator::<64>);
    mul_no_sync(party, &mut ci, &mut cii, &ai, &aii, &bi, &bii, recorder).unwrap();

    RssShare {
        si: ci.to_vec(),
        sii: cii.to_vec(),
    }
}

pub fn mul_no_sync<T: Field, P: Party>(
    party: &mut P,
    ci: &mut [T],
    cii: &mut [T],
    ai: &[T],
    aii: &[T],
    bi: &[T],
    bii: &[T],
    recorder: &mut impl VerificationRecorder<T>,
) -> MpcResult<()> {
    let alphas = recorder.generate_alpha(party, ci.len());
    for (i, alpha_i) in alphas.into_iter().enumerate() {
        ci[i] = ai[i] * bi[i] + ai[i] * bii[i] + aii[i] * bi[i] + alpha_i;
    }
    let rcv = party.receive_field_slice(Direction::Next, cii);
    party.send_field_slice(Direction::Previous, ci);
    rcv.rcv()?;
    Ok(())
}

pub fn pack_bits_to_field_share<F>(bits_si: &[F], bits_sii: &[F]) -> RssShare<F>
where
    F: Field + Copy + Add<Output = F> + Mul<Output = F>,
{
    assert_eq!(bits_si.len(), bits_sii.len());
    let k = bits_si.len();

    // pow = 2^0 in field (we will iterate bits from LSB to MSB)
    let mut pow = F::ONE;
    let mut si_acc = F::ZERO;
    let mut sii_acc = F::ZERO;

    // bits vector is [MSB, ..., LSB], so iterate reversed to get LSB first
    for i in (0..k).rev() {
        // accumulate: acc += bit * pow
        si_acc = si_acc + (bits_si[i] * pow);
        sii_acc = sii_acc + (bits_sii[i] * pow);

        // multiply pow by 2: pow = pow * 2 (in field)
        // pow + pow is valid since 2*pow = pow + pow
        pow = pow + pow;
    }

    RssShare {
        si: si_acc,
        sii: sii_acc,
    }
}
