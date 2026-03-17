use crate::lut_sp::VerificationRecorder;
use crate::lut_sp_malsec::mult_verification::reconstruct;
use crate::rep3_core::{
    network::task::Direction,
    party::{
        DigestExt, MainParty, Party,
        broadcast::{Broadcast, BroadcastContext},
        error::MpcResult,
    },
    share::RssShare,
};
use crate::share::{Field, bs_bool8::BsBool8};
use rand::{Rng, SeedableRng};

pub fn cut_and_bucket<T: Field + DigestExt + Clone>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    n: usize,
    recorder: &mut impl VerificationRecorder<T>,
    b: usize,
    c: usize,
) -> (Vec<T>, Vec<T>) {
    // 1. Generate a secret share of n random bits
    let mut triples = gen_random_bits(party, n * b + c, recorder);

    // 2. shuffle
    // let r = coin_flip::<T>(party, context);

    shuffle(
        triples.si.as_mut_slice(),
        triples.sii.as_mut_slice(),
        373244654,
    );

    let tr_c = RssShare {
        si: triples.si[0..c].to_vec(),
        sii: triples.sii[0..c].to_vec(),
    };

    let mut tr_b = RssShare {
        si: vec![],
        sii: vec![],
    };
    // 4. Divide the remaining shared values into buckets, with b values in each bucket
    let remaining = RssShare {
        si: triples.si[c..].to_vec(),
        sii: triples.sii[c..].to_vec(),
    };
    let resultsi = regroup_by_bucket(&remaining.si, b);
    let resultsii = regroup_by_bucket(&remaining.sii, b);

    let mut ci = vec![T::ZERO; b - 1];
    let mut cii = vec![T::ZERO; b - 1];

    // compute xor

    for j in 1..b {
        fun_xor(
            party,
            &mut ci,
            &mut cii,
            &resultsi[0],
            &resultsi[j],
            &resultsii[0],
            &resultsii[j],
            recorder,
        )
        .unwrap();

        tr_b.si.extend(ci.clone());
        tr_b.sii.extend(cii.clone());
    }

    let mut merged_si = tr_c.si.clone();
    merged_si.extend_from_slice(&tr_b.si);

    let mut merged_sii = tr_c.sii.clone();
    merged_sii.extend_from_slice(&tr_b.sii);

    let merged_share = RssShare {
        si: merged_si,
        sii: merged_sii,
    };

    let merged_open = open_bit(party, context, merged_share);

    merged_open
        .as_ref()
        .unwrap()
        .iter()
        .all(|&x| x == T::ZERO || x == T::ONE);

    println!("len: {}", resultsi[0].len());
    (resultsi[0].clone(), resultsii[0].clone()) // Take the first one of each bucket
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

    let mut ci = vec![T::ZERO; n];
    let mut cii = vec![T::ZERO; n];
    mul_no_sync(party, &mut ci, &mut cii, &ai, &aii, &bi, &bii, recorder).unwrap();

    RssShare { si: ci, sii: cii }
}

fn fun_xor<T: Field, P: Party>(
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
        let temp = ai[i] * bi[i] + ai[i] * bii[i] + aii[i] * bi[i];
        ci[i] = ai[i] + bi[i] - temp - temp + alpha_i;
    }
    let rcv = party.receive_field_slice(Direction::Next, cii);
    party.send_field_slice(Direction::Previous, ci);
    rcv.rcv()?;
    Ok(())
}

fn coin_flip<T: Field + DigestExt>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
) -> MpcResult<T> {
    let r: RssShare<T> = party.generate_random(1)[0];
    reconstruct(party, context, r)
}

fn shuffle<T, U>(a: &mut [T], b: &mut [U], seed: u64) {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    for i in (1..len).rev() {
        let j = rng.gen_range(0..=i);
        a.swap(i, j);
        b.swap(i, j);
    }
}

pub fn open_bit<F: Field + DigestExt>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    rho: RssShare<Vec<F>>,
) -> MpcResult<Vec<F>> {
    party.open_rss(context, &rho.si.as_slice(), &rho.sii.as_slice())
}

fn regroup_by_bucket<T: Clone>(v: &[T], b: usize) -> Vec<Vec<T>> {
    let mut result = vec![Vec::new(); b];
    for chunk in v.chunks(b) {
        for (i, val) in chunk.iter().enumerate() {
            result[i].push(val.clone());
        }
    }
    result
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
