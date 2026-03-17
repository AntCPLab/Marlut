use std::marker::PhantomData;
use std::vec;

use super::{RndOhvOutput, alloc_aligned_blocks};
use crate::lut_sp::VerificationRecorder;
use crate::rep3_core::network::task::Direction;
use crate::rep3_core::party::{Party, error::MpcResult};
use crate::share::Field;
use crate::share::bs_bool8::BsBool8;
use crate::util::aligned_vec::{AlignedAllocator, AlignedVec};
use itertools::izip;

pub struct LUT256SPOffline<T: Field>(PhantomData<T>);

impl<T: Field> LUT256SPOffline<T> {
    pub fn mul_no_sync<P: Party>(
        party: &mut P,
        ci: &mut [T],
        cii: &mut [T],
        ai: &[T],
        aii: &[T],
        bi: &[T],
        bii: &[T],
        recorder: &mut impl VerificationRecorder<T>,
    ) -> MpcResult<()> {
        debug_assert_eq!(ci.len(), ai.len());
        debug_assert_eq!(ci.len(), aii.len());
        debug_assert_eq!(ci.len(), bi.len());
        debug_assert_eq!(ci.len(), bii.len());
        debug_assert_eq!(ci.len(), cii.len());

        let alphas = recorder.generate_alpha_aligned(party, ci.len());
        let use_simd = T::IS_UR
            && T::NBITS == 8
            && cfg!(all(target_arch = "x86_64", target_feature = "avx512bw",));

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
        if use_simd {
            debug_assert!(ci.len() % 32 == 0);
            unsafe {
                use std::arch::x86_64::{
                    _mm256_add_epi8, _mm256_load_epi64, _mm256_store_epi64, _mm512_add_epi16,
                    _mm512_cvtepi16_epi8, _mm512_cvtepu8_epi16, _mm512_mullo_epi16,
                };

                let mut ci_ptr = ci as *mut [T] as *mut T as *mut i8;
                let mut alpha_ptr = &alphas[..] as *const [T] as *const T as *const i8;
                let mut ai_ptr = ai as *const [T] as *const T as *const i8;
                let mut aii_ptr = aii as *const [T] as *const T as *const i8;
                let mut bi_ptr = bi as *const [T] as *const T as *const i8;
                let mut bii_ptr = bii as *const [T] as *const T as *const i8;
                for _ in (0..ci.len()).step_by(32) {
                    let ai = _mm256_load_epi64(ai_ptr as *const i64);
                    let ai = _mm512_cvtepu8_epi16(ai);
                    let aii = _mm256_load_epi64(aii_ptr as *const i64);
                    let aii = _mm512_cvtepu8_epi16(aii);
                    let bi = _mm256_load_epi64(bi_ptr as *const i64);
                    let bi = _mm512_cvtepu8_epi16(bi);
                    let bii = _mm256_load_epi64(bii_ptr as *const i64);
                    let bii = _mm512_cvtepu8_epi16(bii);
                    let bi_plus_bii = _mm512_add_epi16(bi, bii);
                    let result = _mm512_add_epi16(
                        _mm512_mullo_epi16(ai, bi_plus_bii),
                        _mm512_mullo_epi16(aii, bi),
                    );
                    let result = _mm512_cvtepi16_epi8(result);
                    let alpha = _mm256_load_epi64(alpha_ptr as *const i64);
                    let result = _mm256_add_epi8(result, alpha);
                    _mm256_store_epi64(ci_ptr as *mut i64, result);

                    ci_ptr = ci_ptr.add(32);
                    alpha_ptr = alpha_ptr.add(32);
                    ai_ptr = ai_ptr.add(32);
                    aii_ptr = aii_ptr.add(32);
                    bi_ptr = bi_ptr.add(32);
                    bii_ptr = bii_ptr.add(32);
                }
            }
        }

        if !use_simd {
            for (i, alpha_i) in alphas.into_iter().enumerate() {
                ci[i] = ai[i] * bi[i] + ai[i] * bii[i] + aii[i] * bi[i] + alpha_i;
            }
        }
        let rcv = party.receive_field_slice(Direction::Next, cii);
        party.send_field_slice(Direction::Previous, ci);
        rcv.rcv()?;
        recorder.record_mul_triple(ai, aii, bi, bii, ci, cii);
        Ok(())
    }

    fn produce_bits_share(party: &mut impl Party, a2: &mut [T], ai: &[T], aii: &[T]) {
        let n = a2.len();
        let bits = party.generate_random_local_aligned::<BsBool8>((n + 7) / 8);

        let use_simd = T::IS_UR
            && (T::NBITS == 8 || T::NBITS == 16)
            && cfg!(all(target_arch = "x86_64", target_feature = "avx512bw",));
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
        if use_simd && T::NBITS == 8 {
            debug_assert!(n % 64 == 0);

            unsafe {
                use std::arch::x86_64::{
                    _mm512_add_epi8, _mm512_load_epi64, _mm512_maskz_set1_epi8, _mm512_store_epi64,
                    _mm512_sub_epi8,
                };

                let mut a2_ptr = a2 as *mut [T] as *mut T as *mut i8;
                let mut ai_ptr = ai as *const [T] as *const T as *const i8;
                let mut aii_ptr = aii as *const [T] as *const T as *const i8;
                let mut bits_ptr = &bits[..] as *const [BsBool8] as *const BsBool8 as *const u8;

                for _ in (0..n).step_by(64) {
                    let ai = _mm512_load_epi64(ai_ptr as *const i64);
                    let aii = _mm512_load_epi64(aii_ptr as *const i64);
                    let mut array = [0u8; 8];
                    std::ptr::copy_nonoverlapping(
                        bits_ptr,
                        &mut array as *mut [_] as *mut u8,
                        array.len(),
                    );
                    let mask: u64 = u64::from_le_bytes(array);
                    let ai_plus_aii = _mm512_add_epi8(ai, aii);
                    let bits = _mm512_maskz_set1_epi8(mask, 1);
                    let result = _mm512_sub_epi8(bits, ai_plus_aii);
                    _mm512_store_epi64(a2_ptr as *mut i64, result);

                    a2_ptr = a2_ptr.add(64);
                    ai_ptr = ai_ptr.add(64);
                    aii_ptr = aii_ptr.add(64);
                    bits_ptr = bits_ptr.add(8);
                }
            }
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
        if use_simd && T::NBITS == 16 {
            debug_assert!(n % 32 == 0);

            unsafe {
                use std::arch::x86_64::{
                    _mm512_add_epi16, _mm512_load_epi64, _mm512_maskz_set1_epi16,
                    _mm512_store_epi64, _mm512_sub_epi16,
                };

                let mut a2_ptr = a2 as *mut [T] as *mut T as *mut i16;
                let mut ai_ptr = ai as *const [T] as *const T as *const i16;
                let mut aii_ptr = aii as *const [T] as *const T as *const i16;
                let mut bits_ptr = &bits[..] as *const [BsBool8] as *const BsBool8 as *const u8;

                for _ in (0..n).step_by(32) {
                    let ai = _mm512_load_epi64(ai_ptr as *const i64);
                    let aii = _mm512_load_epi64(aii_ptr as *const i64);
                    let mut array = [0u8; 4];
                    std::ptr::copy_nonoverlapping(
                        bits_ptr,
                        &mut array as *mut [_] as *mut u8,
                        array.len(),
                    );
                    let mask: u32 = u32::from_le_bytes(array);
                    let ai_plus_aii = _mm512_add_epi16(ai, aii);
                    let bits = _mm512_maskz_set1_epi16(mask, 1);
                    let result = _mm512_sub_epi16(bits, ai_plus_aii);
                    _mm512_store_epi64(a2_ptr as *mut i64, result);

                    a2_ptr = a2_ptr.add(32);
                    ai_ptr = ai_ptr.add(32);
                    aii_ptr = aii_ptr.add(32);
                    bits_ptr = bits_ptr.add(4);
                }
            }
        }

        if !use_simd {
            for i in 0..n {
                let byte = i / 8;
                let bit = i % 8;
                if (bits[byte].0 & (1 << bit)) == 0 {
                    a2[i] = -ai[i] - aii[i];
                } else {
                    a2[i] = T::ONE - ai[i] - aii[i];
                }
            }
        }
    }

    pub(super) fn random_bits<const MAL: bool>(
        party: &mut impl Party,
        n: usize,
        recorder: &mut impl VerificationRecorder<T>,
    ) -> (AlignedVec<T, 64>, AlignedVec<T, 64>) {
        let i = party.party_id();
        let mut ai = vec::from_elem_in(T::ZERO, n, AlignedAllocator::<64>);
        let mut aii = vec::from_elem_in(T::ZERO, n, AlignedAllocator::<64>);
        let first_master_direction = if i == 1 {
            Direction::Previous
        } else {
            Direction::Next
        };

        let mut rcvs = vec![];
        if i == 0 {
            ai = party.generate_random_local_aligned::<T>(n);
            aii = party.generate_random_local_aligned::<T>(n);
            let mut a2 = vec::from_elem_in(T::ZERO, n, AlignedAllocator::<64>);

            Self::produce_bits_share(party, &mut a2, &ai, &aii);

            party.send_field_slice(Direction::Next, &aii);
            party.send_field_slice(Direction::Next, &a2);
            party.send_field_slice(Direction::Previous, &a2);
            party.send_field_slice(Direction::Previous, &ai);
        } else {
            rcvs.push(party.receive_field_slice(first_master_direction, &mut ai));
            rcvs.push(party.receive_field_slice(first_master_direction, &mut aii));
        }

        let mut bi = vec::from_elem_in(T::ZERO, n, AlignedAllocator::<64>);
        let mut bii = vec::from_elem_in(T::ZERO, n, AlignedAllocator::<64>);
        let second_master_direction = if i == 0 {
            Direction::Next
        } else {
            Direction::Previous
        };

        if i == 1 {
            bi = party.generate_random_local_aligned::<T>(n);
            bii = party.generate_random_local_aligned::<T>(n);
            let mut b0 = vec::from_elem_in(T::ZERO, n, AlignedAllocator::<64>);
            Self::produce_bits_share(party, &mut b0, &bi, &bii);

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
        Self::mul_no_sync(party, &mut ci, &mut cii, &ai, &aii, &bi, &bii, recorder).unwrap();

        if MAL {
            let one = party.constant(T::ONE);
            // Reuse it a bit
            for (ai, aii, ci, cii) in izip!(&mut ai, &mut aii, &ci, &cii) {
                *ai = one.si - *ci;
                *aii = one.sii - *cii;
            }
            Self::mul_no_sync(party, &mut bi, &mut bii, &ai, &aii, &ci, &cii, recorder).unwrap();
            recorder.record_zero_check(&bi, &bii);
        }
        (ci, cii)
    }

    fn simple_mul<P: Party>(
        party: &mut P,
        ci: &mut [T],
        cii: &mut [T],
        ai: &[T],
        aii: &[T],
        bi: &[T],
        bii: &[T],
        recorder: &mut impl VerificationRecorder<T>,
    ) -> MpcResult<()> {
        let alphas = recorder.generate_alpha_aligned(party, ci.len());

        let use_simd = T::IS_UR
            && T::NBITS == 8
            && cfg!(all(target_arch = "x86_64", target_feature = "avx512bw",));

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
        if use_simd {
            debug_assert!(bi.len() % 32 == 0);
            debug_assert!(ci.len() % 32 == 0);
            unsafe {
                use std::arch::x86_64::{
                    _mm256_add_epi8, _mm256_load_epi64, _mm256_store_epi64, _mm512_add_epi16,
                    _mm512_cvtepi16_epi8, _mm512_cvtepu8_epi16, _mm512_mullo_epi16,
                };

                let mut ci_ptr = ci as *mut [T] as *mut T as *mut i8;
                let mut alpha_ptr = &alphas[..] as *const [T] as *const T as *const i8;
                let mut ai_ptr = ai as *const [T] as *const T as *const i8;
                let mut aii_ptr = aii as *const [T] as *const T as *const i8;
                let mut bi_ptr = bi as *const [T] as *const T as *const i8;
                let mut bii_ptr = bii as *const [T] as *const T as *const i8;
                for i in (0..ci.len()).step_by(32) {
                    if i % bi.len() == 0 {
                        bi_ptr = bi as *const [T] as *const T as *const i8;
                        bii_ptr = bii as *const [T] as *const T as *const i8;
                    }
                    let ai = _mm256_load_epi64(ai_ptr as *const i64);
                    let ai = _mm512_cvtepu8_epi16(ai);
                    let aii = _mm256_load_epi64(aii_ptr as *const i64);
                    let aii = _mm512_cvtepu8_epi16(aii);
                    let bi = _mm256_load_epi64(bi_ptr as *const i64);
                    let bi = _mm512_cvtepu8_epi16(bi);
                    let bii = _mm256_load_epi64(bii_ptr as *const i64);
                    let bii = _mm512_cvtepu8_epi16(bii);
                    let bi_plus_bii = _mm512_add_epi16(bi, bii);
                    let result = _mm512_add_epi16(
                        _mm512_mullo_epi16(ai, bi_plus_bii),
                        _mm512_mullo_epi16(aii, bi),
                    );
                    let result = _mm512_cvtepi16_epi8(result);
                    let alpha = _mm256_load_epi64(alpha_ptr as *const i64);
                    let result = _mm256_add_epi8(result, alpha);
                    _mm256_store_epi64(ci_ptr as *mut i64, result);

                    ci_ptr = ci_ptr.add(32);
                    alpha_ptr = alpha_ptr.add(32);
                    ai_ptr = ai_ptr.add(32);
                    aii_ptr = aii_ptr.add(32);
                    bi_ptr = bi_ptr.add(32);
                    bii_ptr = bii_ptr.add(32);
                }
            }
        }

        if !use_simd {
            for (i, alpha_i) in alphas.into_iter().enumerate() {
                ci[i] = ai[i] * bi[i % bi.len()]
                    + ai[i] * bii[i % bii.len()]
                    + aii[i] * bi[i % bi.len()]
                    + alpha_i;
            }
        }

        let rcv = party.receive_field_slice(Direction::Next, cii);
        party.send_field_slice(Direction::Previous, ci);
        rcv.rcv()?;
        recorder.record_mul_triple(ai, aii, bi, bii, ci, cii);
        Ok(())
    }

    pub fn generate_rndohv<P: Party>(
        party: &mut P,
        (r_i, r_ii): (&[T], &[T]),
        dim_bits: usize,
        num_lookups: usize,
        recorder: &mut impl VerificationRecorder<T>,
    ) -> MpcResult<(
        RndOhvOutput<T>,
        Vec<AlignedVec<T, 64>>,
        Vec<AlignedVec<T, 64>>,
    )> {
        let r_i = r_i
            .chunks_exact(num_lookups)
            .map(|x| {
                let mut vec = Vec::with_capacity_in(x.len(), AlignedAllocator::<64>);
                vec.extend(x);
                vec
            })
            .collect::<Vec<_>>();
        let r_ii = r_ii
            .chunks_exact(num_lookups)
            .map(|x| {
                let mut vec = Vec::with_capacity_in(x.len(), AlignedAllocator::<64>);
                vec.extend(x);
                vec
            })
            .collect::<Vec<_>>();

        let size_bytes = (1 << dim_bits) * num_lookups;

        let (mut e_si, mut e_sii) = alloc_aligned_blocks(size_bytes);
        Self::generate_ohv_impl(
            party,
            r_i.clone(),
            r_ii.clone(),
            1 << dim_bits,
            &mut e_si,
            &mut e_sii,
            recorder,
        )?;

        Ok((
            RndOhvOutput {
                e_si,
                e_sii,
                dim_bits,
            },
            r_i,
            r_ii,
        ))
    }

    fn generate_ohv_impl<P: Party>(
        party: &mut P,
        mut bits_si: Vec<AlignedVec<T, 64>>,
        mut bits_sii: Vec<AlignedVec<T, 64>>,
        n: usize,
        out_si: &mut [T],
        out_sii: &mut [T],
        recorder: &mut impl VerificationRecorder<T>,
    ) -> MpcResult<()> {
        let num_lookups = bits_si[0].len();
        if n == 2 {
            debug_assert_eq!(bits_si.len(), 1);
            let one = party.constant(T::ONE);
            for i in 0..num_lookups {
                out_si[i] = one.si - bits_si[0][i];
                out_sii[i] = one.sii - bits_sii[0][i];
                out_si[i + num_lookups] = bits_si[0][i];
                out_sii[i + num_lookups] = bits_sii[0][i];
            }
        } else {
            let msb_si = bits_si.pop().unwrap();
            let msb_sii = bits_sii.pop().unwrap();
            Self::generate_ohv_impl(party, bits_si, bits_sii, n / 2, out_si, out_sii, recorder)?;

            let prev_len = num_lookups * n / 2;
            let (left_si, right_si) = out_si.split_at_mut(prev_len);
            let right_si = &mut right_si[..prev_len];
            let (left_sii, right_sii) = out_sii.split_at_mut(prev_len);
            let right_sii = &mut right_sii[..prev_len];
            let len = prev_len - num_lookups;
            Self::simple_mul(
                party,
                &mut right_si[..len],
                &mut right_sii[..len],
                &left_si[..len],
                &left_sii[..len],
                &msb_si,
                &msb_sii,
                recorder,
            )?;

            for i in 0..num_lookups {
                let e_last_si = msb_si[i]
                    - right_si[i..]
                        .iter()
                        .step_by(num_lookups)
                        .cloned()
                        .sum::<T>();
                let e_last_sii = msb_sii[i]
                    - right_sii[i..]
                        .iter()
                        .step_by(num_lookups)
                        .cloned()
                        .sum::<T>();
                right_si[right_si.len() - num_lookups + i] = e_last_si;
                right_sii[right_sii.len() - num_lookups + i] = e_last_sii;
            }

            for (left_si, right_si, left_sii, right_sii) in
                izip!(left_si, right_si, left_sii, right_sii)
            {
                *left_si -= *right_si;
                *left_sii -= *right_sii;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub mod test {
    use std::fmt::Debug;

    use crate::{
        lut_sp::{NoVerificationRecording, alloc_aligned_blocks, our_offline::LUT256SPOffline},
        rep3_core::{
            share::{HasZero, RssShare},
            test::TestSetup,
        },
        share::{
            Field,
            unsigned_ring::{UR8, UR16},
        },
        util::{
            ArithmeticBlackBox,
            aligned_vec::{AlignedAllocator, AlignedVec},
        },
    };
    use itertools::{MultiUnzip, izip};
    use rand::{RngCore, thread_rng};

    use crate::{
        chida::{ChidaParty, online::test::ChidaSetup},
        share::test::secret_share_vector,
    };

    fn reconstruct_and_check_rndohv_bit(
        sii1: &AlignedVec<UR8, 64>,
        sii2: &AlignedVec<UR8, 64>,
        sii3: &AlignedVec<UR8, 64>,
        input: &Vec<Vec<UR8>>,
        num_bits: usize,
        num_instances: usize,
    ) {
        let mut expected_index = vec![0; num_instances];
        for i in 0..num_bits {
            for j in 0..num_instances {
                expected_index[j] += (1 << i) * (input[i][j].0.0 as usize);
            }
        }

        for i in 0..(1 << num_bits) {
            for j in 0..num_instances {
                let result = (sii1[i * num_instances + j].0)
                    + (sii2[i * num_instances + j].0)
                    + (sii3[i * num_instances + j].0);

                if i == expected_index[j] {
                    assert_eq!(result.0, 1);
                } else {
                    assert_eq!(result.0, 0);
                }
            }
        }
    }

    fn consistent<F: Field + Debug>(
        share1_si: F,
        share1_sii: F,
        share2_si: F,
        share2_sii: F,
        share3_si: F,
        share3_sii: F,
    ) {
        assert_eq!(share1_sii, share2_si,);
        assert_eq!(share2_sii, share3_si,);
        assert_eq!(share3_sii, share1_si,);
    }

    fn test_generate_ohv_output_helper(input_bits: usize, num_instances: usize) {
        let mut rng = thread_rng();
        let input = (0..input_bits)
            .map(|_| {
                (0..num_instances)
                    .map(|_| {
                        if rng.next_u32() % 2 == 0 {
                            UR8::ZERO
                        } else {
                            UR8::ONE
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let shares: (Vec<_>, Vec<_>, Vec<_>) = input
            .iter()
            .map(|input| secret_share_vector(&mut rng, input.iter()))
            .multiunzip();
        let program = |shares: Vec<Vec<RssShare<UR8>>>| {
            move |p: &mut ChidaParty| {
                let size_bytes = (1 << input_bits) * num_instances;

                let (bits_si, bits_sii): (Vec<_>, Vec<_>) = shares
                    .iter()
                    .map(|shares| {
                        let mut si =
                            std::vec::from_elem_in(UR8::ZERO, shares.len(), AlignedAllocator::<64>);
                        let mut sii =
                            std::vec::from_elem_in(UR8::ZERO, shares.len(), AlignedAllocator::<64>);
                        for (si, sii, share) in izip!(&mut si, &mut sii, shares) {
                            *si = share.si;
                            *sii = share.sii;
                        }
                        (si, sii)
                    })
                    .unzip();

                let (mut e_si, mut e_sii) = alloc_aligned_blocks(size_bytes);
                LUT256SPOffline::generate_ohv_impl(
                    p.as_party_mut(),
                    bits_si,
                    bits_sii,
                    1 << input_bits,
                    &mut e_si,
                    &mut e_sii,
                    &mut NoVerificationRecording,
                )
                .unwrap();
                (e_si, e_sii)
            }
        };
        let (((si1, sii1), _), ((si2, sii2), _), ((si3, sii3), _)) =
            ChidaSetup::localhost_setup(program(shares.0), program(shares.1), program(shares.2));
        izip!(&si1, &sii1, &si2, &sii2, &si3, &sii3).for_each(
            |(si1, sii1, si2, sii2, si3, sii3)| {
                consistent(*si1, *sii1, *si2, *sii2, *si3, *sii3);
            },
        );

        reconstruct_and_check_rndohv_bit(&sii1, &sii2, &sii3, &input, input_bits, num_instances);
    }

    #[test]
    fn test_generate_ohv_output() {
        test_generate_ohv_output_helper(1, 32);
        test_generate_ohv_output_helper(2, 32);
        test_generate_ohv_output_helper(3, 32);
        test_generate_ohv_output_helper(16, 32);
        test_generate_ohv_output_helper(13, 32);
    }

    fn test_random_bits_helper<T: Field + Debug>() {
        let n = 128;

        let program = move |p: &mut ChidaParty| {
            let ret = LUT256SPOffline::<T>::random_bits::<false>(
                p.as_party_mut(),
                n,
                &mut NoVerificationRecording,
            );
            <ChidaParty as ArithmeticBlackBox<T>>::io(p).wait_for_completion();
            ret
        };
        let (((si1, sii1), _), ((si2, sii2), _), ((si3, sii3), _)) =
            ChidaSetup::localhost_setup(program, program, program);
        izip!(&si1, &sii1, &si2, &sii2, &si3, &sii3).for_each(
            |(si1, sii1, si2, sii2, si3, sii3)| {
                consistent(*si1, *sii1, *si2, *sii2, *si3, *sii3);
                let sum = *si1 + *si2 + *si3;
                assert!(sum == T::ZERO || sum == T::ONE);
            },
        );
    }

    #[test]
    fn test_random_bits() {
        test_random_bits_helper::<UR8>();
        test_random_bits_helper::<UR16>();
    }
}
