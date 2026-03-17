use std::{mem::take, time::Instant};

use itertools::{Itertools, izip, zip_eq};
use rayon::prelude::*;

use crate::{
    aes::{AesVariant, GF8InvBlackBox},
    lut_sp_boolean::{
        VerificationRecorder,
        lut256_tables::GF8InvTable,
        our_offline::{self, reorder_r},
    },
    lut_sp_boolean_malsec::mult_verification::verify_multiplication_triples,
    rep3_core::{
        network::task::{Direction, IoLayerOwned},
        party::{
            MainParty, Party,
            broadcast::Broadcast,
            error::{MpcError, MpcResult},
        },
        share::{HasZero, RssShare, RssShareVec},
    },
    share::mersenne61::Mersenne61,
    util::aligned_vec::{AlignedAllocator, AlignedVec},
};
use crate::{
    // aes::{AesVariant, GF8InvBlackBox},
    chida::ChidaParty,
    share::{Field, bs_bool8::BsBool8, gf8::GF8},
    util::ArithmeticBlackBox,
};

use super::{LUT256SP, RndOhvPrep};

impl<Recorder: VerificationRecorder, const MAL: bool> GF8InvBlackBox for LUT256SP<Recorder, MAL> {
    fn constant(&self, value: GF8) -> RssShare<GF8> {
        self.inner.constant(value)
    }

    fn do_preprocessing(
        &mut self,
        n_keys: usize,
        n_blocks: usize,
        variant: AesVariant,
    ) -> MpcResult<()> {
        let time = Instant::now();
        assert_eq!(n_keys, 0); // For now

        let res = self.preprocess::<GF8InvTable>(16 * n_blocks, variant.n_rounds());
        println!("Preprocessing time {:?}", time.elapsed());
        res
    }

    fn gf8_inv(&mut self, si: &mut [GF8], sii: &mut [GF8]) -> MpcResult<()> {
        let now = Instant::now();
        let (si, sii): (&mut [BsBool8], &mut [BsBool8]) = unsafe {
            (
                &mut *(si as *mut [GF8] as *mut [BsBool8]),
                &mut *(sii as *mut [GF8] as *mut [BsBool8]),
            )
        };
        self.lut::<true, GF8InvTable>(si, sii)?;
        let (new_si, new_sii) = self.temp_vecs.as_ref().unwrap();
        si.copy_from_slice(&new_si);
        sii.copy_from_slice(&new_sii);

        let time = now.elapsed();
        self.lut_time += time;

        if self.inner.party_index() == 1 {
            println!(
                "Length: {}, LUT Time: {:?},Total Time: {:?}",
                si.len(),
                time,
                self.lut_time
            );
        }
        Ok(())
    }

    fn main_party_mut(&mut self) -> &mut MainParty {
        self.inner.as_party_mut()
    }
}

fn bitwise_repeat(v: &[BsBool8], output_bits: usize) -> Vec<BsBool8> {
    if output_bits == 1 {
        return v.to_vec();
    }
    let mut out = vec![BsBool8::ZERO; output_bits * v.len()];
    match output_bits {
        4 => {
            v.iter()
                .zip(out.chunks_exact_mut(output_bits))
                .for_each(|(v, out)| bitwise_repeat_impl::<4>(*v, out));
        }
        2 => {
            v.iter()
                .zip(out.chunks_exact_mut(output_bits))
                .for_each(|(v, out)| bitwise_repeat_impl::<2>(*v, out));
        }
        8 => {
            v.iter().zip(out.chunks_exact_mut(8)).for_each(|(v, out)| {
                bitwise_repeat_byte(*v, out);
            });
        }
        _ => {
            debug_assert!(output_bits % 8 == 0);
            v.iter()
                .zip(out.chunks_exact_mut(output_bits))
                .for_each(|(v, out)| {
                    bitwise_repeat_bytes(*v, out, output_bits / 8);
                });
        }
    }
    out
}

fn bitwise_repeat_impl<const OUTPUT_BITS: usize>(v: BsBool8, out: &mut [BsBool8]) {
    debug_assert_eq!(out.len(), OUTPUT_BITS);
    for bit in 0..8 {
        let byte_index = bit * OUTPUT_BITS / 8;
        let bit_index = bit * OUTPUT_BITS % 8;
        let mask = (1 << OUTPUT_BITS) - 1;
        if (v.0 & (1 << bit)) != 0 {
            out[byte_index].0 |= mask << bit_index;
        }
    }
}

fn bitwise_repeat_byte(v: BsBool8, out: &mut [BsBool8]) {
    debug_assert_eq!(out.len(), 8);
    for bit in 0..8 {
        if (v.0 & (1 << bit)) != 0 {
            out[bit] = BsBool8::ONE;
        }
    }
}

fn bitwise_repeat_bytes(v: BsBool8, out: &mut [BsBool8], output_bytes: usize) {
    debug_assert_eq!(out.len(), output_bytes * 8);
    for bit in 0..8 {
        if (v.0 & (1 << bit)) != 0 {
            out[bit * output_bytes..(bit + 1) * output_bytes].fill(BsBool8::ONE);
        }
    }
}

// Everything here is easily auto-vectorizable, but LLVM avoids 512-bit operations unless 512-bit
// intrinsics are explicitly used.
// In my experiments, using them is very marginally faster, so here we are.

pub fn xor_block(mut a: &mut [BsBool8], mut b: &[BsBool8]) {
    let len = a.len();

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f",))]
    if len >= 64 {
        unsafe {
            use std::arch::x86_64::{_mm512_load_epi64, _mm512_store_epi64, _mm512_xor_epi64};
            let mut a_ptr = a as *mut [BsBool8] as *mut i64;
            let mut b_ptr = b as *const [BsBool8] as *const i64;
            for _ in (0..len).step_by(64) {
                let x = _mm512_load_epi64(a_ptr);
                let y = _mm512_load_epi64(b_ptr);
                let result = _mm512_xor_epi64(x, y);
                _mm512_store_epi64(a_ptr, result);
                a_ptr = a_ptr.offset(8);
                b_ptr = b_ptr.offset(8);
            }
        }
        if len % 64 == 0 {
            return;
        } else {
            let remainder_start = len / 64 * 64;
            a = &mut a[remainder_start..];
            b = &b[remainder_start..];
        }
    }

    for (x, y) in zip_eq(a, b) {
        *x += *y;
    }
}

pub fn xor_blocks(mut a: &mut [BsBool8], mut b: &[BsBool8], mut c: &[BsBool8]) {
    let len = a.len();

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    if len >= 64 {
        unsafe {
            use std::arch::x86_64::{
                _mm512_load_epi64, _mm512_store_epi64, _mm512_ternarylogic_epi64,
            };
            let mut a_ptr = a as *mut [BsBool8] as *mut i64;
            let mut b_ptr = b as *const [BsBool8] as *const i64;
            let mut c_ptr = c as *const [BsBool8] as *const i64;
            for _ in (0..len).step_by(64) {
                let x = _mm512_load_epi64(a_ptr);
                let y = _mm512_load_epi64(b_ptr);
                let z = _mm512_load_epi64(c_ptr);
                let result = _mm512_ternarylogic_epi64::<0x96>(x, y, z);
                _mm512_store_epi64(a_ptr, result);
                a_ptr = a_ptr.offset(8);
                b_ptr = b_ptr.offset(8);
                c_ptr = c_ptr.offset(8);
            }
        }
        if len % 64 == 0 {
            return;
        } else {
            let remainder_start = len / 64 * 64;
            a = &mut a[remainder_start..];
            b = &b[remainder_start..];
            c = &c[remainder_start..];
        }
    }

    for (x, y, z) in izip!(a, b, c) {
        *x += *y + *z;
    }
}

unsafe fn sub_array<T, const N: usize, const SLICE_SIZE: usize>(
    a: &[T; N],
    index: usize,
) -> &[T; SLICE_SIZE] {
    debug_assert!(SLICE_SIZE * index < N);
    let ptr = a as *const [T] as *const T;
    let array_start = ptr.offset((index * SLICE_SIZE) as isize);
    &*(array_start as *const [T; SLICE_SIZE])
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
pub fn inner_product_large_ss_opt_byte<
    const DIM_BITS: usize,
    const BYTES_PER_LOOKUP: usize,
    const IS_PRESHIFTED: bool,
>(
    party: &mut impl Party,
    ai: &mut AlignedVec<BsBool8, 64>,
    aii: &AlignedVec<BsBool8, 64>,
    bi: &[BsBool8],
    bii: &[BsBool8],
    orig_bi: &[BsBool8],
    orig_bii: &[BsBool8],
    shifts: &[usize],
    recorder: &mut impl VerificationRecorder,
) -> MpcResult<()>
where
    [(); BYTES_PER_LOOKUP * 8]:,
    [(); BYTES_PER_LOOKUP * 64]:,
{
    match BYTES_PER_LOOKUP {
        16 => inner_product_large_ss_opt_byte_16::<DIM_BITS, BYTES_PER_LOOKUP, IS_PRESHIFTED>(
            party, ai, aii, bi, bii, orig_bi, orig_bii, shifts, recorder,
        ),
        8 => inner_product_large_ss_opt_byte_8::<DIM_BITS, BYTES_PER_LOOKUP, IS_PRESHIFTED>(
            party, ai, aii, bi, bii, orig_bi, orig_bii, shifts, recorder,
        ),
        _ => unreachable!(),
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
fn inner_product_large_ss_opt_byte_16<
    const DIM_BITS: usize,
    const BYTES_PER_LOOKUP: usize,
    const IS_PRESHIFTED: bool,
>(
    party: &mut impl Party,
    ai: &mut AlignedVec<BsBool8, 64>,
    aii: &AlignedVec<BsBool8, 64>,
    bi: &[BsBool8],
    bii: &[BsBool8],
    orig_bi: &[BsBool8],
    orig_bii: &[BsBool8],
    shifts: &[usize],
    recorder: &mut impl VerificationRecorder,
) -> MpcResult<()>
where
    [(); BYTES_PER_LOOKUP * 8]:,
    [(); BYTES_PER_LOOKUP * 64]:,
{
    use std::arch::x86_64::{
        __m512i, _cvtu64_mask64, _mm512_load_epi64, _mm512_maskz_loadu_epi8,
        _mm512_permutex2var_epi8, _mm512_set_epi8, _mm512_shuffle_i64x2, _mm512_store_epi64,
        _mm512_ternarylogic_epi64, _mm512_unpackhi_epi64, _mm512_unpacklo_epi64, _mm512_xor_epi64,
    };

    use crate::util::aligned_vec::AlignedAllocator;

    debug_assert_eq!(BYTES_PER_LOOKUP, 1 << DIM_BITS);

    let num_lookups = ai.len() / BYTES_PER_LOOKUP;
    let mut result = Vec::new_in(AlignedAllocator::<64>);
    result.reserve(num_lookups);
    result.extend(recorder.generate_alpha(party, num_lookups));

    let (perm_base_0, perm_base_1) = unsafe {
        (
            _mm512_set_epi8(
                126, 110, 94, 78, 62, 46, 30, 14, 124, 108, 92, 76, 60, 44, 28, 12, 122, 106, 90,
                74, 58, 42, 26, 10, 120, 104, 88, 72, 56, 40, 24, 8, 118, 102, 86, 70, 54, 38, 22,
                6, 116, 100, 84, 68, 52, 36, 20, 4, 114, 98, 82, 66, 50, 34, 18, 2, 112, 96, 80,
                64, 48, 32, 16, 0,
            ),
            _mm512_set_epi8(
                127, 111, 95, 79, 63, 47, 31, 15, 125, 109, 93, 77, 61, 45, 29, 13, 123, 107, 91,
                75, 59, 43, 27, 11, 121, 105, 89, 73, 57, 41, 25, 9, 119, 103, 87, 71, 55, 39, 23,
                7, 117, 101, 85, 69, 53, 37, 21, 5, 115, 99, 83, 67, 51, 35, 19, 3, 113, 97, 81,
                65, 49, 33, 17, 1,
            ),
        )
    };

    debug_assert!(ai.len() % (BYTES_PER_LOOKUP * 64) == 0);
    debug_assert_eq!(BYTES_PER_LOOKUP, 16); // Temporary test
    (
        result.as_chunks_mut::<64>().0,
        ai.as_chunks::<{ BYTES_PER_LOOKUP * 64 }>().0,
        aii.as_chunks::<{ BYTES_PER_LOOKUP * 64 }>().0,
        bi.as_chunks::<{ BYTES_PER_LOOKUP * 8 }>().0,
        bii.as_chunks::<{ BYTES_PER_LOOKUP * 8 }>().0,
    )
        .into_par_iter()
        .for_each(|(out, ai, aii, bi, bii)| {
            fn base_compute(
                ai: &[BsBool8; 64],
                aii: &[BsBool8; 64],
                bi: &[BsBool8; 8],
                bii: &[BsBool8; 8],
            ) -> __m512i {
                unsafe {
                    let bi = u64::from_le_bytes(core::mem::transmute_copy(bi));
                    let bii = u64::from_le_bytes(core::mem::transmute_copy(bii));
                    let mask_bi_bii = _cvtu64_mask64(bi ^ bii);
                    let mask_bi = _cvtu64_mask64(bi);

                    let ai =
                        _mm512_maskz_loadu_epi8(mask_bi_bii, ai as *const [BsBool8] as *const i8);
                    let aii =
                        _mm512_maskz_loadu_epi8(mask_bi, aii as *const [BsBool8] as *const i8);
                    _mm512_xor_epi64(ai, aii)
                }
            }
            let base_compute_idx = |index: usize| -> __m512i {
                unsafe {
                    base_compute(
                        sub_array(ai, index),
                        sub_array(aii, index),
                        sub_array(bi, index),
                        sub_array(bii, index),
                    )
                }
            };

            // a and b each contain 4 output_bytes * 16 bytes to be horizontally XORed
            // Returns 8 instances * 8 output_bytes, the 8 output_instances need to be XORed together
            let merge_base = |a, b| unsafe {
                let left = _mm512_permutex2var_epi8(a, perm_base_0, b);
                let right = _mm512_permutex2var_epi8(a, perm_base_1, b);
                _mm512_xor_epi64(left, right)
            };

            // TODO: Check if it's worth merging these, or if it's better to extract and sum up

            // Returns 4 instances * 16 output_bytes
            let merge_8 = |a, b| unsafe {
                let left = _mm512_unpacklo_epi64(a, b);
                let right = _mm512_unpackhi_epi64(a, b);
                _mm512_xor_epi64(left, right)
            };

            // Returns 4 instances * 16 bytes, such that first two should be added up to form the first
            // 16 output bytes, and the second two to form the second 16 output bytes
            let merge_16 = |a, b| unsafe {
                let left = _mm512_shuffle_i64x2::<0x88>(a, b);
                let right = _mm512_shuffle_i64x2::<0xdd>(a, b);
                _mm512_xor_epi64(left, right)
            };

            // Returns 1 instance * 64 output bytes
            let merge_32 = |a, b, alpha| unsafe {
                let left = _mm512_shuffle_i64x2::<0x88>(a, b);
                let right = _mm512_shuffle_i64x2::<0xdd>(a, b);
                _mm512_ternarylogic_epi64::<0x96>(left, right, alpha)
            };

            let produce_4 = |offset: usize| {
                let result0 = base_compute_idx(offset + 0);
                let result1 = base_compute_idx(offset + 1);
                let result01 = merge_base(result0, result1);

                let result2 = base_compute_idx(offset + 2);
                let result3 = base_compute_idx(offset + 3);
                let result23 = merge_base(result2, result3);

                merge_8(result01, result23)
            };

            let result_0 = produce_4(0);
            let result_4 = produce_4(4);
            let result_first8 = merge_16(result_0, result_4);

            let result_8 = produce_4(8);
            let result_12 = produce_4(12);
            let result_last8 = merge_16(result_8, result_12);

            let alpha = unsafe { _mm512_load_epi64(out as *const [BsBool8] as *const i64) };
            let result = merge_32(result_first8, result_last8, alpha);
            unsafe {
                _mm512_store_epi64(out as *mut [BsBool8] as *mut i64, result);
            }
        });

    recorder.record_ip_triple(
        &ai,
        &aii,
        orig_bi,
        orig_bii,
        &result,
        if IS_PRESHIFTED { &[] } else { shifts },
        8,
        16 * 8,
    );

    ai.truncate(result.len());
    ai.copy_from_slice(&result);
    Ok(())
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
fn inner_product_large_ss_opt_byte_8<
    const DIM_BITS: usize,
    const BYTES_PER_LOOKUP: usize,
    const IS_PRESHIFTED: bool,
>(
    party: &mut impl Party,
    ai: &mut AlignedVec<BsBool8, 64>,
    aii: &AlignedVec<BsBool8, 64>,
    bi: &[BsBool8],
    bii: &[BsBool8],
    orig_bi: &[BsBool8],
    orig_bii: &[BsBool8],
    shifts: &[usize],
    recorder: &mut impl VerificationRecorder,
) -> MpcResult<()>
where
    [(); BYTES_PER_LOOKUP * 8]:,
    [(); BYTES_PER_LOOKUP * 64]:,
{
    use std::arch::x86_64::{
        __m512i, _cvtu64_mask64, _mm512_load_epi64, _mm512_maskz_loadu_epi8,
        _mm512_permutex2var_epi8, _mm512_set_epi8, _mm512_shuffle_i64x2, _mm512_store_epi64,
        _mm512_ternarylogic_epi64, _mm512_xor_epi64,
    };

    use crate::util::aligned_vec::AlignedAllocator;

    debug_assert_eq!(BYTES_PER_LOOKUP, 1 << DIM_BITS);

    let num_lookups = ai.len() / BYTES_PER_LOOKUP;
    let mut result = Vec::new_in(AlignedAllocator::<64>);
    result.reserve(num_lookups);
    result.extend(recorder.generate_alpha(party, num_lookups));

    let (perm_base_0, perm_base_1) = unsafe {
        (
            _mm512_set_epi8(
                126, 118, 110, 102, 94, 86, 78, 70, 62, 54, 46, 38, 30, 22, 14, 6, 124, 116, 108,
                100, 92, 84, 76, 68, 60, 52, 44, 36, 28, 20, 12, 4, 122, 114, 106, 98, 90, 82, 74,
                66, 58, 50, 42, 34, 26, 18, 10, 2, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40,
                32, 24, 16, 8, 0,
            ),
            _mm512_set_epi8(
                127, 119, 111, 103, 95, 87, 79, 71, 63, 55, 47, 39, 31, 23, 15, 7, 125, 117, 109,
                101, 93, 85, 77, 69, 61, 53, 45, 37, 29, 21, 13, 5, 123, 115, 107, 99, 91, 83, 75,
                67, 59, 51, 43, 35, 27, 19, 11, 3, 121, 113, 105, 97, 89, 81, 73, 65, 57, 49, 41,
                33, 25, 17, 9, 1,
            ),
        )
    };

    debug_assert!(ai.len() % (BYTES_PER_LOOKUP * 64) == 0);
    debug_assert_eq!(BYTES_PER_LOOKUP, 8); // Temporary test
    (
        result.as_chunks_mut::<64>().0,
        ai.as_chunks::<{ BYTES_PER_LOOKUP * 64 }>().0,
        aii.as_chunks::<{ BYTES_PER_LOOKUP * 64 }>().0,
        bi.as_chunks::<{ BYTES_PER_LOOKUP * 8 }>().0,
        bii.as_chunks::<{ BYTES_PER_LOOKUP * 8 }>().0,
    )
        .into_par_iter()
        .for_each(|(out, ai, aii, bi, bii)| {
            fn base_compute(
                ai: &[BsBool8; 64],
                aii: &[BsBool8; 64],
                bi: &[BsBool8; 8],
                bii: &[BsBool8; 8],
            ) -> __m512i {
                unsafe {
                    let bi = u64::from_le_bytes(core::mem::transmute_copy(bi));
                    let bii = u64::from_le_bytes(core::mem::transmute_copy(bii));
                    let mask_bi_bii = _cvtu64_mask64(bi ^ bii);
                    let mask_bi = _cvtu64_mask64(bi);

                    let ai =
                        _mm512_maskz_loadu_epi8(mask_bi_bii, ai as *const [BsBool8] as *const i8);
                    let aii =
                        _mm512_maskz_loadu_epi8(mask_bi, aii as *const [BsBool8] as *const i8);
                    _mm512_xor_epi64(ai, aii)
                }
            }
            let base_compute_idx = |index: usize| -> __m512i {
                unsafe {
                    base_compute(
                        sub_array(ai, index),
                        sub_array(aii, index),
                        sub_array(bi, index),
                        sub_array(bii, index),
                    )
                }
            };

            // a and b each contain 8 output_bytes * 8 bytes to be horizontally XORed
            // Returns 4 instances * 16 output_bytes, the 4 output_instances need to be XORed together
            let merge_base = |a, b| unsafe {
                let left = _mm512_permutex2var_epi8(a, perm_base_0, b);
                let right = _mm512_permutex2var_epi8(a, perm_base_1, b);
                _mm512_xor_epi64(left, right)
            };

            // TODO: Check if it's worth merging these, or if it's better to extract and sum up

            // Returns 4 instances * 16 bytes, such that first two should be added up to form the first
            // 16 output bytes, and the second two to form the second 16 output bytes
            let merge_16 = |a, b| unsafe {
                let left = _mm512_shuffle_i64x2::<0x88>(a, b);
                let right = _mm512_shuffle_i64x2::<0xdd>(a, b);
                _mm512_xor_epi64(left, right)
            };

            // Returns 1 instance * 64 output bytes
            let merge_32 = |a, b, alpha| unsafe {
                let left = _mm512_shuffle_i64x2::<0x88>(a, b);
                let right = _mm512_shuffle_i64x2::<0xdd>(a, b);
                _mm512_ternarylogic_epi64::<0x96>(left, right, alpha)
            };

            let produce_4 = |offset: usize| {
                let result0 = base_compute_idx(offset + 0);
                let result1 = base_compute_idx(offset + 1);
                let result01 = merge_base(result0, result1);

                let result2 = base_compute_idx(offset + 2);
                let result3 = base_compute_idx(offset + 3);
                let result23 = merge_base(result2, result3);

                merge_16(result01, result23)
            };

            let result_first4 = produce_4(0);
            let result_last4 = produce_4(4);

            let alpha = unsafe { _mm512_load_epi64(out as *const [BsBool8] as *const i64) };
            let result = merge_32(result_first4, result_last4, alpha);
            unsafe {
                _mm512_store_epi64(out as *mut [BsBool8] as *mut i64, result);
            }
        });

    recorder.record_ip_triple(
        &ai,
        &aii,
        orig_bi,
        orig_bii,
        &result,
        if IS_PRESHIFTED { &[] } else { shifts },
        8,
        8 * 8,
    );

    ai.truncate(result.len());
    ai.copy_from_slice(&result);
    Ok(())
}

pub fn inner_product_large_ss<
    const DIM_BITS: usize,
    const BYTES_PER_BLOCK: usize,
    const BYTES_PER_LOOKUP: usize,
    const IS_PRESHIFTED: bool,
>(
    party: &mut impl Party,
    ai: &mut AlignedVec<BsBool8, 64>,
    aii: &AlignedVec<BsBool8, 64>,
    bi: &[BsBool8],
    bii: &[BsBool8],
    shifts: &[usize],
    recorder: &mut impl VerificationRecorder,
) -> MpcResult<()> {
    // let bytes_per_lookup = ai.len() / num_lookups;
    // debug_assert!(bytes_per_lookup % (1 << dim_bits) == 0);
    debug_assert_eq!(BYTES_PER_LOOKUP, BYTES_PER_BLOCK << DIM_BITS);
    let num_lookups = ai.len() / BYTES_PER_LOOKUP;
    let ohv_block_size = (num_lookups + 7) / 8;
    let mut result = Vec::new_in(AlignedAllocator::<64>);
    result.reserve(BYTES_PER_BLOCK * num_lookups);
    result.extend(recorder.generate_alpha(party, BYTES_PER_BLOCK * num_lookups));

    // TODO: Check if parallel is better?
    (
        result.as_chunks_mut::<BYTES_PER_BLOCK>().0,
        ai.as_chunks::<BYTES_PER_LOOKUP>().0,
        aii.as_chunks::<BYTES_PER_LOOKUP>().0,
        shifts.par_iter(),
    )
        .into_par_iter()
        .enumerate()
        .for_each(|(i, (result, ai, aii, shift))| {
            let byte_index = i / 8;
            let mask = 1 << (i % 8);
            ai.as_chunks::<BYTES_PER_BLOCK>()
                .0
                .iter()
                .zip(aii.as_chunks::<BYTES_PER_BLOCK>().0.iter())
                .enumerate()
                .for_each(|(mut j, (ai, aii))| {
                    if !IS_PRESHIFTED {
                        j ^= shift;
                    }
                    let bi = (bi[j * ohv_block_size + byte_index].0 & mask) != 0;
                    let bii = (bii[j * ohv_block_size + byte_index].0 & mask) != 0;
                    match (bi, bii) {
                        (true, true) => {
                            xor_block(&mut *result, aii);
                        }
                        (true, false) => {
                            xor_blocks(&mut *result, ai, aii);
                        }
                        (false, true) => {
                            xor_block(&mut *result, ai);
                        }
                        (false, false) => {
                            // do nothing
                        }
                    };
                });
        });

    recorder.record_ip_triple(
        &ai,
        &aii,
        &bi,
        &bii,
        &result,
        if IS_PRESHIFTED { &[] } else { shifts },
        BYTES_PER_BLOCK * 8,
        BYTES_PER_LOOKUP * 8,
    );
    ai.truncate(result.len());
    ai.copy_from_slice(&result);
    Ok(())
}

fn sub_byte_accum(byte: BsBool8, output_bits: usize) -> BsBool8 {
    match output_bits {
        4 => BsBool8::new((byte.0 >> 4) ^ (byte.0 & 0xF)),
        2 => BsBool8::new(
            (byte.0 >> 6)
                ^ ((byte.0 & 0b00110000) >> 4)
                ^ ((byte.0 & 0b00001100) >> 2)
                ^ (byte.0 & 0b11),
        ),
        1 => BsBool8::new((byte.0.count_ones() % 2) as u8),
        _ => unreachable!(),
    }
}

fn inner_product_small_ss(
    party: &mut impl Party,
    ai: &mut AlignedVec<BsBool8, 64>,
    aii: &AlignedVec<BsBool8, 64>,
    bi: &[BsBool8],
    bii: &[BsBool8],
    dim_bits: usize,
    num_lookups: usize,
) -> MpcResult<()> {
    let bytes_per_lookup = ai.len() / num_lookups;
    let bits_per_block = bytes_per_lookup * 8 / (1 << dim_bits);

    let bi = bitwise_repeat(bi, bits_per_block);
    let bii = bitwise_repeat(bii, bits_per_block);
    let ohv_bytes_per_lookup = ((1 << dim_bits) + 7) / 8 * bits_per_block;

    debug_assert!(ai.len() <= bi.len());
    let bytes_per_block = if bits_per_block > 8 {
        bits_per_block / 8
    } else {
        1
    };
    let mut result = party
        .generate_alpha::<BsBool8>(bytes_per_block * num_lookups)
        .collect::<Vec<_>>();
    if bits_per_block <= 8 {
        (
            &mut result,
            ai.par_chunks_exact(bytes_per_lookup),
            aii.par_chunks_exact(bytes_per_lookup),
            bi.par_chunks_exact(ohv_bytes_per_lookup),
            bii.par_chunks_exact(ohv_bytes_per_lookup),
        )
            .into_par_iter()
            .for_each(|(result, ai, aii, bi, bii)| {
                for (ai, aii, bi, bii) in izip!(ai, aii, bi, bii) {
                    *result += (*ai + *aii) * *bi + *ai * *bii;
                }
            });

        if bits_per_block < 8 {
            result.par_iter_mut().for_each(|x| {
                *x = sub_byte_accum(*x, bits_per_block);
            });
        }
    } else {
        (
            result.par_chunks_exact_mut(bytes_per_block),
            ai.par_chunks_exact(bytes_per_lookup),
            aii.par_chunks_exact(bytes_per_lookup),
            bi.par_chunks_exact(ohv_bytes_per_lookup),
            bii.par_chunks_exact(ohv_bytes_per_lookup),
        )
            .into_par_iter()
            .for_each(|(result, ai, aii, bi, bii)| {
                izip!(
                    ai.chunks_exact(bytes_per_block),
                    aii.chunks_exact(bytes_per_block),
                    bi.chunks_exact(bytes_per_block),
                    bii.chunks_exact(bytes_per_block)
                )
                .for_each(|(ai, aii, bi, bii)| {
                    for i in 0..bytes_per_block {
                        result[i] += (ai[i] + aii[i]) * bi[i] + ai[i] * bii[i];
                    }
                });
            });
    }
    ai.truncate(result.len());
    ai.copy_from_slice(&result);
    Ok(())
}

pub fn ss_to_rss_shares(
    party: &mut impl Party,
    si: &[BsBool8],
    sii: &mut AlignedVec<BsBool8, 64>,
) -> MpcResult<()> {
    sii.truncate(si.len());
    let recv_data = party.receive_field_slice::<BsBool8>(Direction::Next, sii);
    party.send_field::<BsBool8>(Direction::Previous, si.iter(), si.len());
    recv_data.rcv()?;
    Ok(())
}

pub trait LUT256SPTable<Recorder: VerificationRecorder, const MAL: bool> {
    fn num_input_bits() -> usize;
    fn num_output_bits() -> usize;
    fn is_preshifted() -> bool;
    fn get_dim_bits() -> Vec<usize>;
    fn inner_product_first_dim(
        ohv_si: &[BsBool8],
        ohv_sii: &[BsBool8],
        out_si: &mut [BsBool8],
        out_sii: &mut [BsBool8],
        shifts: &[usize],
    );
    fn lut<const OUTPUT_RSS: bool>(
        data: &mut LUT256SP<Recorder, MAL>,
        v_si: &[BsBool8],
        v_sii: &[BsBool8],
        recorder: &mut Recorder,
    ) -> MpcResult<()>;
}

pub fn compute_shift(c: &[BsBool8], start_bit: usize, num_bits: usize) -> usize {
    (start_bit..start_bit + num_bits)
        .map(|i| {
            let byte_index = i / 8;
            let bit_index = i % 8;
            (((c[byte_index].0 >> bit_index) & 1) as usize) << (i - start_bit)
        })
        .sum::<usize>()
}

impl<Recorder: VerificationRecorder, const MAL: bool> LUT256SP<Recorder, MAL> {
    // 32-bits
    const BLOCK_BITS: usize = 32;

    pub fn preprocess<Table: LUT256SPTable<Recorder, MAL>>(
        &mut self,
        num_lookups_per_block: usize,
        num_blocks: usize,
    ) -> MpcResult<()> {
        let input_bits = Table::num_input_bits();

        let dim_bits = Table::get_dim_bits();
        let ranges = self.inner.as_party_mut().split_range_equally(num_blocks);
        let thread_parties = self
            .inner
            .as_party_mut()
            .create_thread_parties(ranges.clone());
        let recorders = self.inner.as_party_mut().run_in_threadpool(|| {
            let mut thread_recorders = (0..thread_parties.len())
                .map(|_| Recorder::new())
                .collect::<Vec<_>>();
            self.prep_ohv = (thread_parties, &ranges, &mut thread_recorders)
                .into_par_iter()
                .flat_map_iter(|(mut party, (start, end), recorder)| {
                    (*start..*end)
                        .map(|_| {
                            let start = Instant::now();
                            let (prep_ohv, prep_r_si, prep_r_sii): (Vec<_>, Vec<_>, Vec<_>) =
                                dim_bits
                                    .iter()
                                    .map(|bit| {
                                        our_offline::generate_rndohv(
                                            &mut party,
                                            *bit,
                                            num_lookups_per_block,
                                            recorder,
                                        )
                                        .unwrap()
                                    })
                                    .multiunzip();
                            println!("Generate: {:?}", start.elapsed());

                            let start = Instant::now();
                            let prep_r_si = prep_r_si.concat().concat();
                            let prep_r_sii = prep_r_sii.concat().concat();
                            let (r_si, r_sii) = reorder_r(
                                &prep_r_si,
                                &prep_r_sii,
                                input_bits,
                                num_lookups_per_block,
                            );
                            println!("Reorder r: {:?}", start.elapsed());
                            RndOhvPrep {
                                ohvs: prep_ohv,
                                prep_r_si: r_si,
                                prep_r_sii: r_sii,
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            Ok(thread_recorders)
        })?;
        if MAL {
            self.mult_verification(recorders)
        } else {
            Ok(())
        }
    }

    fn mult_verification(&mut self, recorders: Vec<Recorder>) -> MpcResult<()> {
        let (ip_triples, mul_triples) = Recorder::finalize(recorders);
        let result = verify_multiplication_triples::<3, Mersenne61>(
            self.inner.as_party_mut(),
            &mut self.context,
            &ip_triples,
            &mul_triples,
            40,
        )?;

        if !result {
            return Err(MpcError::MultCheck);
        }
        self.inner
            .as_party_mut()
            .compare_view(take(&mut self.context))
    }

    // If OUTPUT_RSS is false, the second vector in the return value should be discarded
    pub fn lut<const OUTPUT_RSS: bool, Table: LUT256SPTable<Recorder, MAL>>(
        &mut self,
        v_si: &[BsBool8],
        v_sii: &[BsBool8],
    ) -> MpcResult<()> {
        let mut recorder = Recorder::new();
        Table::lut::<OUTPUT_RSS>(self, v_si, v_sii, &mut recorder)?;
        if MAL {
            self.online_recorders.push(recorder);
        }
        Ok(())
    }

    pub fn finalize_online(&mut self) -> MpcResult<()> {
        if MAL {
            let recorders = take(&mut self.online_recorders);
            self.mult_verification(recorders)
        } else {
            Ok(())
        }
    }

    fn main_party_mut(&mut self) -> &mut MainParty {
        self.inner.as_party_mut()
    }

    fn constant(&self, value: BsBool8) -> RssShare<BsBool8> {
        self.inner.constant(value)
    }
}

impl<F: Field, Recorder: VerificationRecorder, const MAL: bool> ArithmeticBlackBox<F>
    for LUT256SP<Recorder, MAL>
{
    fn pre_processing(&mut self, n_multiplications: usize) -> MpcResult<()> {
        <ChidaParty as ArithmeticBlackBox<F>>::pre_processing(&mut self.inner, n_multiplications)
    }

    fn io(&self) -> &IoLayerOwned {
        <ChidaParty as ArithmeticBlackBox<F>>::io(&self.inner)
    }

    fn constant(&self, value: F) -> RssShare<F> {
        self.inner.constant(value)
    }

    fn generate_random(&mut self, n: usize) -> RssShareVec<F> {
        self.inner.generate_random(n)
    }

    fn generate_alpha(&mut self, n: usize) -> impl Iterator<Item = F> {
        self.inner.generate_alpha(n)
    }

    // all parties input the same number of inputs
    fn input_round(
        &mut self,
        my_input: &[F],
    ) -> MpcResult<(RssShareVec<F>, RssShareVec<F>, RssShareVec<F>)> {
        self.inner.input_round(my_input)
    }

    fn mul(
        &mut self,
        ci: &mut [F],
        cii: &mut [F],
        ai: &[F],
        aii: &[F],
        bi: &[F],
        bii: &[F],
    ) -> MpcResult<()> {
        self.inner.mul(ci, cii, ai, aii, bi, bii)
    }

    fn output_round(&mut self, si: &[F], sii: &[F]) -> MpcResult<Vec<F>> {
        self.inner.output_round(si, sii)
    }

    fn finalize(&mut self) -> MpcResult<()> {
        <ChidaParty as ArithmeticBlackBox<F>>::finalize(&mut self.inner)
    }
}

#[cfg(test)]
mod test {
    use super::LUT256SPTable;
    use crate::aes::test::{
        test_aes128_keyschedule_gf8, test_aes128_no_keyschedule_gf8, test_aes256_keyschedule_gf8,
        test_aes256_no_keyschedule_gf8,
    };
    use crate::lut_sp_boolean::lut256_tables::GF8_BYTE_TABLE;
    use crate::lut_sp_boolean::test::LUT256Setup;
    use crate::lut_sp_boolean::{
        LUT256SP, NoVerificationRecording, VerificationRecordVec, VerificationRecorder,
        lut256_tables,
    };
    use crate::rep3_core::{share::RssShare, test::TestSetup};
    use crate::share::bs_bool8::BsBool8;
    use crate::share::test::consistent;
    use itertools::izip;
    use marlut_proc_macros::lut_table_impl_boolean;
    use rand::{Rng, RngCore, thread_rng};
    /// [output_bw, thd_n, snd_n]
    fn generate_random_table(input_bw: usize, output_bw: usize) -> Vec<u64> {
        debug_assert!(output_bw <= 64);
        let input_n = 1 << input_bw;
        let mut raw_table = vec![0u64; input_n];
        let mut rng = rand::thread_rng();
        for i in raw_table.iter_mut() {
            *i = rng.gen_range(0..1 << output_bw);
        }
        raw_table
    }

    #[test]
    fn from_u128_table() {
        for shift in 0..256 {
            print!("[");
            for i in 0..256 {
                let index = i ^ shift;
                print!(
                    "{:#04x}, ",
                    (lut256_tables::GF8_U128_TABLE[index % 16] >> ((index / 16) * 8)) & 0xFF
                );
            }
            println!("], ");
        }
    }

    pub struct TestTable1;
    lut_table_impl_boolean! {
        TestTable1; input: 11, output: 1, table: GF8_BYTE_TABLE.0[0]; dim_bits: 5, 6
    }

    pub struct TestTable2;
    lut_table_impl_boolean! {
        TestTable2; input: 10, output: 2, table: GF8_BYTE_TABLE.0[0]; dim_bits: 5, 5
    }

    pub struct TestTable4;
    lut_table_impl_boolean! {
        TestTable4; input: 9, output: 4, table: GF8_BYTE_TABLE.0[0]; dim_bits: 4, 5
    }

    pub struct TestTable8;
    lut_table_impl_boolean! {
        TestTable8; input: 8, output: 8, table: GF8_BYTE_TABLE.0[0]; dim_bits: 3, 3, 2
    }

    pub struct TestTable16;
    lut_table_impl_boolean! {
        TestTable16; input: 7, output: 16, table: GF8_BYTE_TABLE.0[0]; dim_bits: 3, 4
    }

    fn test_lut_ass_helper<
        Recorder: VerificationRecorder,
        const MAL: bool,
        Table: LUT256SPTable<Recorder, MAL>,
    >() {
        let n_input = 64;
        let n_rounds = 1;
        let mut rng = thread_rng();

        let input_bits = Table::num_input_bits();
        let input_bytes = (input_bits + 7) / 8;
        let output_bits = Table::num_output_bits();
        let output_bytes = (output_bits + 7) / 8;
        let size = input_bytes * n_input * n_rounds;
        let mut p1_i = vec![BsBool8(0); size];
        let mut p1_ii = vec![BsBool8(0); size];
        let mut p2_i = vec![BsBool8(0); size];
        let mut p2_ii = vec![BsBool8(0); size];
        let mut p3_i = vec![BsBool8(0); size];
        let mut p3_ii = vec![BsBool8(0); size];

        let table_size = 1 << input_bits;

        let mut input_vals = vec![];
        for (p1_i, p1_ii, p2_i, p2_ii, p3_i, p3_ii) in izip!(
            p1_i.chunks_exact_mut(input_bytes),
            p1_ii.chunks_exact_mut(input_bytes),
            p2_i.chunks_exact_mut(input_bytes),
            p2_ii.chunks_exact_mut(input_bytes),
            p3_i.chunks_exact_mut(input_bytes),
            p3_ii.chunks_exact_mut(input_bytes),
        ) {
            let input_val = rng.next_u32() % table_size;
            input_vals.push(input_val);
            let mut val = input_val;
            for i in 0..input_bytes {
                p1_i[i] = BsBool8::new((rng.next_u32() & 0xFF) as u8);
                p1_ii[i] = BsBool8::new((rng.next_u32() & 0xFF) as u8);
                p2_i[i] = p1_ii[i];
                p2_ii[i] = BsBool8::new((val & 0xFF) as u8 ^ p1_i[i].0 ^ p1_ii[i].0);
                p3_i[i] = p2_ii[i];
                p3_ii[i] = p1_i[i];
                val >>= 8;
            }
        }

        let program = |n_input: usize,
                       n_rounds: usize,
                       input_i: Vec<BsBool8>,
                       input_ii: Vec<BsBool8>| {
            move |p: &mut LUT256SP<Recorder, MAL>| {
                p.preprocess::<Table>(n_input, n_rounds).unwrap();

                let mut out_i = Vec::with_capacity(input_i.len());
                let mut out_ii = Vec::with_capacity(input_ii.len());
                for i in 0..n_rounds {
                    p
                        .lut::<true, Table>(
                            &input_i[i * n_input * input_bytes..(i + 1) * n_input * input_bytes],
                            &input_ii[i * n_input * input_bytes..(i + 1) * n_input * input_bytes],
                        )
                        .unwrap();
                    let (result_i, result_ii) = p.temp_vecs.as_ref().unwrap();
                    assert_eq!(result_i.len(), n_input * output_bytes);
                    out_i.extend(result_i.iter().cloned());
                    out_ii.extend(result_ii.iter().cloned());
                }

                p.finalize_online().unwrap();
                (out_i, out_ii)
            }
        };
        let (((s1_i, s1_ii), _), ((s2_i, s2_ii), _), ((s3_i, s3_ii), _)) =
            LUT256Setup::localhost_setup_multithreads(
                5,
                program(n_input, n_rounds, p1_i, p1_ii),
                program(n_input, n_rounds, p2_i, p2_ii),
                program(n_input, n_rounds, p3_i, p3_ii),
            );

        for (s1_i, s1_ii, s2_i, s2_ii, s3_i, s3_ii) in
            izip!(&s1_i, &s1_ii, &s2_i, &s2_ii, &s3_i, &s3_ii)
        {
            consistent(
                &RssShare::from(*s1_i, *s1_ii),
                &RssShare::from(*s2_i, *s2_ii),
                &RssShare::from(*s3_i, *s3_ii),
            );
        }

        for (input_val, s1, s2, s3) in izip!(
            input_vals,
            s1_ii.chunks_exact(output_bytes),
            s2_ii.chunks_exact(output_bytes),
            s3_ii.chunks_exact(output_bytes)
        ) {
            if output_bits < 8 {
                let bit_start = input_val as usize * output_bits;
                let byte_index = bit_start / 8;
                let bit_index = bit_start % 8;
                let expected = (lut256_tables::GF8_BYTE_TABLE.0[0][byte_index] >> bit_index)
                    & ((1 << output_bits) - 1);
                let res = s1[0] + s2[0] + s3[0];
                assert_eq!(res.0, expected);
            } else {
                let byte_start = input_val as usize * (output_bits / 8);
                let expected = (0..(output_bits / 8))
                    .map(|i| {
                        (lut256_tables::GF8_BYTE_TABLE.0[0][byte_start + i] as usize) << (8 * i)
                    })
                    .sum::<usize>();
                let res = izip!(s1, s2, s3)
                    .enumerate()
                    .map(|(i, (s1, s2, s3))| ((*s1 + *s2 + *s3).0 as usize) << (8 * i))
                    .sum::<usize>();
                assert_eq!(res, expected);
            }
        }
    }

    #[test]
    fn test_lut_ass() {
        for _ in 0..10 {
            test_lut_ass_helper::<NoVerificationRecording, false, TestTable1>();
            test_lut_ass_helper::<NoVerificationRecording, false, TestTable2>();
            test_lut_ass_helper::<NoVerificationRecording, false, TestTable4>();
            test_lut_ass_helper::<NoVerificationRecording, false, TestTable8>();
            test_lut_ass_helper::<NoVerificationRecording, false, TestTable16>();
        }
    }

    #[test]
    fn test_lut_ass_mal() {
        // Smaller than 8 bits is TODO
        // test_lut_ass_helper::<VerificationRecordVec, true, TestTable1>();
        // test_lut_ass_helper::<VerificationRecordVec, true, TestTable2>();
        // test_lut_ass_helper::<VerificationRecordVec, true, TestTable4>();
        test_lut_ass_helper::<VerificationRecordVec, true, TestTable8>();
        test_lut_ass_helper::<VerificationRecordVec, true, TestTable16>();
    }

    #[test]
    fn aes128_keyschedule_lut256() {
        test_aes128_keyschedule_gf8::<LUT256Setup<NoVerificationRecording, false>, _>(Some(5))
    }

    #[test]
    fn aes128_no_keyschedule_lut256() {
        test_aes128_no_keyschedule_gf8::<LUT256Setup<NoVerificationRecording, false>, _>(
            128,
            Some(5),
        )
    }

    #[test]
    fn aes256_keyschedule_lut256() {
        test_aes256_keyschedule_gf8::<LUT256Setup<NoVerificationRecording, false>, _>(Some(5))
    }

    #[test]
    fn aes_256_no_keyschedule_lut256() {
        test_aes256_no_keyschedule_gf8::<LUT256Setup<NoVerificationRecording, false>, _>(1, Some(5))
    }
}
