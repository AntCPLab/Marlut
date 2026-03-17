use super::{alloc_aligned_blocks, RndOhvOutput};
use crate::lut_sp_boolean::VerificationRecorder;
use crate::rep3_core::network::task::Direction;
use crate::rep3_core::party::{error::MpcResult, Party};
use crate::rep3_core::share::HasZero;
use crate::share::bs_bool8::BsBool8;
use crate::share::Field;
use crate::util::aligned_vec::{AlignedAllocator, AlignedVec};
use itertools::{izip, Itertools};
use rayon::prelude::*;

// This is heavily over-engineered
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bitalg",
    target_feature = "avx512f",
))]
fn reorder_r_opt8(r_si: &[BsBool8], num_bits: usize) -> AlignedVec<BsBool8, 64> {
    debug_assert_eq!(num_bits, 8);
    use std::arch::x86_64::{
        _cvtmask64_u64, _mm256_setr_epi32, _mm512_bitshuffle_epi64_mask, _mm512_i32gather_epi64,
        _mm512_permutexvar_epi8, _mm512_set1_epi64, _mm512_set_epi8, _mm512_setr_epi64,
        _mm512_store_epi64,
    };

    let block_size = r_si.len() / 8;
    debug_assert!(block_size % 8 == 0);
    let mut output = Vec::new_in(AlignedAllocator::<64>);
    output.resize(r_si.len(), BsBool8::ZERO);

    unsafe {
        let offsets = _mm256_setr_epi32(
            0,
            block_size as i32,
            2 * block_size as i32,
            3 * block_size as i32,
            4 * block_size as i32,
            5 * block_size as i32,
            6 * block_size as i32,
            7 * block_size as i32,
        );

        let trans8x8shuf = _mm512_set_epi8(
            63, 55, 47, 39, 31, 23, 15, 7, 62, 54, 46, 38, 30, 22, 14, 6, 61, 53, 45, 37, 29, 21,
            13, 5, 60, 52, 44, 36, 28, 20, 12, 4, 59, 51, 43, 35, 27, 19, 11, 3, 58, 50, 42, 34,
            26, 18, 10, 2, 57, 49, 41, 33, 25, 17, 9, 1, 56, 48, 40, 32, 24, 16, 8, 0,
        );

        let bitshuffle_0 = _mm512_set1_epi64(0x3830282018100800);
        let bitshuffle_1 = _mm512_set1_epi64(0x3931292119110901);
        let bitshuffle_2 = _mm512_set1_epi64(0x3a322a221a120a02);
        let bitshuffle_3 = _mm512_set1_epi64(0x3b332b231b130b03);
        let bitshuffle_4 = _mm512_set1_epi64(0x3c342c241c140c04);
        let bitshuffle_5 = _mm512_set1_epi64(0x3d352d251d150d05);
        let bitshuffle_6 = _mm512_set1_epi64(0x3e362e261e160e06);
        let bitshuffle_7 = _mm512_set1_epi64(0x3f372f271f170f07);

        let mut r_si_ptr = r_si as *const [BsBool8] as *const i64;
        let mut output_ptr = &mut output[..] as *mut [BsBool8] as *mut i64;
        for _ in (0..block_size).step_by(8) {
            let values = _mm512_i32gather_epi64::<1>(offsets, r_si_ptr);
            let shuffled = _mm512_permutexvar_epi8(trans8x8shuf, values);
            let bits0 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_0)) as i64;
            let bits1 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_1)) as i64;
            let bits2 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_2)) as i64;
            let bits3 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_3)) as i64;
            let bits4 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_4)) as i64;
            let bits5 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_5)) as i64;
            let bits6 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_6)) as i64;
            let bits7 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_7)) as i64;
            let bit_shuffled =
                _mm512_setr_epi64(bits0, bits1, bits2, bits3, bits4, bits5, bits6, bits7);
            let result = _mm512_permutexvar_epi8(trans8x8shuf, bit_shuffled);
            _mm512_store_epi64(output_ptr, result);
            r_si_ptr = r_si_ptr.offset(1);
            output_ptr = output_ptr.offset(8);
        }
    }
    output
}

// This is heavily over-engineered
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bitalg",
    target_feature = "avx512f",
))]
fn bit_transpose_16(r_si: &[BsBool8]) -> AlignedVec<BsBool8, 64> {
    use std::arch::x86_64::{
        _cvtmask64_u64, _mm256_setr_epi32, _mm512_bitshuffle_epi64_mask, _mm512_i32gather_epi64,
        _mm512_permutex2var_epi8, _mm512_permutexvar_epi8, _mm512_set1_epi64, _mm512_set_epi8,
        _mm512_setr_epi64, _mm512_store_epi64,
    };

    let block_size = r_si.len() / 16;
    debug_assert!(block_size % 8 == 0);
    let mut output = Vec::new_in(AlignedAllocator::<64>);
    output.resize(r_si.len(), BsBool8::ZERO);

    unsafe {
        let offsets = _mm256_setr_epi32(
            0,
            block_size as i32,
            2 * block_size as i32,
            3 * block_size as i32,
            4 * block_size as i32,
            5 * block_size as i32,
            6 * block_size as i32,
            7 * block_size as i32,
        );

        let trans8x8shuf = _mm512_set_epi8(
            63, 55, 47, 39, 31, 23, 15, 7, 62, 54, 46, 38, 30, 22, 14, 6, 61, 53, 45, 37, 29, 21,
            13, 5, 60, 52, 44, 36, 28, 20, 12, 4, 59, 51, 43, 35, 27, 19, 11, 3, 58, 50, 42, 34,
            26, 18, 10, 2, 57, 49, 41, 33, 25, 17, 9, 1, 56, 48, 40, 32, 24, 16, 8, 0,
        );

        let bitshuffle_0 = _mm512_set1_epi64(0x3830282018100800);
        let bitshuffle_1 = _mm512_set1_epi64(0x3931292119110901);
        let bitshuffle_2 = _mm512_set1_epi64(0x3a322a221a120a02);
        let bitshuffle_3 = _mm512_set1_epi64(0x3b332b231b130b03);
        let bitshuffle_4 = _mm512_set1_epi64(0x3c342c241c140c04);
        let bitshuffle_5 = _mm512_set1_epi64(0x3d352d251d150d05);
        let bitshuffle_6 = _mm512_set1_epi64(0x3e362e261e160e06);
        let bitshuffle_7 = _mm512_set1_epi64(0x3f372f271f170f07);

        let mut top_ptr = r_si as *const [BsBool8] as *const i64;
        let mut bottom_ptr = top_ptr.offset(block_size as isize);
        let mut output_ptr = &mut output[..] as *mut [BsBool8] as *mut i64;
        let process_transposed = |ptr| {
            let values = _mm512_i32gather_epi64::<1>(offsets, ptr);
            let shuffled = _mm512_permutexvar_epi8(trans8x8shuf, values);
            let bits0 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_0)) as i64;
            let bits1 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_1)) as i64;
            let bits2 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_2)) as i64;
            let bits3 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_3)) as i64;
            let bits4 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_4)) as i64;
            let bits5 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_5)) as i64;
            let bits6 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_6)) as i64;
            let bits7 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_7)) as i64;
            let bit_shuffled =
                _mm512_setr_epi64(bits0, bits1, bits2, bits3, bits4, bits5, bits6, bits7);
            bit_shuffled
        };

        let mergebuf1 = _mm512_set_epi8(
            123, 59, 115, 51, 107, 43, 99, 35, 91, 27, 83, 19, 75, 11, 67, 3, 122, 58, 114, 50,
            106, 42, 98, 34, 90, 26, 82, 18, 74, 10, 66, 2, 121, 57, 113, 49, 105, 41, 97, 33, 89,
            25, 81, 17, 73, 9, 65, 1, 120, 56, 112, 48, 104, 40, 96, 32, 88, 24, 80, 16, 72, 8, 64,
            0,
        );
        let mergebuf2 = _mm512_set_epi8(
            127, 63, 119, 55, 111, 47, 103, 39, 95, 31, 87, 23, 79, 15, 71, 7, 126, 62, 118, 54,
            110, 46, 102, 38, 94, 30, 86, 22, 78, 14, 70, 6, 125, 61, 117, 53, 109, 45, 101, 37,
            93, 29, 85, 21, 77, 13, 69, 5, 124, 60, 116, 52, 108, 44, 100, 36, 92, 28, 84, 20, 76,
            12, 68, 4,
        );
        for _ in (0..block_size).step_by(8) {
            let top = process_transposed(top_ptr);
            let bottom = process_transposed(bottom_ptr);
            top_ptr = top_ptr.offset(1);
            bottom_ptr = bottom_ptr.offset(1);

            let left = _mm512_permutex2var_epi8(top, mergebuf1, bottom);
            _mm512_store_epi64(output_ptr, left);
            output_ptr = output_ptr.offset(8);

            let right = _mm512_permutex2var_epi8(top, mergebuf2, bottom);
            _mm512_store_epi64(output_ptr, right);
            output_ptr = output_ptr.offset(8);
        }
    }
    output
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bitalg",
    target_feature = "avx512f",
))]
pub fn slice_and_double_16(r_si: &[BsBool8]) -> AlignedVec<BsBool8, 64> {
    use std::arch::x86_64::{
        _cvtmask64_u64, _mm256_i32gather_epi32, _mm256_setr_epi32, _mm512_bitshuffle_epi64_mask,
        _mm512_castsi256_si512, _mm512_permutex2var_epi8, _mm512_permutexvar_epi8,
        _mm512_set1_epi64, _mm512_set_epi8, _mm512_setr_epi64, _mm512_store_epi64,
    };

    let block_size = r_si.len() / 16;
    debug_assert!(block_size % 4 == 0);
    let mut output = Vec::new_in(AlignedAllocator::<64>);
    output.resize(r_si.len() * 2, BsBool8::ZERO);

    unsafe {
        let offsets = _mm256_setr_epi32(
            0,
            block_size as i32,
            2 * block_size as i32,
            3 * block_size as i32,
            4 * block_size as i32,
            5 * block_size as i32,
            6 * block_size as i32,
            7 * block_size as i32,
        );

        let trans8x8shuf = _mm512_set_epi8(
            31, 31, 27, 27, 23, 23, 19, 19, 15, 15, 11, 11, 7, 7, 3, 3, 30, 30, 26, 26, 22, 22, 18,
            18, 14, 14, 10, 10, 6, 6, 2, 2, 29, 29, 25, 25, 21, 21, 17, 17, 13, 13, 9, 9, 5, 5, 1,
            1, 28, 28, 24, 24, 20, 20, 16, 16, 12, 12, 8, 8, 4, 4, 0, 0,
        );

        let bitshuffle_0 = _mm512_set1_epi64(0x3830282018100800);
        let bitshuffle_1 = _mm512_set1_epi64(0x3931292119110901);
        let bitshuffle_2 = _mm512_set1_epi64(0x3a322a221a120a02);
        let bitshuffle_3 = _mm512_set1_epi64(0x3b332b231b130b03);
        let bitshuffle_4 = _mm512_set1_epi64(0x3c342c241c140c04);
        let bitshuffle_5 = _mm512_set1_epi64(0x3d352d251d150d05);
        let bitshuffle_6 = _mm512_set1_epi64(0x3e362e261e160e06);
        let bitshuffle_7 = _mm512_set1_epi64(0x3f372f271f170f07);

        let mut top_ptr = r_si as *const [BsBool8] as *const i32;
        let mut bottom_ptr = top_ptr.offset((2 * block_size) as isize);
        let mut output_ptr = &mut output[..] as *mut [BsBool8] as *mut i64;
        let process_transposed = |ptr| {
            let values = _mm256_i32gather_epi32::<1>(ptr, offsets);
            let shuffled = _mm512_permutexvar_epi8(trans8x8shuf, _mm512_castsi256_si512(values));
            let bits0 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_0)) as i64;
            let bits1 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_1)) as i64;
            let bits2 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_2)) as i64;
            let bits3 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_3)) as i64;
            let bits4 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_4)) as i64;
            let bits5 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_5)) as i64;
            let bits6 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_6)) as i64;
            let bits7 = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(shuffled, bitshuffle_7)) as i64;
            let bit_shuffled =
                _mm512_setr_epi64(bits0, bits1, bits2, bits3, bits4, bits5, bits6, bits7);
            bit_shuffled
        };

        let mergebuf1 = _mm512_set_epi8(
            123, 122, 59, 58, 115, 114, 51, 50, 107, 106, 43, 42, 99, 98, 35, 34, 91, 90, 27, 26,
            83, 82, 19, 18, 75, 74, 11, 10, 67, 66, 3, 2, 121, 120, 57, 56, 113, 112, 49, 48, 105,
            104, 41, 40, 97, 96, 33, 32, 89, 88, 25, 24, 81, 80, 17, 16, 73, 72, 9, 8, 65, 64, 1,
            0,
        );
        let mergebuf2 = _mm512_set_epi8(
            127, 126, 63, 62, 119, 118, 55, 54, 111, 110, 47, 46, 103, 102, 39, 38, 95, 94, 31, 30,
            87, 86, 23, 22, 79, 78, 15, 14, 71, 70, 7, 6, 125, 124, 61, 60, 117, 116, 53, 52, 109,
            108, 45, 44, 101, 100, 37, 36, 93, 92, 29, 28, 85, 84, 21, 20, 77, 76, 13, 12, 69, 68,
            5, 4,
        );
        for _ in (0..block_size).step_by(4) {
            let top = process_transposed(top_ptr);
            let bottom = process_transposed(bottom_ptr);
            top_ptr = top_ptr.offset(1);
            bottom_ptr = bottom_ptr.offset(1);

            let left = _mm512_permutex2var_epi8(top, mergebuf1, bottom);
            _mm512_store_epi64(output_ptr, left);
            output_ptr = output_ptr.offset(8);

            let right = _mm512_permutex2var_epi8(top, mergebuf2, bottom);
            _mm512_store_epi64(output_ptr, right);
            output_ptr = output_ptr.offset(8);
        }
    }
    output
}

fn reorder_r_small_opt(
    r_si: &[BsBool8],
    r_sii: &[BsBool8],
    num_bits: usize,
    num_lookups: usize,
) -> (AlignedVec<BsBool8, 64>, AlignedVec<BsBool8, 64>) {
    let block_size = r_si.len() / num_bits;

    let (mut si, mut sii) = alloc_aligned_blocks(num_lookups);
    si.par_iter_mut().enumerate().for_each(|(i, out)| {
        let byte_index = i / 8;
        let bit_index = i % 8;
        *out = BsBool8::new(
            (0..num_bits)
                .map(|j| ((r_si[j * block_size + byte_index].0 >> bit_index) & 1) << j)
                .reduce(|x, y| x | y)
                .unwrap(),
        );
    });
    sii.par_iter_mut().enumerate().for_each(|(i, out)| {
        let byte_index = i / 8;
        let bit_index = i % 8;
        *out = BsBool8::new(
            (0..num_bits)
                .map(|j| ((r_sii[j * block_size + byte_index].0 >> bit_index) & 1) << j)
                .reduce(|x, y| x | y)
                .unwrap(),
        );
    });
    (si, sii)
}

pub fn reorder_r(
    r_si: &[BsBool8],
    r_sii: &[BsBool8],
    num_bits: usize,
    num_lookups: usize,
) -> (AlignedVec<BsBool8, 64>, AlignedVec<BsBool8, 64>) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512bitalg",
        target_feature = "avx512f",
    ))]
    if num_bits == 8 && num_lookups % 64 == 0 {
        return (
            reorder_r_opt8(r_si, num_bits),
            reorder_r_opt8(r_sii, num_bits),
        );
    }

    if num_bits <= 8 {
        return reorder_r_small_opt(r_si, r_sii, num_bits, num_lookups);
    }
    let num_bytes = (num_bits + 7) / 8;
    let block_size = r_si.len() / num_bits;

    let (mut si, mut sii) = alloc_aligned_blocks(num_lookups * num_bytes);
    si.par_chunks_exact_mut(num_bytes)
        .enumerate()
        .for_each(|(i, out)| {
            let byte_index = i / 8;
            let bit_index = i % 8;
            (0..num_bits)
                .chunks(8)
                .into_iter()
                .enumerate()
                .for_each(|(k, chunk)| {
                    out[k] = BsBool8::new(
                        chunk
                            .map(|j| {
                                ((r_si[j * block_size + byte_index].0 >> bit_index) & 1) << (j % 8)
                            })
                            .reduce(|x, y| x | y)
                            .unwrap(),
                    );
                });
        });
    sii.par_chunks_exact_mut(num_bytes)
        .enumerate()
        .for_each(|(i, out)| {
            let byte_index = i / 8;
            let bit_index = i % 8;
            (0..num_bits)
                .chunks(8)
                .into_iter()
                .enumerate()
                .for_each(|(k, chunk)| {
                    out[k] = BsBool8::new(
                        chunk
                            .map(|j| {
                                ((r_sii[j * block_size + byte_index].0 >> bit_index) & 1) << (j % 8)
                            })
                            .reduce(|x, y| x | y)
                            .unwrap(),
                    );
                });
        });
    (si, sii)
}

pub fn generate_rndohv<P: Party>(
    party: &mut P,
    dim_bits: usize,
    num_lookups: usize,
    recorder: &mut impl VerificationRecorder,
) -> MpcResult<(RndOhvOutput, Vec<Vec<BsBool8>>, Vec<Vec<BsBool8>>)> {
    let size_per_entry = (num_lookups + 7) / 8;
    let r: (Vec<Vec<BsBool8>>, Vec<Vec<BsBool8>>) = (0..dim_bits)
        .map(|_| party.generate_random_raw(size_per_entry))
        .unzip();

    let size_bytes = (1 << dim_bits) * size_per_entry;
    // let size = (size_bytes + 63) / 64 * 64; // align up to 64 bytes

    let (mut e_si, mut e_sii) = alloc_aligned_blocks(size_bytes);
    generate_ohv_impl(
        party,
        r.0.clone(),
        r.1.clone(),
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
        r.0,
        r.1,
    ))
}

pub fn generate_ohv_impl<P: Party>(
    party: &mut P,
    mut bits_si: Vec<Vec<BsBool8>>,
    mut bits_sii: Vec<Vec<BsBool8>>,
    n: usize,
    out_si: &mut [BsBool8],
    out_sii: &mut [BsBool8],
    recorder: &mut impl VerificationRecorder,
) -> MpcResult<()> {
    let num_lookups = bits_si[0].len();
    if n == 2 {
        debug_assert_eq!(bits_si.len(), 1);
        for i in 0..num_lookups {
            out_si[i] = bits_si[0][i] + BsBool8::ONE;
            out_sii[i] = bits_sii[0][i] + BsBool8::ONE;
            out_si[i + num_lookups] = bits_si[0][i];
            out_sii[i + num_lookups] = bits_sii[0][i];
        }
    } else {
        let msb_si = bits_si.pop().unwrap();
        let msb_sii = bits_sii.pop().unwrap();
        generate_ohv_impl(party, bits_si, bits_sii, n / 2, out_si, out_sii, recorder)?;

        let prev_len = num_lookups * n / 2;
        let (left_si, right_si) = out_si.split_at_mut(prev_len);
        let right_si = &mut right_si[..prev_len];
        let (left_sii, right_sii) = out_sii.split_at_mut(prev_len);
        let right_sii = &mut right_sii[..prev_len];
        let len = prev_len - num_lookups;
        simple_mul(
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
                + right_si[i..]
                    .iter()
                    .step_by(num_lookups)
                    .cloned()
                    .sum::<BsBool8>();
            let e_last_sii = msb_sii[i]
                + right_sii[i..]
                    .iter()
                    .step_by(num_lookups)
                    .cloned()
                    .sum::<BsBool8>();
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

fn mul_no_sync<P: Party>(
    party: &mut P,
    ci: &mut [BsBool8],
    cii: &mut [BsBool8],
    ai: &[BsBool8],
    aii: &[BsBool8],
    bi: &[BsBool8],
    bii: &[BsBool8],
    recorder: &mut impl VerificationRecorder,
) -> MpcResult<()> {
    debug_assert_eq!(ci.len(), ai.len());
    debug_assert_eq!(ci.len(), aii.len());
    debug_assert_eq!(ci.len(), bi.len());
    debug_assert_eq!(ci.len(), bii.len());
    debug_assert_eq!(ci.len(), cii.len());

    let alphas = recorder.generate_alpha(party, ci.len());
    for (i, alpha_i) in alphas.into_iter().enumerate() {
        ci[i] = ai[i] * bi[i] + ai[i] * bii[i] + aii[i] * bi[i] + alpha_i;
    }
    let rcv = party.receive_field_slice(Direction::Next, cii);
    party.send_field_slice(Direction::Previous, ci);
    rcv.rcv()?;
    recorder.record_mul_triple(ai, aii, bi, bii, ci, cii);
    Ok(())
}

fn simple_mul<P: Party>(
    party: &mut P,
    ci: &mut [BsBool8],
    cii: &mut [BsBool8],
    ai: &[BsBool8],
    aii: &[BsBool8],
    msb_si: &[BsBool8],
    msb_sii: &[BsBool8],
    recorder: &mut impl VerificationRecorder,
) -> MpcResult<()> {
    let bi = msb_si
        .iter()
        .cloned()
        .cycle()
        .take(ai.len())
        .collect::<Vec<_>>();
    let bii = msb_sii
        .iter()
        .cloned()
        .cycle()
        .take(ai.len())
        .collect::<Vec<_>>();
    mul_no_sync(party, ci, cii, ai, aii, &bi, &bii, recorder)
}

pub fn slice_and_shift_ohv(
    ohv: &AlignedVec<BsBool8, 64>,
    shift: &[usize],
) -> AlignedVec<BsBool8, 64> {
    let block_size = (shift.len() + 7) / 8;
    let num_blocks = ohv.len() / block_size;
    let out_size_per_lookup = (num_blocks + 7) / 8;
    let mut out = Vec::new_in(AlignedAllocator::<64>);
    out.resize(shift.len() * out_size_per_lookup, BsBool8::ZERO);
    (
        shift.par_iter(),
        out.par_chunks_exact_mut(out_size_per_lookup),
    )
        .into_par_iter()
        .enumerate()
        .for_each(|(i, (shift, out))| {
            let byte_index = i / 8;
            let bit_index = i % 8;
            (0..num_blocks)
                .chunks(8)
                .into_iter()
                .zip_eq(out)
                .for_each(|(chunk, out)| {
                    *out = BsBool8::new(
                        chunk
                            .map(|j| {
                                let new_index = j ^ shift;
                                ((ohv[new_index * block_size + byte_index].0 >> bit_index) & 1)
                                    << (j % 8)
                            })
                            .reduce(|x, y| x | y)
                            .unwrap(),
                    );
                })
        });
    out
}

pub fn slice_ohv(ohv: &AlignedVec<BsBool8, 64>, num_lookups: usize) -> AlignedVec<BsBool8, 64> {
    let block_size = (num_lookups + 7) / 8;
    let num_blocks = ohv.len() / block_size;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512bitalg",
        target_feature = "avx512f",
    ))]
    if num_blocks == 16 && num_lookups % 64 == 0 {
        return bit_transpose_16(&ohv);
    }

    let out_size_per_lookup = (num_blocks + 7) / 8;
    let mut out = Vec::new_in(AlignedAllocator::<64>);
    out.resize(num_lookups * out_size_per_lookup, BsBool8::ZERO);
    out.par_chunks_exact_mut(out_size_per_lookup)
        .enumerate()
        .for_each(|(i, out)| {
            let byte_index = i / 8;
            let bit_index = i % 8;
            (0..num_blocks)
                .chunks(8)
                .into_iter()
                .zip_eq(out)
                .for_each(|(chunk, out)| {
                    *out = BsBool8::new(
                        chunk
                            .map(|j| {
                                ((ohv[j * block_size + byte_index].0 >> bit_index) & 1) << (j % 8)
                            })
                            .reduce(|x, y| x | y)
                            .unwrap(),
                    );
                })
        });
    out
}

#[cfg(test)]
pub mod test {
    use std::fmt::Debug;

    use crate::{
        lut_sp_boolean::{our_offline::generate_ohv_impl, NoVerificationRecording},
        rep3_core::{
            party::RngExt,
            share::{HasZero, RssShare},
            test::TestSetup,
        },
        share::Field,
    };
    use itertools::{izip, MultiUnzip};
    use rand::thread_rng;

    use crate::{
        chida::{online::test::ChidaSetup, ChidaParty},
        share::{bs_bool8::BsBool8, test::secret_share_vector},
    };

    fn reconstruct_and_check_rndohv_bit(
        sii1: &Vec<BsBool8>,
        sii2: &Vec<BsBool8>,
        sii3: &Vec<BsBool8>,
        input: &Vec<Vec<BsBool8>>,
        num_bits: usize,
        num_instances: usize,
    ) {
        let mut expected_index = vec![0; num_instances];
        for i in 0..num_bits {
            for j in 0..num_instances {
                let byte_index = j / 8;
                let bit_index = j % 8;
                expected_index[j] +=
                    (1 << i) * (((input[i][byte_index].0 >> bit_index) & 0x1) as usize);
            }
        }

        let offset = (num_instances + 7) / 8;
        for i in 0..(1 << num_bits) {
            for j in 0..num_instances {
                let byte_index = j / 8;
                let bit_index = j % 8;
                let result = ((sii1[i * offset + byte_index].0 >> bit_index)
                    ^ (sii2[i * offset + byte_index].0 >> bit_index)
                    ^ (sii3[i * offset + byte_index].0 >> bit_index))
                    & 0x1;

                if i == expected_index[j] {
                    assert_eq!(result, 1);
                } else {
                    assert_eq!(result, 0);
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
            .map(|_| BsBool8::generate(&mut rng, (num_instances + 7) / 8))
            .collect::<Vec<_>>();
        let shares: (Vec<_>, Vec<_>, Vec<_>) = input
            .iter()
            .map(|input| secret_share_vector(&mut rng, input.iter()))
            .multiunzip();
        let program = |shares: Vec<Vec<RssShare<BsBool8>>>| {
            move |p: &mut ChidaParty| {
                let size_bytes = (1 << input_bits) * (num_instances + 7) / 8;
                // let size = (size_bytes + 63) / 64 * 64; // align up to 64 bytes

                let (bits_si, bits_sii): (Vec<_>, Vec<_>) = shares
                    .iter()
                    .map(|shares| {
                        let (si, sii): (Vec<_>, Vec<_>) =
                            shares.iter().map(|share| (share.si, share.sii)).unzip();
                        (si, sii)
                    })
                    .unzip();

                let mut e_si = vec![BsBool8::ZERO; size_bytes];
                let mut e_sii = vec![BsBool8::ZERO; size_bytes];
                generate_ohv_impl(
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
        test_generate_ohv_output_helper(1, 10);
        test_generate_ohv_output_helper(2, 10);
        test_generate_ohv_output_helper(3, 10);
        test_generate_ohv_output_helper(16, 10);
        test_generate_ohv_output_helper(13, 10);
    }

    #[test]
    fn generate_trans8x8() {
        for i in (0..8).rev() {
            for j in (0..8).rev() {
                print!("{}, ", j * 8 + i);
            }
        }
        println!("");
    }

    #[test]
    fn generate_trans8x8_2() {
        for index in (0..64).rev() {
            let col = index / 16;
            let row = index % 16;
            if row % 2 == 1 {
                print!("{}, ", 64 + row / 2 * 8 + col);
            } else {
                print!("{}, ", row / 2 * 8 + col);
            }
        }
        println!("");
        for index in (0..64).rev() {
            let col = index / 16 + 4;
            let row = index % 16;
            if row % 2 == 1 {
                print!("{}, ", 64 + row / 2 * 8 + col);
            } else {
                print!("{}, ", row / 2 * 8 + col);
            }
        }
    }

    #[test]
    fn generate_trans8x8_3() {
        for index in (0..64).rev() {
            let row = index / 16;
            let col = index % 16;
            print!("{}, ", (col / 2) * 4 + row);
        }
        println!("");
    }

    #[test]
    fn generate_trans8x8_4() {
        for index in (0..64).rev() {
            let i = index / 2;
            let j = index % 2;
            let col = i / 16;
            let row = i % 16;
            if row % 2 == 1 {
                print!("{}, ", 64 + row / 2 * 8 + col * 2 + j);
            } else {
                print!("{}, ", row / 2 * 8 + col * 2 + j);
            }
        }
        println!("");
        for index in (0..64).rev() {
            let i = index / 2;
            let j = index % 2;
            let col = i / 16 + 2;
            let row = i % 16;
            if row % 2 == 1 {
                print!("{}, ", 64 + row / 2 * 8 + col * 2 + j);
            } else {
                print!("{}, ", row / 2 * 8 + col * 2 + j);
            }
        }
    }

    #[test]
    fn generate_bitshuffle() {
        for i in 0..8 {
            print!("let bitshuffle_{} = _mm512_set1_epi64(0x", i);
            for k in (0..8).rev() {
                print!("{:02x}", k * 8 + i);
            }
            println!(");");
        }
    }

    #[test]
    fn generate_merge_base_8() {
        for index in (0..64).rev() {
            let col = (index / 16) * 2;
            let row = index % 16;
            print!("{}, ", row * 8 + col);
        }
        println!("");

        for index in (0..64).rev() {
            let col = (index / 16) * 2 + 1;
            let row = index % 16;
            print!("{}, ", row * 8 + col);
        }
        println!("");
    }

    #[test]
    fn generate_merge_base_16() {
        for index in (0..64).rev() {
            let col = (index / 8) * 2;
            let row = index % 8;
            print!("{}, ", row * 16 + col);
        }
        println!("");

        for index in (0..64).rev() {
            let col = (index / 8) * 2 + 1;
            let row = index % 8;
            print!("{}, ", row * 16 + col);
        }
        println!("");
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512bitalg",
        target_feature = "avx512f",
    ))]
    #[test]
    fn test_reorder_r_opt8() {
        use super::{reorder_r_opt8, reorder_r_small_opt};

        let num_lookups = 1024;
        let num_bits = 8;

        let mut rng = thread_rng();
        let r_si = BsBool8::generate(&mut rng, num_lookups * num_bits / 8);
        let r_sii = BsBool8::generate(&mut rng, num_lookups * num_bits / 8);
        let (expected_r_si, expected_r_sii) =
            reorder_r_small_opt(&r_si, &r_sii, num_bits, num_lookups);
        let actual_r_si = reorder_r_opt8(&r_si, num_bits);
        let actual_r_sii = reorder_r_opt8(&r_sii, num_bits);
        assert_eq!(expected_r_si, actual_r_si);
        assert_eq!(expected_r_sii, actual_r_sii);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512bitalg",
        target_feature = "avx512f",
    ))]
    #[test]
    fn test_transpose_bit_16() {
        use super::{bit_transpose_16, reorder_r};

        let num_lookups = 1024;
        let num_bits = 16;

        let mut rng = thread_rng();
        let r_si = BsBool8::generate(&mut rng, num_lookups * num_bits / 8);
        let r_sii = BsBool8::generate(&mut rng, num_lookups * num_bits / 8);
        let (expected_r_si, expected_r_sii) = reorder_r(&r_si, &r_sii, num_bits, num_lookups);
        let actual_r_si = bit_transpose_16(&r_si);
        let actual_r_sii = bit_transpose_16(&r_sii);
        assert_eq!(expected_r_si, actual_r_si);
        assert_eq!(expected_r_sii, actual_r_sii);
    }

    fn bit_repeat(val: u8) -> [u8; 2] {
        fn sub_repeat(val: u8) -> u8 {
            match val {
                0b0000 => 0b00000000,
                0b0001 => 0b00000011,
                0b0010 => 0b00001100,
                0b0011 => 0b00001111,
                0b0100 => 0b00110000,
                0b0101 => 0b00110011,
                0b0110 => 0b00111100,
                0b0111 => 0b00111111,
                0b1000 => 0b11000000,
                0b1001 => 0b11000011,
                0b1010 => 0b11001100,
                0b1011 => 0b11001111,
                0b1100 => 0b11110000,
                0b1101 => 0b11110011,
                0b1110 => 0b11111100,
                0b1111 => 0b11111111,
                _ => unreachable!(),
            }
        }
        [
            sub_repeat(val & 0b1111),
            sub_repeat((val & 0b11110000) >> 4),
        ]
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512bitalg",
        target_feature = "avx512f",
    ))]
    #[test]
    fn test_slice_and_double_16() {
        use super::{reorder_r, slice_and_double_16};

        let num_lookups = 2048;
        let num_bits = 16;

        let mut rng = thread_rng();
        let r_si = BsBool8::generate(&mut rng, num_lookups * num_bits / 8);
        let r_sii = BsBool8::generate(&mut rng, num_lookups * num_bits / 8);
        let (expected_r_si, _) = reorder_r(&r_si, &r_sii, num_bits, num_lookups);
        let expected_r_si = expected_r_si
            .iter()
            .flat_map(|val| {
                let [a, b] = bit_repeat(val.0);
                [BsBool8::new(a), BsBool8::new(b)]
            })
            .collect::<Vec<_>>();

        let actual_r_si = slice_and_double_16(&r_si);

        assert_eq!(expected_r_si, actual_r_si);
    }
}
