/// This file contains a lot of building blocks for the LUT implementation.
/// It is so long because it contains a lot of SIMD optimizations, some of them specialized.
/// The real implementation is composed by the `lut_table_impl!` macro in marlut-proc-macros.

use std::{fmt::Debug, mem::take, time::Instant};

use itertools::{Itertools, izip, zip_eq};
use rayon::prelude::*;

use crate::{
    aes::{AesVariant, GF8InvBlackBox},
    lut_sp::{VerificationRecorder, lut256_tables::GF8InvTable, our_offline::LUT256SPOffline},
    lut_sp_malsec::mult_verification::verify_multiplication_triples,
    rep3_core::{
        network::task::{Direction, IoLayerOwned},
        party::{
            DigestExt, MainParty, Party,
            broadcast::Broadcast,
            error::{MpcError, MpcResult},
        },
        share::{Lift, RssShare, RssShareGeneral, RssShareVec},
    },
    share::{Empty, mersenne61::Mersenne61, unsigned_ring::UR8},
    util::{
        aligned_vec::{AlignedAllocator, AlignedVec},
        mul_triple_vec::InnerProductTriple,
    },
};
use crate::{
    // aes::{AesVariant, GF8InvBlackBox},
    chida::ChidaParty,
    share::{Field, gf8::GF8},
    util::ArithmeticBlackBox,
};

use super::{LUT256SP, RndOhvPrep};

impl<Recorder: VerificationRecorder<UR8>, const MAL: bool> GF8InvBlackBox
    for LUT256SP<UR8, Recorder, MAL>
{
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
        let (si, sii): (&mut [UR8], &mut [UR8]) = unsafe {
            (
                &mut *(si as *mut [GF8] as *mut [UR8]),
                &mut *(sii as *mut [GF8] as *mut [UR8]),
            )
        };
        self.lut::<true, GF8InvTable>(si, sii)?;
        let (result_i, result_ii) = self.temp_vecs.as_ref().unwrap();
        si.copy_from_slice(&result_i);
        sii.copy_from_slice(&result_ii);

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

unsafe fn sub_array<T, const N: usize, const SLICE_SIZE: usize>(
    a: &[T; N],
    index: usize,
) -> &[T; SLICE_SIZE] {
    debug_assert!(SLICE_SIZE * index < N);
    let ptr = a as *const [T] as *const T;
    unsafe {
        let array_start = ptr.offset((index * SLICE_SIZE) as isize);
        &*(array_start as *const [T; SLICE_SIZE])
    }
}

fn weak_mult_block<T: Field>(out: &mut [T], ai: &[T], aii: &[T], bi: T, bii: T) {
    for (out, ai, aii) in izip!(out, ai, aii) {
        *out += *ai * (bi + bii) + *aii * bi;
    }
}

pub fn mult_add_block<T: Field>(out: &mut [T], ai: &[T], bi: T) {
    for (out, ai) in zip_eq(out, ai) {
        *out += *ai * bi;
    }
}

// Preshifted version with SIMD optimziations

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
pub fn inner_product_first_dim_preshifted_4<
    T: Field,
    const OUTPUT_BITS: usize,
    const ELEMS_PER_BLOCK: usize,
    const FIRST_DIM_BITS: usize,
>(
    ohv_si: &[T],
    ohv_sii: &[T],
    out_si: &mut [T],
    out_sii: &mut [T],
    table: &[u8],
    num_lookups: usize,
    i: usize,
) {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi8, _mm_store_si128, _mm256_castsi256_si128, _mm256_extracti128_si256,
        _mm256_load_epi64, _mm256_set1_epi16, _mm512_castsi256_si512, _mm512_cvtepi16_epi8,
        _mm512_cvtepu8_epi16, _mm512_mask_set1_epi16, _mm512_setzero_si512,
    };

    unsafe {
        let mut sum_i = _mm512_setzero_si512();
        let mut sum_ii = _mm512_setzero_si512();
        let mask_high = 0xffff0000u32;
        let mut table_ptr = table as *const [u8] as *const u8 as *const i8;
        for j in (0..(1 << FIRST_DIM_BITS)).step_by(2) {
            use std::arch::x86_64::{_mm512_add_epi16, _mm512_mullo_epi16};

            let (bi, bii) = (ohv_si[j * num_lookups + i], ohv_sii[j * num_lookups + i]);
            let (bi_2, bii_2) = (
                ohv_si[(j + 1) * num_lookups + i],
                ohv_sii[(j + 1) * num_lookups + i],
            );
            let bi = _mm512_castsi256_si512(_mm256_set1_epi16(bi.as_raw() as u16 as i16));
            let bi = _mm512_mask_set1_epi16(bi, mask_high, bi_2.as_raw() as u16 as i16);
            let bii = _mm512_castsi256_si512(_mm256_set1_epi16(bii.as_raw() as u16 as i16));
            let bii = _mm512_mask_set1_epi16(bii, mask_high, bii_2.as_raw() as u16 as i16);
            let table = _mm256_load_epi64(table_ptr as *const i64);
            let extended = _mm512_cvtepu8_epi16(table);
            let product_i = _mm512_mullo_epi16(extended, bi);
            let product_ii = _mm512_mullo_epi16(extended, bii);
            sum_i = _mm512_add_epi16(sum_i, product_i);
            sum_ii = _mm512_add_epi16(sum_ii, product_ii);

            table_ptr = table_ptr.add(32);
        }
        let sum_i = _mm512_cvtepi16_epi8(sum_i);
        let sum_ii = _mm512_cvtepi16_epi8(sum_ii);
        let sum_i_high = _mm256_extracti128_si256::<1>(sum_i);
        let sum_ii_high = _mm256_extracti128_si256::<1>(sum_ii);
        let sum_i = _mm_add_epi8(_mm256_castsi256_si128(sum_i), sum_i_high);
        let sum_ii = _mm_add_epi8(_mm256_castsi256_si128(sum_ii), sum_ii_high);
        _mm_store_si128(out_si as *mut [T] as *mut T as *mut __m128i, sum_i);
        _mm_store_si128(out_sii as *mut [T] as *mut T as *mut __m128i, sum_ii);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
fn inner_product_first_dim_ur8_65536_impl<
    T: Field,
    const COL_SHIFT_K: u8,
    const FIRST_DIM_BITS: u8,
>(
    ohv_si: &[T],
    ohv_sii: &[T],
    out_si: &mut [T],
    out_sii: &mut [T],
    table: &[u8],
    num_lookups: usize,
    i: usize,
    shift: usize,
) {
    let col_shift = shift as u8;
    let row = (shift >> 8) as u8;
    unsafe {
        use std::arch::x86_64::{
            __m256i, _mm256_add_epi8, _mm256_loadu_epi8, _mm256_mask_blend_epi8,
            _mm256_setzero_si256, _mm256_store_si256, _mm512_cvtepi16_epi8, _mm512_cvtepu8_epi16,
            _mm512_mullo_epi16, _mm512_set1_epi16,
        };

        let mut out_si_ptr = out_si as *mut [T] as *mut [u8] as *mut u8 as *mut i8;
        let mut out_sii_ptr = out_sii as *mut [T] as *mut [u8] as *mut u8 as *mut i8;

        let rows_per_lookup = 1 << (8 - FIRST_DIM_BITS);
        for row_group in 0..rows_per_lookup {
            let mut row = row.wrapping_add(row_group);
            let mut table_ptr = (table as *const [u8] as *const u8 as *const i8)
                .add(((row as usize) * 256) + col_shift as usize);

            let mut sum_i = [_mm256_setzero_si256(); 8];
            let mut sum_ii = [_mm256_setzero_si256(); 8];

            // XXX: This reads some invalid memory.
            let mut last_row_end = _mm256_loadu_epi8(
                (table as *const [u8] as *const u8 as *const i8)
                    .offset((row as isize) * 256 - 32 + (col_shift % 32) as isize),
            );
            let blend_mask = if col_shift % 32 == 0 {
                0
            } else {
                !((1u32 << (32 - (col_shift % 32))) - 1)
            };
            for j in 0..(1 << FIRST_DIM_BITS) {
                let bi = ohv_si[j * num_lookups + i];
                let bii = ohv_sii[j * num_lookups + i];

                let bi = _mm512_set1_epi16(bi.as_raw() as u16 as i16);
                let bii = _mm512_set1_epi16(bii.as_raw() as u16 as i16);

                for k in 0..COL_SHIFT_K {
                    let table = _mm256_loadu_epi8(table_ptr);
                    let extended = _mm512_cvtepu8_epi16(table);
                    let product_i = _mm512_cvtepi16_epi8(_mm512_mullo_epi16(extended, bi));
                    let product_ii = _mm512_cvtepi16_epi8(_mm512_mullo_epi16(extended, bii));
                    sum_i[k as usize] = _mm256_add_epi8(sum_i[k as usize], product_i);
                    sum_ii[k as usize] = _mm256_add_epi8(sum_ii[k as usize], product_ii);

                    table_ptr = table_ptr.add(32);
                }

                // Row end: need to concatenate with previous one
                let row_end = _mm256_loadu_epi8(table_ptr);
                let table_val = _mm256_mask_blend_epi8(blend_mask, row_end, last_row_end);
                if rows_per_lookup == 1 {
                    // Otherwise there's no need to retain this
                    last_row_end = row_end;
                }
                let extended = _mm512_cvtepu8_epi16(table_val);
                let product_i = _mm512_cvtepi16_epi8(_mm512_mullo_epi16(extended, bi));
                let product_ii = _mm512_cvtepi16_epi8(_mm512_mullo_epi16(extended, bii));
                sum_i[COL_SHIFT_K as usize] =
                    _mm256_add_epi8(sum_i[COL_SHIFT_K as usize], product_i);
                sum_ii[COL_SHIFT_K as usize] =
                    _mm256_add_epi8(sum_ii[COL_SHIFT_K as usize], product_ii);
                table_ptr = table_ptr.offset(32 - 256);

                for k in (COL_SHIFT_K + 1)..8 {
                    let table = _mm256_loadu_epi8(table_ptr);
                    let extended = _mm512_cvtepu8_epi16(table);
                    let product_i = _mm512_cvtepi16_epi8(_mm512_mullo_epi16(extended, bi));
                    let product_ii = _mm512_cvtepi16_epi8(_mm512_mullo_epi16(extended, bii));
                    sum_i[k as usize] = _mm256_add_epi8(sum_i[k as usize], product_i);
                    sum_ii[k as usize] = _mm256_add_epi8(sum_ii[k as usize], product_ii);

                    table_ptr = table_ptr.add(32);
                }

                row = row.wrapping_add(rows_per_lookup);
                table_ptr = table_ptr.add(256 * (rows_per_lookup as usize));
                if rows_per_lookup == 1 {
                    if row == 0 {
                        table_ptr = (table as *const [u8] as *const u8 as *const i8)
                            .add(col_shift as usize);
                        last_row_end = _mm256_loadu_epi8(
                            (table as *const [u8] as *const u8 as *const i8)
                                .offset((col_shift % 32) as isize - 32),
                        );
                    }
                } else {
                    if row < rows_per_lookup {
                        table_ptr = (table as *const [u8] as *const u8 as *const i8)
                            .add(((row as usize) * 256) + col_shift as usize);
                    }
                    last_row_end = _mm256_loadu_epi8(
                        (table as *const [u8] as *const u8 as *const i8)
                            .offset((row as isize) * 256 - 32 + (col_shift % 32) as isize),
                    );
                }
            }

            for k in 0..8 {
                _mm256_store_si256(out_si_ptr as *mut __m256i, sum_i[k]);
                _mm256_store_si256(out_sii_ptr as *mut __m256i, sum_ii[k]);

                out_si_ptr = out_si_ptr.add(32);
                out_sii_ptr = out_sii_ptr.add(32);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
pub fn inner_product_first_dim_ur8_65536<T: Field, const FIRST_DIM_BITS: u8>(
    ohv_si: &[T],
    ohv_sii: &[T],
    out_si: &mut [T],
    out_sii: &mut [T],
    table: &[u8],
    num_lookups: usize,
    i: usize,
    shift: usize,
) {
    match (shift & 0b11100000) >> 5 {
        0 => inner_product_first_dim_ur8_65536_impl::<T, 7, FIRST_DIM_BITS>(
            ohv_si,
            ohv_sii,
            out_si,
            out_sii,
            table,
            num_lookups,
            i,
            shift,
        ),
        1 => inner_product_first_dim_ur8_65536_impl::<T, 6, FIRST_DIM_BITS>(
            ohv_si,
            ohv_sii,
            out_si,
            out_sii,
            table,
            num_lookups,
            i,
            shift,
        ),
        2 => inner_product_first_dim_ur8_65536_impl::<T, 5, FIRST_DIM_BITS>(
            ohv_si,
            ohv_sii,
            out_si,
            out_sii,
            table,
            num_lookups,
            i,
            shift,
        ),
        3 => inner_product_first_dim_ur8_65536_impl::<T, 4, FIRST_DIM_BITS>(
            ohv_si,
            ohv_sii,
            out_si,
            out_sii,
            table,
            num_lookups,
            i,
            shift,
        ),
        4 => inner_product_first_dim_ur8_65536_impl::<T, 3, FIRST_DIM_BITS>(
            ohv_si,
            ohv_sii,
            out_si,
            out_sii,
            table,
            num_lookups,
            i,
            shift,
        ),
        5 => inner_product_first_dim_ur8_65536_impl::<T, 2, FIRST_DIM_BITS>(
            ohv_si,
            ohv_sii,
            out_si,
            out_sii,
            table,
            num_lookups,
            i,
            shift,
        ),
        6 => inner_product_first_dim_ur8_65536_impl::<T, 1, FIRST_DIM_BITS>(
            ohv_si,
            ohv_sii,
            out_si,
            out_sii,
            table,
            num_lookups,
            i,
            shift,
        ),
        7 => inner_product_first_dim_ur8_65536_impl::<T, 0, FIRST_DIM_BITS>(
            ohv_si,
            ohv_sii,
            out_si,
            out_sii,
            table,
            num_lookups,
            i,
            shift,
        ),
        _ => unreachable!(),
    }
}

// Not yet tested

// #[cfg(all(target_arch = "x86_64", target_feature = "avx512f",))]
// pub fn inner_product_first_dim_preshifted_large_opt_8<
//     T: Field,
//     const OUTPUT_BITS: usize,
//     const ELEMS_PER_BLOCK: usize,
// >(
//     out: &mut [T],
//     table: &[u8],
//     bi: T,
//     offset: usize,
// ) {
//     use std::arch::x86_64::{
//         _mm512_cvtepu8_epi16, _mm256_load_epi64, _mm256_store_epi64, _mm512_cvtepi16_epi8,
//         _mm512_mullo_epi16, _mm512_set1_epi16, _mm256_add_epi8
//     };
//     unsafe {
//         let b = _mm512_set1_epi16(bi.as_raw() as u16 as i16);
//         let mut out_ptr = out as *mut [T] as *mut T as *mut i8;
//         let mut table_ptr = table as *const [u8] as *const u8 as *const i8;
//         table_ptr = table_ptr.add(offset);
//         for _ in (0..ELEMS_PER_BLOCK).step_by(32) {
//             let table = _mm256_load_epi64(table_ptr as *const i64);
//             let extended = _mm512_cvtepu8_epi16(table);
//             let product = _mm512_mullo_epi16(extended, b);
//             let low = _mm512_cvtepi16_epi8(product);
//             let out = _mm256_load_epi64(out_ptr as *const i64);
//             let result = _mm256_add_epi8(out, low);
//             _mm256_store_epi64(out_ptr as *mut i64, result);
//             out_ptr = out_ptr.add(32);
//             table_ptr = table_ptr.add(32);
//         }
//     }
// }

pub fn inner_product_first_dim_preshifted<
    T: Field,
    const OUTPUT_BITS: usize,
    const ELEMS_PER_BLOCK: usize,
>(
    out: &mut [T],
    table: &[u8],
    bi: T,
    offset: usize,
) {
    if T::NBITS == 8 {
        // #[cfg(all(target_arch = "x86_64", target_feature = "avx512f",))]
        // {
        //     return inner_product_first_dim_preshifted_large_opt_8::<T, OUTPUT_BITS, ELEMS_PER_BLOCK>(
        //         out, table, bi, offset,
        //     );
        // }
        for i in 0..ELEMS_PER_BLOCK {
            out[i] += T::from_raw(table[offset + i] as usize) * bi;
        }
    } else if T::NBITS == 16 {
        // TODO: Add SIMD optimization here
        for i in 0..ELEMS_PER_BLOCK {
            unsafe {
                out[i] += T::from_raw(
                    std::mem::transmute_copy::<_, u16>(&table[(offset + i) << 1]) as usize,
                ) * bi;
            }
        }
    } else {
        panic!("Unsupported NBITS");
    }
}

// Unoptimized non-preshifted version:

fn to_le_u16(val: usize) -> [u16; 4] {
    [
        (val & 0xffff) as u16,
        ((val >> 16) & 0xffff) as u16,
        ((val >> 32) & 0xffff) as u16,
        ((val >> 48) & 0xffff) as u16,
    ]
}

fn from_le_u16(vals: [u16; 4]) -> usize {
    (vals[0] as usize)
        | ((vals[1] as usize) << 16)
        | ((vals[2] as usize) << 16)
        | ((vals[3] as usize) << 16)
}

fn read_table<T: Field, const OUTPUT_BITS: usize>(table: &[u8], index: usize) -> T {
    if OUTPUT_BITS == 8 {
        T::from_raw(table[index] as usize)
    } else if OUTPUT_BITS < 8 {
        let index = index * OUTPUT_BITS;
        let byte_index = index / 8;
        let bit_index = index % 8;
        let mask = ((1 << OUTPUT_BITS) - 1) << bit_index;
        T::from_raw(((table[byte_index] & mask) >> bit_index) as usize)
    } else if OUTPUT_BITS == 16 {
        T::from_raw((table[index << 1] as usize) | ((table[(index << 1) | 1] as usize) << 8))
    } else {
        panic!("Unsupported OUTPUT_BITS");
    }
}

// This is specialized for the case where the input bits == 16, but the idea is in fact general
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
pub fn inner_product_first_dim_block_opt_16_single_elem<T: Field, const OUTPUT_BITS: usize>(
    out_si: &mut [T],
    out_sii: &mut [T],
    table: &[u8],
    bi: T,
    bii: T,
    shift: usize,
    offset: usize,
) {
    debug_assert!(out_si.len() % 32 == 0);

    unsafe {
        use std::arch::x86_64::{
            _mm512_add_epi16, _mm512_load_epi64, _mm512_loadu_epi16, _mm512_mullo_epi16,
            _mm512_set1_epi16, _mm512_store_epi64,
        };

        let mut out_si_ptr = out_si as *mut [T] as *mut T as *mut i16;
        let mut out_sii_ptr = out_sii as *mut [T] as *mut T as *mut i16;
        let table_ptr = table as *const [u8] as *const u8 as *const i16;

        let mut index = (shift as u16).wrapping_add(offset as u16);
        let bi_raw: u16 = std::mem::transmute_copy(&bi);
        let bii_raw: u16 = std::mem::transmute_copy(&bii);
        let bi = _mm512_set1_epi16(bi_raw as i16);
        let bii = _mm512_set1_epi16(bii_raw as i16);

        for _ in (0..out_si.len()).step_by(32) {
            if index > u16::MAX - 32 {
                for i in 0..32 {
                    let table = (table[(index as usize) << 1] as u16)
                        | ((table[((index as usize) << 1) | 1] as u16) << 8);
                    *out_si_ptr.add(i) =
                        (*out_si_ptr.add(i) as u16).wrapping_add(table.wrapping_mul(bi_raw)) as i16;
                    *out_sii_ptr.add(i) = (*out_sii_ptr.add(i) as u16)
                        .wrapping_add(table.wrapping_mul(bii_raw))
                        as i16;
                    index = index.wrapping_add(1);
                }
            } else {
                let table = _mm512_loadu_epi16(table_ptr.add(index as usize) as *const i16);

                let si = _mm512_load_epi64(out_si_ptr as *const i64);
                let si = _mm512_add_epi16(si, _mm512_mullo_epi16(table, bi));
                let sii = _mm512_load_epi64(out_sii_ptr as *const i64);
                let sii = _mm512_add_epi16(sii, _mm512_mullo_epi16(table, bii));
                _mm512_store_epi64(out_si_ptr as *mut i64, si);
                _mm512_store_epi64(out_sii_ptr as *mut i64, sii);

                index = index.wrapping_add(32);
            }

            out_si_ptr = out_si_ptr.add(32);
            out_sii_ptr = out_sii_ptr.add(32);
        }
    }
}

pub fn inner_product_first_dim_block<T: Field, const OUTPUT_BITS: usize>(
    out: &mut [T],
    table: &[u8],
    bi: T,
    shift: usize,
    offset: usize,
) {
    if T::IS_UR {
        if T::NBITS == 8 {
            if OUTPUT_BITS == 8 && table.len() == 1 << 16 {
                for i in 0..out.len() {
                    let index_low = ((offset + i) as u8).wrapping_add(shift as u8);
                    let index_high = (((offset + i) >> 8) as u8).wrapping_add((shift >> 8) as u8);
                    out[i] += read_table::<T, OUTPUT_BITS>(
                        table,
                        index_low as usize | ((index_high as usize) << 8),
                    ) * bi;
                }
            } else {
                let shift_bytes = shift.to_le_bytes();
                for i in 0..out.len() {
                    let mut bytes = (offset + i).to_le_bytes();
                    for j in 0..bytes.len() {
                        bytes[j] = u8::wrapping_add(shift_bytes[j], bytes[j]);
                    }
                    let index = usize::from_le_bytes(bytes);
                    out[i] += read_table::<T, OUTPUT_BITS>(table, index) * bi;
                }
            }
        } else if T::NBITS == 16 {
            let shift_bytes = to_le_u16(shift);
            for i in 0..out.len() {
                let mut bytes = to_le_u16(offset + i);
                for j in 0..bytes.len() {
                    bytes[j] = u16::wrapping_add(shift_bytes[j], bytes[j]);
                }
                let index = from_le_u16(bytes);
                out[i] += read_table::<T, OUTPUT_BITS>(table, index) * bi;
            }
        } else {
            panic!("Unsupported NBITS");
        }
    } else {
        if T::NBITS == 8 {
            if OUTPUT_BITS == 8 && table.len() == 256 {
                // One byte
                for i in 0..out.len() {
                    let index = (offset + i) ^ shift;
                    out[i] += read_table::<T, OUTPUT_BITS>(table, index) * bi;
                }
            } else {
                let shift_bytes = shift.to_le_bytes();
                for i in 0..out.len() {
                    let mut bytes = (offset + i).to_le_bytes();
                    for j in 0..bytes.len() {
                        bytes[j] ^= shift_bytes[j];
                    }
                    let index = usize::from_le_bytes(bytes);
                    out[i] += read_table::<T, OUTPUT_BITS>(table, index) * bi;
                }
            }
        } else {
            panic!("Unsupported NBITS");
        }
    }
}

pub fn inner_product_large_ss<
    const DIM_BITS: usize,
    const BYTES_PER_BLOCK: usize,
    const BYTES_PER_LOOKUP: usize,
    T: Field,
    Recorder: VerificationRecorder<T>,
>(
    party: &mut impl Party,
    ai: &mut AlignedVec<T, 64>,
    aii: &AlignedVec<T, 64>,
    bi: &[T],
    bii: &[T],
    recorder: &mut Recorder,
) -> MpcResult<()> {
    // let bytes_per_lookup = ai.len() / num_lookups;
    // debug_assert!(bytes_per_lookup % (1 << dim_bits) == 0);
    debug_assert_eq!(BYTES_PER_LOOKUP, BYTES_PER_BLOCK << DIM_BITS);
    let num_lookups = ai.len() / BYTES_PER_LOOKUP;
    let mut result = Vec::new_in(AlignedAllocator::<64>);
    result.reserve(BYTES_PER_BLOCK * num_lookups);
    result.extend(recorder.generate_alpha(party, BYTES_PER_BLOCK * num_lookups));

    // TODO: Check if parallel is better?
    (
        result.as_chunks_mut::<BYTES_PER_BLOCK>().0,
        ai.as_chunks::<BYTES_PER_LOOKUP>().0,
        aii.as_chunks::<BYTES_PER_LOOKUP>().0,
    )
        .into_par_iter()
        .enumerate()
        .for_each(|(i, (result, ai, aii))| {
            ai.as_chunks::<BYTES_PER_BLOCK>()
                .0
                .iter()
                .zip(aii.as_chunks::<BYTES_PER_BLOCK>().0)
                .enumerate()
                .for_each(|(j, (ai, aii))| {
                    let bi = bi[j * num_lookups + i];
                    let bii = bii[j * num_lookups + i];
                    weak_mult_block(&mut *result, ai, aii, bi, bii);
                });
        });
    recorder.record_ip_triple(
        &ai,
        &aii,
        &bi,
        &bii,
        &result,
        BYTES_PER_BLOCK,
        BYTES_PER_LOOKUP,
    );
    ai.truncate(result.len());
    ai.copy_from_slice(&result);
    Ok(())
}

pub fn ss_to_rss_shares<T: Field>(
    party: &mut impl Party,
    si: &[T],
    sii: &mut AlignedVec<T, 64>,
) -> MpcResult<()> {
    sii.truncate(si.len());
    let recv_data = party.receive_field_slice::<T>(Direction::Next, sii);
    party.send_field::<T>(Direction::Previous, si.iter(), si.len());
    recv_data.rcv()?;
    Ok(())
}

pub trait LUT256SPTable<T: Field, Recorder: VerificationRecorder<T>, const MAL: bool>:
    LUT256SPMalTable<T>
{
    fn num_input_bits() -> usize;
    fn num_output_bits() -> usize;
    fn is_preshifted() -> bool;
    fn get_dim_bits() -> Vec<usize>;
    fn inner_product_first_dim(
        ohv_si: &[T],
        ohv_sii: &[T],
        out_si: &mut [T],
        out_sii: &mut [T],
        shifts: &[usize],
    );
    fn lut<const OUTPUT_RSS: bool>(
        data: &mut LUT256SP<T, Recorder, MAL>,
        v_si: &[T],
        v_sii: &[T],
        recorder: &mut Recorder,
    ) -> MpcResult<()>;
}

pub trait LUT256SPMalTable<T: Field> {
    // Dispatches to templated implementations in mult_verification
    fn process_inner_product_triple_self(
        triple: &InnerProductTriple<T>,
        gammas: &[u64],
        coeff: &[u8],
        x1: &mut [RssShareGeneral<Empty, T>],
        x2: &mut [RssShareGeneral<T, Empty>],
    ) -> RssShare<T>;
    fn process_inner_product_triple_next(
        triple: &InnerProductTriple<T>,
        gammas: &[u64],
        coeff: &[u8],
        x2: &mut [RssShareGeneral<Empty, T>],
    ) -> RssShareGeneral<Empty, T>;
    fn process_inner_product_triple_prev(
        triple: &InnerProductTriple<T>,
        gammas: &[u64],
        coeff: &[u8],
        x1: &mut [RssShareGeneral<T, Empty>],
    ) -> RssShareGeneral<T, Empty>;
}

pub fn compute_shift<T: Field>(c: &[T], start_bit: usize, num_bits: usize) -> usize {
    (start_bit..start_bit + num_bits)
        .map(|i| {
            let byte_index = i / T::NBITS;
            let bit_index = i % T::NBITS;
            (((c[byte_index].as_raw() >> bit_index) & 1) as usize) << (i - start_bit)
        })
        .sum::<usize>()
}

impl<
    T: Field + DigestExt + Debug + Lift<Mersenne61, Output = Mersenne61>,
    Recorder: VerificationRecorder<T>,
    const MAL: bool,
> LUT256SP<T, Recorder, MAL>
{
    // 32-bits
    const BLOCK_BITS: usize = 32;

    pub fn preprocess<Table: LUT256SPTable<T, Recorder, MAL>>(
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
        let mut main_recorder = Recorder::new();

        let start = Instant::now();
        let (bits_si, bits_sii) = LUT256SPOffline::random_bits::<MAL>(
            self.inner.as_party_mut(),
            input_bits * num_lookups_per_block * num_blocks,
            &mut main_recorder,
        );
        println!("Random bits: {:?}", start.elapsed());

        self.inner.as_party_mut().wait_for_completion();

        let start = Instant::now();
        let mut recorders = self.inner.as_party_mut().run_in_threadpool(|| {
            let mut thread_recorders = (0..thread_parties.len())
                .map(|_| Recorder::new())
                .collect::<Vec<_>>();
            self.prep_ohv = (thread_parties, &ranges, &mut thread_recorders)
                .into_par_iter()
                .flat_map_iter(|(mut party, (start, end), recorder)| {
                    (*start..*end)
                        .map(|i| {
                            let start = Instant::now();
                            let (mut bits_si, mut bits_sii) = (
                                &bits_si[input_bits * num_lookups_per_block * i
                                    ..input_bits * num_lookups_per_block * (i + 1)],
                                &bits_sii[input_bits * num_lookups_per_block * i
                                    ..input_bits * num_lookups_per_block * (i + 1)],
                            );
                            let (prep_ohv, prep_r_si, prep_r_sii): (
                                Vec<_>,
                                Vec<Vec<AlignedVec<T, 64>>>,
                                Vec<Vec<AlignedVec<T, 64>>>,
                            ) = dim_bits
                                .iter()
                                .map(|bit| {
                                    let (left_si, right_si) =
                                        bits_si.split_at(bit * num_lookups_per_block);
                                    let (left_sii, right_sii) =
                                        bits_sii.split_at(bit * num_lookups_per_block);
                                    bits_si = right_si;
                                    bits_sii = right_sii;
                                    LUT256SPOffline::generate_rndohv(
                                        &mut party,
                                        (left_si, left_sii),
                                        *bit,
                                        num_lookups_per_block,
                                        recorder,
                                    )
                                    .unwrap()
                                })
                                .multiunzip();
                            println!("Generate: {:?}", start.elapsed());

                            let prep_r_si = prep_r_si.concat();
                            let prep_r_sii = prep_r_sii.concat();
                            let prep_r_si = prep_r_si
                                .into_iter()
                                .chunks(T::NBITS)
                                .into_iter()
                                .flat_map(|mut chunk_it| {
                                    let mut multiplier = 2;
                                    let mut acc = chunk_it.next().unwrap();
                                    for vec in chunk_it {
                                        acc.par_iter_mut()
                                            .zip(vec.into_par_iter())
                                            .for_each(|(x, y)| *x += *y * T::from_raw(multiplier));
                                        multiplier *= 2;
                                    }
                                    acc
                                })
                                .collect::<Vec<_>>();
                            let prep_r_sii = prep_r_sii
                                .into_iter()
                                .chunks(T::NBITS)
                                .into_iter()
                                .flat_map(|mut chunk_it| {
                                    let mut multiplier = 2;
                                    let mut acc = chunk_it.next().unwrap();
                                    for vec in chunk_it {
                                        acc.par_iter_mut()
                                            .zip(vec.into_par_iter())
                                            .for_each(|(x, y)| *x += *y * T::from_raw(multiplier));
                                        multiplier *= 2;
                                    }
                                    acc
                                })
                                .collect::<Vec<_>>();

                            RndOhvPrep {
                                ohvs: prep_ohv,
                                prep_r_si,
                                prep_r_sii,
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            Ok(thread_recorders)
        })?;
        println!("Total generate: {:?}", start.elapsed());
        recorders.push(main_recorder);

        if MAL {
            self.mult_verification::<Table>(recorders)
        } else {
            Ok(())
        }
    }

    pub fn mult_verification<Table: LUT256SPTable<T, Recorder, MAL>>(
        &mut self,
        recorders: Vec<Recorder>,
    ) -> MpcResult<()> {
        let (ip_triples, mul_triples, zero_si, zero_sii) = Recorder::finalize(recorders);
        let result = verify_multiplication_triples::<3, T, Mersenne61, Table>(
            self.inner.as_party_mut(),
            &mut self.context,
            &ip_triples,
            &mul_triples,
            40,
        )?;

        if !result {
            return Err(MpcError::MultCheck);
        }
        if !zero_si.is_empty() {
            let zeroes =
                self.inner
                    .as_party_mut()
                    .open_rss(&mut self.context, &zero_si, &zero_sii)?;
            if !zeroes.iter().all(|x| *x == T::ZERO) {
                return Err(MpcError::MultCheck);
            }
        }
        self.inner
            .as_party_mut()
            .compare_view(take(&mut self.context))
    }

    // If OUTPUT_RSS is false, the second vector in the return value should be discarded
    pub fn lut<const OUTPUT_RSS: bool, Table: LUT256SPTable<T, Recorder, MAL>>(
        &mut self,
        v_si: &[T],
        v_sii: &[T],
    ) -> MpcResult<()> {
        let mut recorder = Recorder::new();
        Table::lut::<OUTPUT_RSS>(self, v_si, v_sii, &mut recorder)?;
        if MAL {
            self.online_recorders.push(recorder);
        }
        Ok(())
    }

    pub fn finalize_online<Table: LUT256SPTable<T, Recorder, MAL>>(&mut self) -> MpcResult<()> {
        if MAL {
            let recorders = take(&mut self.online_recorders);
            self.mult_verification::<Table>(recorders)
        } else {
            Ok(())
        }
    }

    pub fn main_party_mut(&mut self) -> &mut MainParty {
        self.inner.as_party_mut()
    }

    fn constant(&self, value: T) -> RssShare<T> {
        self.inner.constant(value)
    }
}

impl<F: Field, T: Field, Recorder: VerificationRecorder<T>, const MAL: bool> ArithmeticBlackBox<F>
    for LUT256SP<T, Recorder, MAL>
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
    use std::fmt::Debug;

    use super::LUT256SPTable;
    use crate::aes::test::{
        test_aes128_keyschedule_gf8, test_aes128_no_keyschedule_gf8, test_aes256_keyschedule_gf8,
        test_aes256_no_keyschedule_gf8,
    };
    use crate::lut_sp::lut256_tables::{FP8AddTable, FP16ExpTable, GF8_BYTE_TABLE, GF8InvTable};
    use crate::lut_sp::test::LUT256Setup;
    use crate::lut_sp::{
        LUT256SP, NoVerificationRecording, VerificationRecordVec, VerificationRecorder,
        lut256_tables,
    };
    use crate::rep3_core::party::DigestExt;
    use crate::rep3_core::share::Lift;
    use crate::rep3_core::{share::RssShare, test::TestSetup};
    use crate::share::gf8::GF8;
    use crate::share::mersenne61::Mersenne61;
    use crate::share::test::consistent;
    use crate::share::unsigned_ring::{UR8, UR16};
    use crate::share::{Field, FieldLike};
    use itertools::izip;
    use marlut_proc_macros::lut_table_impl;
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

    #[test]
    fn get_preshifted_tables() {
        for shift in 0..256 {
            println!("[");
            for i in 0..256 {
                let index = (shift as u8).wrapping_add(i as u8);
                print!("{:#04x}, ", lut256_tables::GF8_BYTE_TABLE[index as usize]);
            }
            println!("], ");
        }
    }

    pub struct TestTable1;
    lut_table_impl! {
        TestTable1; input: 11, output: 1, table: GF8_BYTE_TABLE; dim_bits: 5, 6
    }

    pub struct TestTable2;
    lut_table_impl! {
        TestTable2; input: 10, output: 2, table: GF8_BYTE_TABLE; dim_bits: 5, 5
    }

    pub struct TestTable4;
    lut_table_impl! {
        TestTable4; input: 8, output: 4, table: GF8_BYTE_TABLE; dim_bits: 4, 4
    }

    pub struct TestTable8;
    lut_table_impl! {
        TestTable8; input: 8, output: 8, table: GF8_BYTE_TABLE; dim_bits: 3, 3, 2
    }

    pub struct TestTable16;
    lut_table_impl! {
        TestTable16; input: 7, output: 16, table: GF8_BYTE_TABLE; dim_bits: 3, 4
    }

    fn test_lut_ass_helper<
        T: Field + DigestExt + Debug + Lift<Mersenne61, Output = Mersenne61>,
        Recorder: VerificationRecorder<T>,
        const MAL: bool,
        Table: LUT256SPTable<T, Recorder, MAL>,
    >(
        ref_table: &[u8],
    ) {
        let n_input = 64;
        let n_rounds = 1;
        let mut rng = thread_rng();

        let input_bits = Table::num_input_bits();
        let input_bytes = (input_bits + 7) / 8;
        let input_elems = input_bytes / T::NBYTES;
        let output_bits = Table::num_output_bits();
        let output_bytes = (output_bits + 7) / 8;
        let output_elems = output_bytes / T::NBYTES;

        let size = input_elems * n_input * n_rounds;
        let mut p1_i = vec![T::ZERO; size];
        let mut p1_ii = vec![T::ZERO; size];
        let mut p2_i = vec![T::ZERO; size];
        let mut p2_ii = vec![T::ZERO; size];
        let mut p3_i = vec![T::ZERO; size];
        let mut p3_ii = vec![T::ZERO; size];

        let table_size = 1 << input_bits;

        let mut input_vals = vec![];
        T::fill(&mut rng, &mut p1_i);
        T::fill(&mut rng, &mut p1_ii);
        for (p1_i, p1_ii, p2_i, p2_ii, p3_i, p3_ii) in izip!(
            p1_i.chunks_exact_mut(input_elems),
            p1_ii.chunks_exact_mut(input_elems),
            p2_i.chunks_exact_mut(input_elems),
            p2_ii.chunks_exact_mut(input_elems),
            p3_i.chunks_exact_mut(input_elems),
            p3_ii.chunks_exact_mut(input_elems),
        ) {
            let input_val = rng.next_u32() % table_size;
            input_vals.push(input_val);
            let mut val = input_val;
            for i in 0..input_elems {
                p2_i[i] = p1_ii[i];
                p2_ii[i] = T::from_raw(val as usize) - p1_i[i] - p1_ii[i];
                p3_i[i] = p2_ii[i];
                p3_ii[i] = p1_i[i];
                val >>= T::NBITS;
            }
        }

        let program = |n_input: usize, n_rounds: usize, input_i: Vec<T>, input_ii: Vec<T>| {
            move |p: &mut LUT256SP<T, Recorder, MAL>| {
                p.preprocess::<Table>(n_input, n_rounds).unwrap();

                let mut out_i = Vec::with_capacity(input_i.len());
                let mut out_ii = Vec::with_capacity(input_ii.len());
                for i in 0..n_rounds {
                    p.lut::<true, Table>(
                        &input_i[i * n_input * input_elems..(i + 1) * n_input * input_elems],
                        &input_ii[i * n_input * input_elems..(i + 1) * n_input * input_elems],
                    )
                    .unwrap();

                    let (result_i, result_ii) = p.temp_vecs.as_ref().unwrap();
                    assert_eq!(result_i.len(), n_input * output_elems);
                    out_i.extend(result_i.iter().cloned());
                    out_ii.extend(result_ii.iter().cloned());
                }

                p.finalize_online::<Table>().unwrap();
                (out_i, out_ii)
            }
        };
        let (((s1_i, s1_ii), _), ((s2_i, s2_ii), _), ((s3_i, s3_ii), _)) =
            LUT256Setup::localhost_setup_multithreads(
                8,
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
            s1_ii.chunks_exact(output_elems),
            s2_ii.chunks_exact(output_elems),
            s3_ii.chunks_exact(output_elems)
        ) {
            if output_bits < 8 {
                let bit_start = input_val as usize * output_bits;
                let byte_index = bit_start / 8;
                let bit_index = bit_start % 8;
                let expected = (ref_table[byte_index] >> bit_index) & ((1 << output_bits) - 1);
                let res = s1[0] + s2[0] + s3[0];
                assert_eq!(res.as_raw(), expected as usize);
            } else {
                let byte_start = input_val as usize * (output_bits / 8);
                let expected = (0..(output_bits / 8))
                    .map(|i| (ref_table[byte_start + i] as usize) << (8 * i))
                    .sum::<usize>();
                let res = izip!(s1, s2, s3)
                    .enumerate()
                    .map(|(i, (s1, s2, s3))| ((*s1 + *s2 + *s3).as_raw()) << (8 * i))
                    .sum::<usize>();
                assert_eq!(res, expected);
            }
        }
    }

    #[test]
    fn test_lut_ass() {
        for _ in 0..10 {
            test_lut_ass_helper::<UR8, NoVerificationRecording, false, TestTable4>(
                &lut256_tables::GF8_BYTE_TABLE,
            );
            test_lut_ass_helper::<UR8, NoVerificationRecording, false, TestTable8>(
                &lut256_tables::GF8_BYTE_TABLE,
            );
            test_lut_ass_helper::<UR8, NoVerificationRecording, false, GF8InvTable>(
                &lut256_tables::GF8_BYTE_TABLE,
            );

            test_lut_ass_helper::<GF8, NoVerificationRecording, false, TestTable8>(
                &lut256_tables::GF8_BYTE_TABLE,
            );
        }
    }

    #[test]
    fn test_lut_ass_fp16_exp() {
        for _ in 0..10 {
            test_lut_ass_helper::<UR16, NoVerificationRecording, false, FP16ExpTable>(
                &lut256_tables::FP16_EXP_TABLE,
            );
        }
    }

    #[test]
    fn test_lut_ass_fp8_add() {
        for _ in 0..10 {
            test_lut_ass_helper::<UR8, NoVerificationRecording, false, FP8AddTable>(
                &lut256_tables::FP8_ADD_TABLE,
            );
        }
    }

    #[test]
    fn test_lut_ass_mal() {
        test_lut_ass_helper::<UR8, VerificationRecordVec<UR8>, true, TestTable4>(
            &lut256_tables::GF8_BYTE_TABLE,
        );
        test_lut_ass_helper::<UR8, VerificationRecordVec<UR8>, true, TestTable8>(
            &lut256_tables::GF8_BYTE_TABLE,
        );
    }

    #[test]
    fn aes128_keyschedule_lut256() {
        test_aes128_keyschedule_gf8::<LUT256Setup<UR8, NoVerificationRecording, false>, _>(Some(5))
    }

    #[test]
    fn aes128_no_keyschedule_lut256() {
        test_aes128_no_keyschedule_gf8::<LUT256Setup<UR8, NoVerificationRecording, false>, _>(
            128,
            Some(5),
        )
    }

    #[test]
    fn aes256_keyschedule_lut256() {
        test_aes256_keyschedule_gf8::<LUT256Setup<UR8, NoVerificationRecording, false>, _>(Some(5))
    }

    #[test]
    fn aes_256_no_keyschedule_lut256() {
        test_aes256_no_keyschedule_gf8::<LUT256Setup<UR8, NoVerificationRecording, false>, _>(
            1,
            Some(5),
        )
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
    fn test_inner_product_first_dim_ur8_65536_to_256() {
        use crate::lut_sp::our_online::{
            inner_product_first_dim_block, inner_product_first_dim_ur8_65536,
        };
        use crate::{
            rep3_core::{party::RngExt, share::HasZero},
            util::aligned_vec::AlignedAllocator,
        };
        use std::vec;
        let num_lookups = 20;
        let mut rng = thread_rng();
        let shifts = (0..num_lookups)
            .map(|_| (rng.next_u32() as u16) as usize)
            .collect::<Vec<_>>();
        let ohv_si = UR8::generate(&mut rng, num_lookups * 256);
        let ohv_sii = UR8::generate(&mut rng, num_lookups * 256);
        let mut out_si_ref =
            vec::from_elem_in(UR8::ZERO, num_lookups * 256, AlignedAllocator::<64>);
        let mut out_sii_ref =
            vec::from_elem_in(UR8::ZERO, num_lookups * 256, AlignedAllocator::<64>);
        let mut out_si_test =
            vec::from_elem_in(UR8::ZERO, num_lookups * 256, AlignedAllocator::<64>);
        let mut out_sii_test =
            vec::from_elem_in(UR8::ZERO, num_lookups * 256, AlignedAllocator::<64>);
        izip!(
            out_si_ref.as_chunks_mut::<256>().0,
            out_sii_ref.as_chunks_mut::<256>().0,
            out_si_test.as_chunks_mut::<256>().0,
            out_sii_test.as_chunks_mut::<256>().0,
            shifts.iter()
        )
        .enumerate()
        .for_each(|(i, (out_si, out_sii, out_si_test, out_sii_test, shift))| {
            for j in 0..256 {
                let bi = ohv_si[j * num_lookups + i];
                let bii = ohv_sii[j * num_lookups + i];
                let offset = j * 256;
                inner_product_first_dim_block::<UR8, 8>(
                    &mut *out_si,
                    &lut256_tables::FP8_ADD_TABLE,
                    bi,
                    *shift,
                    offset,
                );
                inner_product_first_dim_block::<UR8, 8>(
                    &mut *out_sii,
                    &lut256_tables::FP8_ADD_TABLE,
                    bii,
                    *shift,
                    offset,
                );
            }
            inner_product_first_dim_ur8_65536::<UR8, 8>(
                &ohv_si,
                &ohv_sii,
                out_si_test,
                out_sii_test,
                &lut256_tables::FP8_ADD_TABLE,
                num_lookups,
                i,
                *shift,
            );
        });
        assert_eq!(out_si_ref, out_si_test);
        assert_eq!(out_sii_ref, out_sii_test);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
    fn test_inner_product_first_dim_ur8_65536_6_bits() {
        use crate::lut_sp::our_online::{
            inner_product_first_dim_block, inner_product_first_dim_ur8_65536,
        };
        use crate::{
            rep3_core::{party::RngExt, share::HasZero},
            util::aligned_vec::AlignedAllocator,
        };
        use std::vec;
        let num_lookups = 50;
        let mut rng = thread_rng();
        let shifts = (0..num_lookups)
            .map(|_| (rng.next_u32() as u16) as usize)
            .collect::<Vec<_>>();
        let ohv_si = UR8::generate(&mut rng, num_lookups * 256);
        let ohv_sii = UR8::generate(&mut rng, num_lookups * 256);
        let mut out_si_ref =
            vec::from_elem_in(UR8::ZERO, num_lookups * (1 << 10), AlignedAllocator::<64>);
        let mut out_sii_ref =
            vec::from_elem_in(UR8::ZERO, num_lookups * (1 << 10), AlignedAllocator::<64>);
        let mut out_si_test =
            vec::from_elem_in(UR8::ZERO, num_lookups * (1 << 10), AlignedAllocator::<64>);
        let mut out_sii_test =
            vec::from_elem_in(UR8::ZERO, num_lookups * (1 << 10), AlignedAllocator::<64>);
        izip!(
            out_si_ref.as_chunks_mut::<1024>().0,
            out_sii_ref.as_chunks_mut::<1024>().0,
            out_si_test.as_chunks_mut::<1024>().0,
            out_sii_test.as_chunks_mut::<1024>().0,
            shifts.iter()
        )
        .enumerate()
        .for_each(|(i, (out_si, out_sii, out_si_test, out_sii_test, shift))| {
            for j in 0..(1 << 6) {
                let bi = ohv_si[j * num_lookups + i];
                let bii = ohv_sii[j * num_lookups + i];
                let offset = j * (1 << 10);
                inner_product_first_dim_block::<UR8, 8>(
                    &mut *out_si,
                    &lut256_tables::FP8_ADD_TABLE,
                    bi,
                    *shift,
                    offset,
                );
                inner_product_first_dim_block::<UR8, 8>(
                    &mut *out_sii,
                    &lut256_tables::FP8_ADD_TABLE,
                    bii,
                    *shift,
                    offset,
                );
            }
            inner_product_first_dim_ur8_65536::<UR8, 6>(
                &ohv_si,
                &ohv_sii,
                out_si_test,
                out_sii_test,
                &lut256_tables::FP8_ADD_TABLE,
                num_lookups,
                i,
                *shift,
            );
        });
        assert_eq!(out_si_ref, out_si_test);
        assert_eq!(out_sii_ref, out_sii_test);
    }
}
