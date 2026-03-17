//! This module implements the 2^{61}-1 Mersenne prime field
use std::{
    borrow::Borrow,
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::share::{PrimeField, RssShare};
use crate::{
    rep3_core::{
        network::NetSerializable,
        party::{DigestExt, RngExt},
        share::HasZero,
    },
    share::BasicFieldLike,
};
use marlut_proc_macros::mersenne61_derive_lagrange_interextrapolation;
use rand::{CryptoRng, Rng};
use sha2::Digest;

use super::{BitDecompose, Empty, Field, FieldLike, HasTwo, InnerProduct, Invertible};

pub const MODULUS: u64 = (1 << 61) - 1;

fn modulo_mersenne61(val: u64) -> u64 {
    let val = (val & MODULUS) + (val >> 61);
    if val >= MODULUS { val - MODULUS } else { val }
}

fn modulo_mersenne61_carry(val: u64, carry: u64) -> u64 {
    let val = (val & MODULUS) + (val >> 61) + (carry << 3);
    if val >= MODULUS { val - MODULUS } else { val }
}

#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Mersenne61(pub u64);

impl Mersenne61 {
    pub fn new(val: u64) -> Self {
        debug_assert!(val < MODULUS);
        Self(val)
    }

    /// Returns a binary representation of the vector.
    pub fn as_u64(&self) -> u64 {
        self.0
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512vbmi2",
        target_feature = "avx512vl"
    ))]
    pub fn mul_multiple_opt_4(c: &mut [Mersenne61], a: &[Mersenne61], b: &[Mersenne61]) {
        use std::arch::x86_64::{
            _mm256_add_epi64, _mm256_and_si256, _mm256_load_epi64, _mm256_mul_epu32,
            _mm256_set1_epi64x, _mm256_shrdi_epi64, _mm256_slli_epi64, _mm256_srli_epi64,
            _mm256_store_epi64,
        };

        debug_assert!(c.len() % 4 == 0);

        unsafe {
            let modulus = _mm256_set1_epi64x(0x1FFFFFFFFFFFFFFF);
            let mut a_ptr = a as *const [Mersenne61] as *const i64;
            let mut b_ptr = b as *const [Mersenne61] as *const i64;
            let mut c_ptr = c as *mut [Mersenne61] as *mut i64;
            for _ in (0..c.len()).step_by(4) {
                let a = _mm256_load_epi64(a_ptr);
                let b = _mm256_load_epi64(b_ptr);

                let a_high = _mm256_srli_epi64::<32>(a);
                let b_high = _mm256_srli_epi64::<32>(b);

                let r0 = _mm256_mul_epu32(a_high, b_high);
                let r1 = _mm256_mul_epu32(a_high, b);
                let r2 = _mm256_mul_epu32(a, b_high);
                let r3 = _mm256_mul_epu32(a, b);

                let r1 = _mm256_add_epi64(r1, r2);

                let rl = _mm256_and_si256(r3, modulus);
                let rh = _mm256_shrdi_epi64::<61>(r3, r0);

                let rl = _mm256_add_epi64(rl, rh);

                let h0 = _mm256_and_si256(_mm256_slli_epi64::<32>(r1), modulus);
                let rl = _mm256_add_epi64(rl, h0);

                let h1 = _mm256_srli_epi64::<29>(r1);
                let rl = _mm256_add_epi64(rl, h1);

                let rh = _mm256_srli_epi64::<61>(rl);
                let rl = _mm256_add_epi64(_mm256_and_si256(rl, modulus), rh);

                let rh = _mm256_srli_epi64::<61>(rl);
                let rl = _mm256_add_epi64(_mm256_and_si256(rl, modulus), rh);
                _mm256_store_epi64(c_ptr, rl);

                a_ptr = a_ptr.offset(4);
                b_ptr = b_ptr.offset(4);
                c_ptr = c_ptr.offset(4);
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512vbmi2",
        target_feature = "avx512vl"
    ))]
    pub fn mul_multiple_opt_8(c: &mut [Mersenne61], a: &[Mersenne61], b: &[Mersenne61]) {
        use std::arch::x86_64::{
            _mm512_add_epi64, _mm512_and_si512, _mm512_load_epi64, _mm512_mul_epu32,
            _mm512_set1_epi64, _mm512_shrdi_epi64, _mm512_slli_epi64, _mm512_srli_epi64,
            _mm512_store_epi64,
        };
        debug_assert!(c.len() % 8 == 0);

        unsafe {
            let modulus = _mm512_set1_epi64(0x1FFFFFFFFFFFFFFF);
            let mut a_ptr = a as *const [Mersenne61] as *const i64;
            let mut b_ptr = b as *const [Mersenne61] as *const i64;
            let mut c_ptr = c as *mut [Mersenne61] as *mut i64;
            for _ in (0..c.len()).step_by(8) {
                let a = _mm512_load_epi64(a_ptr);
                let b = _mm512_load_epi64(b_ptr);

                let a_high = _mm512_srli_epi64::<32>(a);
                let b_high = _mm512_srli_epi64::<32>(b);

                let r0 = _mm512_mul_epu32(a_high, b_high);
                let r1 = _mm512_mul_epu32(a_high, b);
                let r2 = _mm512_mul_epu32(a, b_high);
                let r3 = _mm512_mul_epu32(a, b);

                let r1 = _mm512_add_epi64(r1, r2);

                let rl = _mm512_and_si512(r3, modulus);
                let rh = _mm512_shrdi_epi64::<61>(r3, r0);

                let rl = _mm512_add_epi64(rl, rh);

                let h0 = _mm512_and_si512(_mm512_slli_epi64::<32>(r1), modulus);
                let rl = _mm512_add_epi64(rl, h0);

                let h1 = _mm512_srli_epi64::<29>(r1);
                let rl = _mm512_add_epi64(rl, h1);

                let rh = _mm512_srli_epi64::<61>(rl);
                let rl = _mm512_add_epi64(_mm512_and_si512(rl, modulus), rh);

                _mm512_store_epi64(c_ptr, rl);

                a_ptr = a_ptr.offset(8);
                b_ptr = b_ptr.offset(8);
                c_ptr = c_ptr.offset(8);
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512vbmi2",
        target_feature = "avx512vl"
    ))]
    pub fn mul_assign_multiple_opt_8(c: &mut [Mersenne61], b: &[Mersenne61]) {
        use std::arch::x86_64::{
            _mm512_add_epi64, _mm512_and_si512, _mm512_load_epi64, _mm512_mul_epu32,
            _mm512_set1_epi64, _mm512_shrdi_epi64, _mm512_slli_epi64, _mm512_srli_epi64,
            _mm512_store_epi64,
        };
        debug_assert!(c.len() % 8 == 0);

        unsafe {
            let modulus = _mm512_set1_epi64(0x1FFFFFFFFFFFFFFF);
            let mut b_ptr = b as *const [Mersenne61] as *const i64;
            let mut c_ptr = c as *mut [Mersenne61] as *mut i64;
            for _ in (0..c.len()).step_by(8) {
                let a = _mm512_load_epi64(c_ptr);
                let b = _mm512_load_epi64(b_ptr);

                let a_high = _mm512_srli_epi64::<32>(a);
                let b_high = _mm512_srli_epi64::<32>(b);

                let r0 = _mm512_mul_epu32(a_high, b_high);
                let r1 = _mm512_mul_epu32(a_high, b);
                let r2 = _mm512_mul_epu32(a, b_high);
                let r3 = _mm512_mul_epu32(a, b);

                let r1 = _mm512_add_epi64(r1, r2);

                let rl = _mm512_and_si512(r3, modulus);
                let rh = _mm512_shrdi_epi64::<61>(r3, r0);

                let rl = _mm512_add_epi64(rl, rh);

                let h0 = _mm512_and_si512(_mm512_slli_epi64::<32>(r1), modulus);
                let rl = _mm512_add_epi64(rl, h0);

                let h1 = _mm512_srli_epi64::<29>(r1);
                let rl = _mm512_add_epi64(rl, h1);

                let rh = _mm512_srli_epi64::<61>(rl);
                let rl = _mm512_add_epi64(_mm512_and_si512(rl, modulus), rh);

                _mm512_store_epi64(c_ptr, rl);

                b_ptr = b_ptr.offset(8);
                c_ptr = c_ptr.offset(8);
            }
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512ifma",
        target_feature = "avx512vbmi2",
    ))]
    pub fn mul_multiple_8(c: &mut [Mersenne61], a: &[Mersenne61], b: &[Mersenne61]) {
        use std::arch::x86_64::{
            _mm512_add_epi64, _mm512_and_si512, _mm512_load_epi64, _mm512_madd52hi_epu64,
            _mm512_madd52lo_epu64, _mm512_set1_epi64, _mm512_setzero_epi32, _mm512_shrdi_epi64,
            _mm512_slli_epi64, _mm512_srli_epi64, _mm512_store_epi64,
        };

        debug_assert!(c.len() % 8 == 0);

        unsafe {
            let modulus = _mm512_set1_epi64(0x1FFFFFFFFFFFFFFF);
            let mut a_ptr = a as *const [Mersenne61] as *const i64;
            let mut b_ptr = b as *const [Mersenne61] as *const i64;
            let mut c_ptr = c as *mut [Mersenne61] as *mut i64;
            for _ in (0..c.len()).step_by(8) {
                let a = _mm512_load_epi64(a_ptr);
                let b = _mm512_load_epi64(b_ptr);

                let a_high = _mm512_srli_epi64::<52>(a);
                let b_high = _mm512_srli_epi64::<52>(b);

                let r0 = _mm512_setzero_epi32();
                let r1 = _mm512_setzero_epi32();
                let r2 = _mm512_setzero_epi32();

                let r0 = _mm512_madd52lo_epu64(r0, a, b);
                let r2 = _mm512_madd52lo_epu64(r2, a_high, b_high);
                let r1 = _mm512_madd52hi_epu64(r1, a, b);
                let r2 = _mm512_madd52hi_epu64(r2, a_high, b);
                let r1 = _mm512_madd52lo_epu64(r1, a, b_high);
                let r1 = _mm512_madd52lo_epu64(r1, a_high, b);
                let r2 = _mm512_madd52hi_epu64(r2, a, b_high);

                let r0 = _mm512_slli_epi64::<12>(r0);
                let r0 = _mm512_shrdi_epi64::<12>(r0, r1);
                let r0 = _mm512_and_si512(r0, modulus);
                let r1 = _mm512_srli_epi64::<9>(r1);
                let r2 = _mm512_slli_epi64::<43>(r2);

                let r1 = _mm512_add_epi64(r1, r2);
                let r0 = _mm512_add_epi64(r0, r1);
                let r1 = _mm512_srli_epi64::<61>(r0);
                let r0 = _mm512_and_si512(r0, modulus);
                let r0 = _mm512_add_epi64(r0, r1);
                _mm512_store_epi64(c_ptr, r0);

                a_ptr = a_ptr.offset(8);
                b_ptr = b_ptr.offset(8);
                c_ptr = c_ptr.offset(8);
            }
        }
    }
}

impl BasicFieldLike for Mersenne61 {}

impl FieldLike for Mersenne61 {
    const NBYTES: usize = 8;
    const NBITS: usize = 61;

    fn as_raw(&self) -> usize {
        self.0 as usize
    }

    fn from_raw(a: usize) -> Self {
        Self::new(modulo_mersenne61(a as u64))
    }
}

impl Field for Mersenne61 {
    /// Each component is one
    const ONE: Self = Self(1);

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl Sum for Mersenne61 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl HasZero for Mersenne61 {
    const ZERO: Self = Self(0);
}

impl NetSerializable for Mersenne61 {
    fn serialized_size(n_elements: usize) -> usize {
        n_elements * 8
    }

    fn as_byte_vec(it: impl IntoIterator<Item = impl Borrow<Self>>, _len: usize) -> Vec<u8> {
        it.into_iter()
            .flat_map(|el| el.borrow().0.to_le_bytes())
            .collect()
    }

    fn as_byte_vec_slice(elements: &[Self]) -> Vec<u8> {
        elements.iter().flat_map(|x| x.0.to_le_bytes()).collect()
    }

    fn from_byte_vec(v: Vec<u8>, _len: usize) -> Vec<Self> {
        v.chunks_exact(8)
            .map(|r| Self(u64::from_le_bytes(r.try_into().unwrap())))
            .collect()
    }

    fn from_byte_slice(v: Vec<u8>, dest: &mut [Self]) {
        v.chunks_exact(8)
            .zip(dest)
            .for_each(|(r, dst)| *dst = Self(u64::from_le_bytes(r.try_into().unwrap())))
    }
}

impl Neg for Mersenne61 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        if self.0 == 0 {
            self
        } else {
            Self(MODULUS - self.0)
        }
    }
}

impl Mul for Mersenne61 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let (val, carry) = self.0.widening_mul(rhs.0);
        Self(modulo_mersenne61_carry(val, carry))
    }
}

impl Sub for Mersenne61 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(modulo_mersenne61(self.0 + MODULUS - rhs.0))
    }
}

impl Add for Mersenne61 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(modulo_mersenne61(self.0 + rhs.0))
    }
}

impl AddAssign for Mersenne61 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = modulo_mersenne61(self.0 + rhs.0);
    }
}

impl SubAssign for Mersenne61 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = modulo_mersenne61(self.0 + MODULUS - rhs.0);
    }
}

impl MulAssign for Mersenne61 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Add<Empty> for Mersenne61 {
    type Output = Self;
    fn add(self, _rhs: Empty) -> Self::Output {
        self
    }
}
impl Sub<Empty> for Mersenne61 {
    type Output = Self;
    fn sub(self, _rhs: Empty) -> Self::Output {
        self
    }
}
impl Mul<Empty> for Mersenne61 {
    type Output = Empty;
    fn mul(self, _rhs: Empty) -> Self::Output {
        Empty
    }
}

impl<Fp: PrimeField> BitDecompose<Fp> for Mersenne61 {
    type Output = Fp;
    fn bit_decompose(&self, len: usize) -> impl Iterator<Item = Self::Output> {
        (0..len)
            .into_iter()
            .map(|i| Fp::from((self.0 & (1u64 << i) != 0) as u64))
    }
}

impl RngExt for Mersenne61 {
    fn fill<R: Rng + CryptoRng>(rng: &mut R, buf: &mut [Self]) {
        let mut v = vec![0u8; 8 * buf.len()];
        rng.fill_bytes(&mut v);
        buf.iter_mut()
            .zip(v.chunks_exact(8))
            .for_each(|(x, r)| x.0 = modulo_mersenne61(u64::from_le_bytes(r.try_into().unwrap())))
    }

    fn generate<R: Rng + CryptoRng>(rng: &mut R, n: usize) -> Vec<Self> {
        let mut r = vec![0; 8 * n];
        rng.fill_bytes(&mut r);
        r.chunks_exact(8)
            .map(|r| Self(modulo_mersenne61(u64::from_le_bytes(r.try_into().unwrap()))))
            .collect()
    }
}

impl DigestExt for Mersenne61 {
    fn update<D: Digest>(digest: &mut D, message: &[Self]) {
        for x in message {
            digest.update(x.0.to_le_bytes());
        }
    }
}

impl HasTwo for Mersenne61 {
    const TWO: Self = Self(2);
}

impl From<u64> for Mersenne61 {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

fn square_times<const N: usize>(val: Mersenne61) -> Mersenne61 {
    let mut ret = val;
    for _ in 0..N {
        ret = ret * ret;
    }
    ret
}

impl Invertible for Mersenne61 {
    fn inverse(self) -> Self {
        // Custom implementation
        let x = self;
        let x2 = x * x;
        let x3 = x2 * x; // 2^2 - 1
        let x12 = square_times::<2>(x3);
        let x24 = x12 * x12;
        let x4_0 = x12 * x3; // 2^4 - 1
        let x8_0 = square_times::<4>(x4_0) * x4_0; // 2^8 - 1
        let x16_0 = square_times::<8>(x8_0) * x8_0; // 2^16 - 1
        let x32_0 = square_times::<16>(x16_0) * x16_0; // 2^32 - 1
        let x48_0 = square_times::<16>(x32_0) * x16_0; // 2^48 - 1
        let x56_0 = square_times::<8>(x48_0) * x8_0; // 2^56 - 1
        let x61_0 = square_times::<5>(x56_0); // 2^61 - 2^5
        x61_0 * x24 * x3 * x2
    }
}

impl PrimeField for Mersenne61 {}

impl InnerProduct for Mersenne61 {
    fn inner_product(a: &[Self], b: &[Self]) -> Self {
        a.iter().zip(b).fold(Self::ZERO, |s, (a, b)| s + *a * *b)
    }

    fn weak_inner_product(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        a.iter().zip(b).fold(Self::ZERO, |sum, (x, y)| {
            (x.si + x.sii) * (y.si + y.sii) - x.si * y.si + sum
        })
    }

    fn weak_inner_product2(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        a.iter().zip(b).step_by(2).fold(Self::ZERO, |sum, (x, y)| {
            (x.si + x.sii) * (y.si + y.sii) - x.si * y.si + sum
        })
    }

    fn weak_inner_product3(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        a.chunks(2)
            .zip(b.chunks(2))
            .fold(Self::ZERO, |sum, (x, y)| {
                let x = x[0] + (x[1] - x[0]) * Self::TWO;
                let y = y[0] + (y[1] - y[0]) * Self::TWO;
                (x.si + x.sii) * (y.si + y.sii) - x.si * y.si + sum
            })
    }
}

mersenne61_derive_lagrange_interextrapolation! { 3 } //Degree is 3 ? change to 2

#[cfg(test)]
mod test {
    use crate::{
        rep3_core::party::RngExt,
        share::{Field, HasZero, Invertible},
        util::aligned_vec::{AlignedAllocator, AlignedVec},
    };

    use super::Mersenne61;

    fn get_test_values() -> Vec<Mersenne61> {
        vec![
            Mersenne61::new(0),
            Mersenne61::new(1),
            Mersenne61::new(0xffffffffffffffe),
            Mersenne61::new(0xfffffffffefffff),
        ]
    }

    fn get_non_zero_test_values() -> Vec<Mersenne61> {
        vec![
            Mersenne61::new(1),
            Mersenne61::new(0xffffffffffffffe),
            Mersenne61::new(0xffffffffeffffff),
        ]
    }

    #[test]
    fn test_inv() {
        let test_elements = get_non_zero_test_values();
        for &x in &test_elements {
            let inv = x.inverse();
            assert_eq!(x * inv, Mersenne61::ONE);
            assert_eq!(inv * x, Mersenne61::ONE);
        }
    }

    #[test]
    fn test_mul() {
        let test_elements = get_test_values();
        for &x in &test_elements {
            for &y in &test_elements {
                //println!("{0:?} * {1:?} = {2:?} = {3:?}",x,y,x*y,y*x);
                assert_eq!(x * y, y * x)
            }
        }
    }

    #[test]
    fn test_add_sub() {
        let test_elements = get_test_values();
        for &x in &test_elements {
            for &y in &test_elements {
                assert_eq!(x + y, y + x);
                assert_eq!(x + y - y, x);
                assert_eq!(y + x - x, y);
                assert_eq!(-x + y, y - x);
                assert_eq!(-x - y, -(x + y));
            }
        }
    }

    // Test Lagrange
    use marlut_proc_macros::{
        mersenne61_fixed_lagrange_extrapolation, mersenne61_fixed_lagrange_interpolation,
        mersenne61_lagrange_interpolation,
    };
    use rand::{CryptoRng, Rng, thread_rng};
    mersenne61_fixed_lagrange_interpolation! {4, 5}
    mersenne61_fixed_lagrange_interpolation! {4, 7}
    mersenne61_lagrange_interpolation! {4}

    fn lagrange_poly(num_points: usize, eval_point: Mersenne61, idx: usize) -> Mersenne61 {
        let mut result = Mersenne61::ONE;
        for i in 0..num_points {
            if i == idx {
                continue;
            }
            result *= (eval_point - Mersenne61::from(i as u64))
                * (Mersenne61::from(idx as u64) - Mersenne61::from(i as u64)).inverse();
        }
        result
    }

    fn naive_lagrange(evals: &[Mersenne61], eval_point: Mersenne61) -> Mersenne61 {
        let mut result = Mersenne61::ZERO;
        for (i, eval) in evals.iter().enumerate() {
            result += lagrange_poly(evals.len(), eval_point, i) * *eval;
        }
        result
    }

    mersenne61_fixed_lagrange_extrapolation! {4, 8}

    // #[test]
    // fn test_generated_lagrange_impl() {
    //     let evals = get_test_values();
    //     assert_eq!(
    //         naive_lagrange(&evals, Mersenne61::from(69420)),
    //         mersenne61_lagrange_interpolation_4(&evals, Mersenne61::from(69420))
    //     );
    //     assert_eq!(
    //         naive_lagrange(&evals, Mersenne61::from(5)),
    //         mersenne61_fixed_lagrange_interpolation_4_5(&evals)
    //     );
    //     assert_eq!(
    //         naive_lagrange(&evals, Mersenne61::from(7)),
    //         mersenne61_fixed_lagrange_interpolation_4_7(&evals)
    //     );

    //     let expected_extrapolated_evals = (4..8).map(|i| naive_lagrange(&evals, Mersenne61::from(i)))
    //         .collect::<Vec<_>>();
    //     let mut extrapolated_evals = vec![Mersenne61::ZERO; 4];
    //     mersenne61_fixed_lagrange_extrapolation_4_8(&evals, &mut extrapolated_evals);
    //     assert_eq!(expected_extrapolated_evals, extrapolated_evals);
    // }

    fn random_aligned_vec(
        rng: &mut (impl Rng + CryptoRng),
        len: usize,
    ) -> AlignedVec<Mersenne61, 64> {
        let mut a = Vec::new_in(AlignedAllocator::<64>);
        a.resize(len, Mersenne61::ZERO);
        Mersenne61::fill(rng, &mut a);
        a
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512ifma",
        target_feature = "avx512vbmi2",
        target_feature = "avx512vl"
    ))]
    fn test_mul_multiple() {
        let mut rng = thread_rng();

        let a = random_aligned_vec(&mut rng, 128);
        let b = random_aligned_vec(&mut rng, 128);
        let expected = a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| *a * *b)
            .collect::<Vec<_>>();
        let mut actual = random_aligned_vec(&mut rng, 128);
        Mersenne61::mul_multiple_opt_4(&mut actual, &a, &b);
        assert_eq!(expected, actual);

        let mut actual = random_aligned_vec(&mut rng, 128);
        Mersenne61::mul_multiple_opt_8(&mut actual, &a, &b);
        assert_eq!(expected, actual);

        let mut actual = random_aligned_vec(&mut rng, 128);
        Mersenne61::mul_multiple_8(&mut actual, &a, &b);
        assert_eq!(expected, actual);
    }
}
