//! This module provides implementations of various finite [Field]s and replicated secret sharing.
//!
//! The provided field operations are **not constant-time**.
pub mod bs_bool16;
pub mod bs_bool8;
pub mod gf2p64;
pub mod gf4;
mod gf4_bs_table;
pub mod gf8;
mod gf8_tables;
pub mod mersenne61;
pub mod unsigned_ring;
pub mod wol;

use std::iter::Sum;
use std::ops::{Add, AddAssign, BitAnd, BitXor, BitXorAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

use crate::rep3_core::network::NetSerializable;
use crate::rep3_core::party::{DigestExt, RngExt};
use crate::rep3_core::share::{HasZero, RssShare, RssShareGeneral};
use rand::{CryptoRng, Rng};
use std::borrow::Borrow;

// An empty type that is a zero-byte representation for zeros

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Empty;

impl<T> Add<T> for Empty {
    type Output = T;
    fn add(self, rhs: T) -> Self::Output {
        rhs
    }
}

impl<T: Neg> Sub<T> for Empty {
    type Output = <T as Neg>::Output;
    fn sub(self, rhs: T) -> Self::Output {
        -rhs
    }
}

impl Neg for Empty {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self
    }
}

impl<T> Mul<T> for Empty {
    type Output = Self;
    fn mul(self, _rhs: T) -> Self::Output {
        Self
    }
}

impl<T> BitAnd<T> for Empty {
    type Output = Self;
    fn bitand(self, _rhs: T) -> Self::Output {
        Self
    }
}

impl<T> BitXor<T> for Empty {
    type Output = Self;
    fn bitxor(self, _rhs: T) -> Self::Output {
        Self
    }
}

impl<F: FieldLike> AddAssign<F> for Empty {
    fn add_assign(&mut self, _rhs: F) {}
}

impl<F: FieldLike> SubAssign<F> for Empty {
    fn sub_assign(&mut self, _rhs: F) {}
}

impl<F: FieldLike> MulAssign<F> for Empty {
    fn mul_assign(&mut self, _rhs: F) {}
}

impl<F: FieldLike> BitXorAssign<F> for Empty {
    fn bitxor_assign(&mut self, _rhs: F) {}
}

impl Copy for Empty {}

impl HasZero for Empty {
    const ZERO: Self = Self;
}

impl Sum for Empty {
    fn sum<I: Iterator<Item = Self>>(_iter: I) -> Self {
        Self
    }
}

impl NetSerializable for Empty {
    fn serialized_size(_n_elements: usize) -> usize {
        0
    }

    fn as_byte_vec(_it: impl IntoIterator<Item = impl Borrow<Self>>, _len: usize) -> Vec<u8> {
        vec![]
    }

    fn as_byte_vec_slice(_elements: &[Self]) -> Vec<u8> {
        vec![]
    }

    fn from_byte_vec(_v: Vec<u8>, len: usize) -> Vec<Self> {
        vec![Self; len]
    }

    fn from_byte_slice(_v: Vec<u8>, _dest: &mut [Self]) {}
}

impl BasicFieldLike for Empty {}

impl FieldLike for Empty {
    const NBYTES: usize = 0;

    fn as_raw(&self) -> usize {
        0
    }
    fn from_raw(a: usize) -> Self {
        Self
    }
}

impl RngExt for Empty {
    fn fill<R: Rng + CryptoRng>(_rng: &mut R, _buf: &mut [Self]) {}

    fn generate<R: Rng + CryptoRng>(_rng: &mut R, n: usize) -> Vec<Self> {
        vec![Self; n]
    }
}

impl DigestExt for Empty {
    fn update<D: sha2::Digest>(_digest: &mut D, _message: &[Self]) {}
}

impl Rem<u32> for Empty {
    type Output = Empty;

    fn rem(self, _rhs: u32) -> Self::Output {
        Self
    }
}

impl CountOnes for Empty {
    type Output = Empty;
    fn count_ones(&self) -> Empty {
        Empty
    }
}

impl CountOnesParity for Empty {
    type Output = Empty;
    fn count_ones_parity(&self) -> Empty {
        Empty
    }
}

impl<Fp: PrimeField> BitDecompose<Fp> for Empty {
    type Output = Empty;
    fn bit_decompose(&self, len: usize) -> impl Iterator<Item = Empty> {
        (0..len).map(|_| Empty)
    }
}

/// A finite field, or empty
pub trait BasicFieldLike:
    Default
    + HasZero
    + NetSerializable
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Sum<Self>
    + Clone
    + Copy
    + PartialEq
    + AddAssign
    + SubAssign
    + Send
    + Sync
{
}

pub trait FieldLike:
    BasicFieldLike
    + Mul<Output = Self>
    + Add<Empty, Output = Self>
    + Sub<Empty, Output = Self>
    + Mul<Empty, Output = Empty>
    + MulAssign
    + RngExt
{
    // + AsRef<[u8]>
    /// The field size in byte
    const NBYTES: usize;

    // /// Returns the size in byte of a serialization of n_elements many field elements
    // fn serialized_size(n_elements: usize) -> usize;

    /// The field size in bits
    const NBITS: usize = 8 * Self::NBYTES;

    const IS_UR: bool = false;

    fn as_raw(&self) -> usize;
    fn from_raw(a: usize) -> Self;
}

pub trait Field: FieldLike {
    // /// Zero the neutral element of addition
    // const ZERO: Self;

    /// One the neutral element of multiplication
    const ONE: Self;

    /// Returns if the value is zero
    fn is_zero(&self) -> bool;

    // /// Serializes the field elements
    // fn as_byte_vec(it: impl IntoIterator<Item = impl Borrow<Self>>, len: usize) -> Vec<u8>;

    // /// Deserializes field elements from a byte vector
    // fn from_byte_vec(v: Vec<u8>, len: usize) -> Vec<Self>;

    // /// Deserializes field elements from a byte vector into a slice
    // fn from_byte_slice(v: Vec<u8>, dest: &mut [Self]);
}

/// Field that provide a method to compute multiplicative inverses.
pub trait Invertible: FieldLike {
    /// Multiplicative Inverse (zero may map to zero)
    fn inverse(self) -> Self;
}

/// Extension field of `GF(2)` that provides `2` as a constant.
pub trait HasTwo: FieldLike {
    /// The polynomial `X` of degree `1` in the field over `GF(2)`,
    /// i.e.,  `2` if one considers a binary representation of field elements.
    const TWO: Self;
}

/// Field that provides methods to compute inner products.
pub trait InnerProduct: FieldLike {
    /// Computes the dot product of vectors `x` and `y`.
    ///
    /// This function assumes that both vectors are of equal length.
    fn inner_product(a: &[Self], b: &[Self]) -> Self;

    /// Computes the (weak) dot product of replicated sharing vectors `[[x]]` and `[[y]]`.
    ///
    /// The result is a sum sharing of the inner product.
    /// This function assumes that both vectors are of equal length.    
    fn weak_inner_product(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self;

    /// Computes the dot product of vectors x and y given as replicated shares
    /// considering only elements at even positions (0,2,4,6,...).
    /// The result is a sum sharing.
    ///
    /// This function assumes that both vectors are of equal length.    
    fn weak_inner_product2(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self;

    /// Computes the dot product of vectors x' and y' where
    /// `
    /// x'[i] = x[2i] + (x[2i] + x[2i+1])* Self::TWO, and
    /// y'[i] = y[2i] + (y[2i] + y[2i+1])* Self::TWO
    /// `
    /// and x, y are given as replicated shares.
    fn weak_inner_product3(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self;
}

// Try to bit-decompose into the given field. This may return an array of the desired
// type, or may return Empty if the original type is Empty.
pub trait BitDecompose<F> {
    type Output: BasicFieldLike + Mul<F, Output = Self::Output>;

    fn bit_decompose(&self, len: usize) -> impl Iterator<Item = Self::Output>;
}

pub trait CountOnes {
    type Output: Copy
        + Send
        + Sync
        + std::ops::Rem<u32, Output = Self::Output>
        + std::ops::Add<Output = Self::Output>;

    fn count_ones(&self) -> Self::Output;
}

pub trait CountOnesParity {
    type Output: Copy
        + Send
        + Sync;

    fn count_ones_parity(&self) -> Self::Output;
}

pub trait PrimeField: Field + HasTwo + From<u64> + DigestExt + Invertible {}

pub trait LagrangeInterExtrapolate<const DEGREE: usize>
where
    Self: Sized,
{
    fn extrapolate<T1: FieldLike + Mul<Self, Output = T1>, U1: FieldLike + Mul<Self, Output = U1>>(
        evals: &[RssShareGeneral<T1, U1>],
        out: &mut [RssShareGeneral<T1, U1>],
    );
    fn interpolate<T1: FieldLike + Mul<Self, Output = T1>, U1: FieldLike + Mul<Self, Output = U1>>(
        evals: &[RssShareGeneral<T1, U1>],
        eval_point: Self,
    ) -> RssShareGeneral<T1, U1>;
    // This interpolates at degree = 2 * DEGREE
    fn interpolate_target<
        T1: crate::share::FieldLike + std::ops::Mul<Self, Output = T1>,
        U1: crate::share::FieldLike + std::ops::Mul<Self, Output = U1>,
    >(
        evals: &[crate::share::RssShareGeneral<T1, U1>],
        eval_point: Self,
    ) -> crate::share::RssShareGeneral<T1, U1>;
}

#[cfg(any(test, feature = "benchmark-helper"))]
pub mod test {
    use crate::rep3_core::party::RngExt;
    use crate::rep3_core::share::{RssShare, RssShareVec};
    use crate::share::gf4::GF4;
    use crate::share::gf8::GF8;
    use rand::{CryptoRng, Rng, thread_rng};
    use std::borrow::Borrow;
    use std::fmt::Debug;

    use super::Field;

    pub fn consistent<F: Field + Debug>(
        share1: &RssShare<F>,
        share2: &RssShare<F>,
        share3: &RssShare<F>,
    ) {
        assert_eq!(
            share1.sii, share2.si,
            "share1 and share2 are inconsistent: share1={:?}, share2={:?}, share3={:?}",
            share1, share2, share3
        );
        assert_eq!(
            share2.sii, share3.si,
            "share2 and share3 are inconsistent: share1={:?}, share2={:?}, share3={:?}",
            share1, share2, share3
        );
        assert_eq!(
            share3.sii, share1.si,
            "share1 and share3 are inconsistent: share1={:?}, share2={:?}, share3={:?}",
            share1, share2, share3
        );
    }

    pub fn assert_eq<F: Field + Debug>(
        share1: RssShare<F>,
        share2: RssShare<F>,
        share3: RssShare<F>,
        value: F,
    ) {
        let actual = share1.si + share2.si + share3.si;
        assert_eq!(actual, value, "Expected {:?}, got {:?}", value, actual);
    }

    pub fn secret_share<F: Field, R: Rng + CryptoRng>(
        rng: &mut R,
        x: &F,
    ) -> (RssShare<F>, RssShare<F>, RssShare<F>) {
        let r = F::generate(rng, 2);
        let x1 = RssShare::from(x.clone() - r[0] - r[1], r[0]);
        let x2 = RssShare::from(r[0], r[1]);
        let x3 = RssShare::from(r[1], x.clone() - r[0] - r[1]);
        (x1, x2, x3)
    }

    pub fn secret_share_vector<F: Field, R: Rng + CryptoRng>(
        rng: &mut R,
        elements: impl IntoIterator<Item = impl Borrow<F>>,
    ) -> (RssShareVec<F>, RssShareVec<F>, RssShareVec<F>) {
        let (s1, (s2, s3)) = elements
            .into_iter()
            .map(|value| {
                let (s1, s2, s3) = secret_share(rng, value.borrow());
                (s1, (s2, s3))
            })
            .unzip();
        (s1, s2, s3)
    }

    pub fn random_secret_shared_vector<F: Field>(
        n: usize,
    ) -> (Vec<F>, RssShareVec<F>, RssShareVec<F>, RssShareVec<F>) {
        let mut rng = thread_rng();
        let x: Vec<F> = RngExt::generate(&mut rng, n);
        let (s1, s2, s3) = secret_share_vector(&mut rng, x.iter());

        (x, s1, s2, s3)
    }

    #[test]
    fn cmul_gf8() {
        const N: usize = 100;
        let mut rng = thread_rng();
        let x: Vec<GF8> = GF8::generate(&mut rng, N);
        let c: Vec<GF8> = GF8::generate(&mut rng, N);

        for i in 0..N {
            let (x1, x2, x3) = secret_share::<GF8, _>(&mut rng, &x[i]);
            let cx1 = x1 * c[i].clone();
            let cx2 = x2 * c[i].clone();
            let cx3 = x3 * c[i].clone();

            consistent(&cx1, &cx2, &cx3);
            assert_eq(cx1, cx2, cx3, x[i] * c[i]);
        }
    }

    #[test]
    fn cmul_gf4() {
        const N: usize = 100;
        let mut rng = thread_rng();
        let x: Vec<GF4> = GF4::generate(&mut rng, N);
        let c: Vec<GF4> = GF4::generate(&mut rng, N);

        for i in 0..N {
            let (x1, x2, x3) = secret_share(&mut rng, &x[i]);
            let cx1 = x1 * c[i].clone();
            let cx2 = x2 * c[i].clone();
            let cx3 = x3 * c[i].clone();

            consistent(&cx1, &cx2, &cx3);
            assert_eq(cx1, cx2, cx3, x[i] * c[i]);
        }
    }
}
