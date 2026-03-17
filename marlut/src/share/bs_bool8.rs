//! This module implements the 8-bit vector field `GF(2)^8`.
use std::{
    borrow::Borrow,
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{
    rep3_core::{
        network::NetSerializable,
        party::{DigestExt, RngExt},
        share::HasZero,
    },
    share::{BasicFieldLike, CountOnesParity, PrimeField},
};
use rand::{CryptoRng, Rng};
use sha2::Digest;

use super::{BitDecompose, CountOnes, Empty, Field, FieldLike, InnerProduct};

/// An element in `GF(2)^8`, i.e. a vector of 8 booleans.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct BsBool8(pub u8);

impl BsBool8 {
    pub fn new(bits: u8) -> Self {
        Self(bits)
    }

    /// Returns a binary representation of the vector.
    pub fn as_u8(&self) -> u8 {
        self.0
    }
}

impl BasicFieldLike for BsBool8 {}

impl FieldLike for BsBool8 {
    const NBYTES: usize = 1;

    fn as_raw(&self) -> usize {
        self.0 as usize
    }

    fn from_raw(a: usize) -> Self {
        Self(a as u8)
    }
}

impl Field for BsBool8 {
    /// Each component is one
    const ONE: Self = Self(0xff);

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl Sum for BsBool8 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl HasZero for BsBool8 {
    const ZERO: Self = Self(0x00);
}

impl NetSerializable for BsBool8 {
    fn serialized_size(n_elements: usize) -> usize {
        n_elements
    }

    fn as_byte_vec(it: impl IntoIterator<Item = impl Borrow<Self>>, _len: usize) -> Vec<u8> {
        it.into_iter().map(|el| el.borrow().0 as u8).collect()
    }

    fn as_byte_vec_slice(elements: &[Self]) -> Vec<u8> {
        elements.iter().map(|x| x.0).collect()
    }

    fn from_byte_vec(v: Vec<u8>, _len: usize) -> Vec<Self> {
        v.into_iter().map(Self).collect()
    }

    fn from_byte_slice(v: Vec<u8>, dest: &mut [Self]) {
        v.into_iter()
            .zip(dest)
            .for_each(|(byte, dst)| *dst = Self(byte))
    }
}

impl Neg for BsBool8 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(self.0)
    }
}

impl Mul for BsBool8 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl Sub for BsBool8 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl Add for BsBool8 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl AddAssign for BsBool8 {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl SubAssign for BsBool8 {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl MulAssign for BsBool8 {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl Add<Empty> for BsBool8 {
    type Output = Self;
    fn add(self, _rhs: Empty) -> Self::Output {
        self
    }
}
impl Sub<Empty> for BsBool8 {
    type Output = Self;
    fn sub(self, _rhs: Empty) -> Self::Output {
        self
    }
}
impl Mul<Empty> for BsBool8 {
    type Output = Empty;
    fn mul(self, _rhs: Empty) -> Self::Output {
        Empty
    }
}

impl<Fp: PrimeField> BitDecompose<Fp> for BsBool8 {
    type Output = Fp;
    fn bit_decompose(&self, len: usize) -> impl Iterator<Item = Self::Output> {
        assert_eq!(len, 8);
        (0..8).into_iter()
            .map(|i| Fp::from((self.0 & (1 << i) != 0) as u64))
    }
}

impl CountOnes for BsBool8 {
    type Output = u32;
    fn count_ones(&self) -> Self::Output {
        self.0.count_ones()
    }
}

impl CountOnesParity for BsBool8 {
    type Output = bool;
    fn count_ones_parity(&self) -> Self::Output {
        self.0.count_ones() % 2 != 0
    }
}

impl RngExt for BsBool8 {
    fn fill<R: Rng + CryptoRng>(rng: &mut R, buf: &mut [Self]) {
        let mut v = vec![0u8; buf.len()];
        rng.fill_bytes(&mut v);
        buf.iter_mut().zip(v).for_each(|(x, r)| x.0 = r)
    }

    fn generate<R: Rng + CryptoRng>(rng: &mut R, n: usize) -> Vec<Self> {
        let mut r = vec![0; n];
        rng.fill_bytes(&mut r);
        r.into_iter().map(Self).collect()
    }
}

impl DigestExt for BsBool8 {
    fn update<D: Digest>(digest: &mut D, message: &[Self]) {
        for x in message {
            digest.update([x.0]);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::rep3_core::{network::NetSerializable, party::RngExt, share::HasZero};
    use crate::share::bs_bool16::BsBool16;
    use rand::thread_rng;

    #[test]
    fn serialization() {
        let mut rng = thread_rng();
        let list_even: Vec<BsBool16> = BsBool16::generate(&mut rng, 500);
        let list_odd: Vec<BsBool16> = BsBool16::generate(&mut rng, 45);

        assert_eq!(
            list_even,
            BsBool16::from_byte_vec(
                BsBool16::as_byte_vec(&list_even, list_even.len()),
                list_even.len()
            )
        );
        assert_eq!(
            list_odd,
            BsBool16::from_byte_vec(
                BsBool16::as_byte_vec(&list_odd, list_odd.len()),
                list_odd.len()
            )
        );

        let mut slice_even = [BsBool16::ZERO; 500];
        let mut slice_odd = [BsBool16::ZERO; 45];

        BsBool16::from_byte_slice(
            BsBool16::as_byte_vec(&list_even, list_even.len()),
            &mut slice_even,
        );
        assert_eq!(&list_even, &slice_even);

        BsBool16::from_byte_slice(
            BsBool16::as_byte_vec(&list_odd, list_odd.len()),
            &mut slice_odd,
        );
        assert_eq!(&list_odd, &slice_odd);
    }
}
