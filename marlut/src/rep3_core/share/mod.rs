use super::{network::NetSerializable, party::DigestExt};
use crate::rep3_core::party::RngExt;
use crate::share::{BasicFieldLike, CountOnesParity, Empty, Field, PrimeField};
use crate::share::{BitDecompose, FieldLike};
use rand::{CryptoRng, Rng};
use std::ops::{BitAnd, BitXor, BitXorAssign, MulAssign, SubAssign};
use std::{
    borrow::Borrow,
    iter::Sum,
    ops::{Add, AddAssign, Mul, Neg, Sub},
};

// Provides the neutral element of addition
pub trait HasZero {
    /// Zero the neutral element of addition
    const ZERO: Self;
}

/// A party's RSS-share of a (2,3)-shared field element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RssShareGeneral<T, U> {
    /// The first share of the party.
    pub si: T,
    /// The second share of the party.
    pub sii: U,
}

impl<T, U> RssShareGeneral<T, U> {
    /// Computes an RSS-share given two shares.
    pub fn from(si: T, sii: U) -> Self {
        Self { si, sii }
    }
}

impl<T: Add<T2>, U: Add<U2>, T2, U2> Add<RssShareGeneral<T2, U2>> for RssShareGeneral<T, U> {
    type Output = RssShareGeneral<<T as Add<T2>>::Output, <U as Add<U2>>::Output>;

    fn add(self, rhs: RssShareGeneral<T2, U2>) -> Self::Output {
        RssShareGeneral {
            si: self.si + rhs.si,
            sii: self.sii + rhs.sii,
        }
    }
}

impl<T: Sub<T2>, U: Sub<U2>, T2, U2> Sub<RssShareGeneral<T2, U2>> for RssShareGeneral<T, U> {
    type Output = RssShareGeneral<<T as Sub<T2>>::Output, <U as Sub<U2>>::Output>;

    fn sub(self, rhs: RssShareGeneral<T2, U2>) -> Self::Output {
        RssShareGeneral {
            si: self.si - rhs.si,
            sii: self.sii - rhs.sii,
        }
    }
}

impl<F: Clone, T: BitAnd<F>, U: BitAnd<F>> BitAnd<F> for RssShareGeneral<T, U> {
    type Output = RssShareGeneral<<T as BitAnd<F>>::Output, <U as BitAnd<F>>::Output>;

    fn bitand(self, rhs: F) -> Self::Output {
        RssShareGeneral {
            si: self.si & rhs.clone(),
            sii: self.sii & rhs.clone(),
        }
    }
}

impl<T: BitXor<T2>, U: BitXor<U2>, T2, U2> BitXor<RssShareGeneral<T2, U2>> for RssShareGeneral<T, U> {
    type Output = RssShareGeneral<<T as BitXor<T2>>::Output, <U as BitXor<U2>>::Output>;

    fn bitxor(self, rhs: RssShareGeneral<T2, U2>) -> Self::Output {
        RssShareGeneral {
            si: self.si ^ rhs.si,
            sii: self.sii ^ rhs.sii,
        }
    }
}

/// Multiplies the RSS-share with a scalar.
impl<F: Clone, T: Mul<F>, U: Mul<F>> Mul<F> for RssShareGeneral<T, U> {
    type Output = RssShareGeneral<<T as Mul<F>>::Output, <U as Mul<F>>::Output>;

    fn mul(self, rhs: F) -> Self::Output {
        RssShareGeneral {
            si: self.si * rhs.clone(),
            sii: self.sii * rhs.clone(),
        }
    }
}

impl<F: Clone, T: MulAssign<F>, U: MulAssign<F>> MulAssign<F> for RssShareGeneral<T, U> {
    fn mul_assign(&mut self, rhs: F) {
        self.si *= rhs.clone();
        self.sii *= rhs;
    }
}

// impl<F: Field, T: Add<F>, U: Add<F>> Add<F> for RssShareGeneral<T, U> {
//     type Output = RssShareGeneral<<T as Add<F>>::Output, <U as Add<F>>::Output>;

//     fn add(self, rhs: F) -> Self::Output {
//         RssShareGeneral {
//             si: self.si + rhs.clone(),
//             sii: self.sii + rhs.clone(),
//         }
//     }
// }

// impl<F: Field, T: Sub<F>, U: Sub<F>> Sub<F> for RssShareGeneral<T, U> {
//     type Output = RssShareGeneral<<T as Sub<F>>::Output, <U as Sub<F>>::Output>;

//     fn sub(self, rhs: F) -> Self::Output {
//         RssShareGeneral {
//             si: self.si - rhs.clone(),
//             sii: self.sii - rhs.clone(),
//         }
//     }
// }

impl<T: AddAssign, U: AddAssign> AddAssign for RssShareGeneral<T, U> {
    fn add_assign(&mut self, rhs: Self) {
        self.si += rhs.si;
        self.sii += rhs.sii;
    }
}

impl<T: SubAssign, U: SubAssign> SubAssign for RssShareGeneral<T, U> {
    fn sub_assign(&mut self, rhs: Self) {
        self.si -= rhs.si;
        self.sii -= rhs.sii;
    }
}

impl<T: Neg, U: Neg> Neg for RssShareGeneral<T, U> {
    type Output = RssShareGeneral<<T as Neg>::Output, <U as Neg>::Output>;

    fn neg(self) -> Self::Output {
        Self::Output {
            si: -self.si,
            sii: -self.sii,
        }
    }
}

impl<T: Default, U: Default> Default for RssShareGeneral<T, U> {
    fn default() -> Self {
        Self {
            si: T::default(),
            sii: U::default(),
        }
    }
}

impl<T: BitXorAssign, U: BitXorAssign> BitXorAssign for RssShareGeneral<T, U> {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.si ^= rhs.si;
        self.sii ^= rhs.sii;
    }
}

impl<T: Add<Output = T> + HasZero, U: Add<Output = U> + HasZero> Sum for RssShareGeneral<T, U> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, e| Self {
            si: acc.si + e.si,
            sii: acc.sii + e.sii,
        })
        .unwrap_or(Self::ZERO)
    }
}

impl<T: Copy, U: Copy> Copy for RssShareGeneral<T, U> {}

impl<T: HasZero, U: HasZero> HasZero for RssShareGeneral<T, U> {
    const ZERO: Self = Self {
        si: T::ZERO,
        sii: U::ZERO,
    };
}

impl<T: NetSerializable + Clone, U: NetSerializable + Clone> NetSerializable
    for RssShareGeneral<T, U>
{
    fn serialized_size(n_elements: usize) -> usize {
        T::serialized_size(n_elements) + U::serialized_size(n_elements)
    }

    fn as_byte_vec(it: impl IntoIterator<Item = impl Borrow<Self>>, _len: usize) -> Vec<u8> {
        it.into_iter()
            .flat_map(|el| {
                let share = el.borrow();
                let mut ret1 = T::as_byte_vec_slice(&[share.si.clone()]);
                let mut ret2 = U::as_byte_vec_slice(&[share.sii.clone()]);
                ret1.append(&mut ret2);
                ret1
            })
            .collect()
    }

    fn as_byte_vec_slice(elements: &[Self]) -> Vec<u8> {
        elements
            .iter()
            .flat_map(|el| {
                let share = el;
                let mut ret1 = T::as_byte_vec_slice(&[share.si.clone()]);
                let mut ret2 = U::as_byte_vec_slice(&[share.sii.clone()]);
                ret1.append(&mut ret2);
                ret1
            })
            .collect()
    }

    fn from_byte_vec(v: Vec<u8>, _len: usize) -> Vec<Self> {
        let size_T = T::serialized_size(1);
        let size_U = U::serialized_size(1);
        v.chunks_exact(size_T + size_U)
            .map(|data| {
                Self::from(
                    T::from_byte_vec(data[..size_T].to_vec(), 1)[0].clone(),
                    U::from_byte_vec(data[size_T..].to_vec(), 1)[0].clone(),
                )
            })
            .collect()
    }

    fn from_byte_slice(v: Vec<u8>, dest: &mut [Self]) {
        let size_T = T::serialized_size(1);
        let size_U = U::serialized_size(1);
        v.chunks_exact(size_T + size_U)
            .zip(dest.iter_mut())
            .for_each(|(data, out)| {
                *out = Self::from(
                    T::from_byte_vec(data[..size_T].to_vec(), 1)[0].clone(),
                    U::from_byte_vec(data[size_T..].to_vec(), 1)[0].clone(),
                )
            });
    }
}

impl<T1: BasicFieldLike, U1: BasicFieldLike> BasicFieldLike for RssShareGeneral<T1, U1> {}

impl<T: BitDecompose<Fp>, U: BitDecompose<Fp>, Fp: FieldLike> BitDecompose<Fp>
    for RssShareGeneral<T, U>
{
    type Output = RssShareGeneral<<T as BitDecompose<Fp>>::Output, <U as BitDecompose<Fp>>::Output>;

    fn bit_decompose(&self, len: usize) -> impl Iterator<Item = Self::Output> {
        self.si
            .bit_decompose(len)
            .into_iter()
            .zip(self.sii.bit_decompose(len))
            .map(|(si, sii)| RssShareGeneral::from(si, sii))
    }
}

impl<F: PrimeField> From<RssShareGeneral<F, Empty>> for RssShare<F> {
    fn from(value: RssShareGeneral<F, Empty>) -> Self {
        Self {
            si: value.si,
            sii: F::ZERO,
        }
    }
}

impl<F: PrimeField> From<RssShareGeneral<Empty, F>> for RssShare<F> {
    fn from(value: RssShareGeneral<Empty, F>) -> Self {
        Self {
            si: F::ZERO,
            sii: value.sii,
        }
    }
}

impl<F: PrimeField> From<RssShareGeneral<Empty, Empty>> for RssShare<F> {
    fn from(_value: RssShareGeneral<Empty, Empty>) -> Self {
        Self {
            si: F::ZERO,
            sii: F::ZERO,
        }
    }
}

pub trait Lift<F: FieldLike> {
    type Output: BasicFieldLike + Mul<F, Output = Self::Output>;
    fn lift(self) -> Self::Output;
}

impl<F: Field, Fp: PrimeField> Lift<Fp> for F {
    type Output = Fp;
    fn lift(self) -> Self::Output {
        Fp::from(self.as_raw() as u64)
    }
}

impl<Fp: PrimeField> Lift<Fp> for u32 {
    type Output = Fp;
    fn lift(self) -> Self::Output {
        Fp::from(self as u64)
    }
}

impl<Fp: FieldLike> Lift<Fp> for Empty {
    type Output = Empty;
    fn lift(self) -> Self::Output {
        Empty
    }
}

impl<T1: Lift<Fp>, T2: Lift<Fp>, Fp: PrimeField> Lift<Fp> for RssShareGeneral<T1, T2> {
    type Output = RssShareGeneral<<T1 as Lift<Fp>>::Output, <T2 as Lift<Fp>>::Output>;
    fn lift(self) -> Self::Output {
        RssShareGeneral::from(self.si.lift(), self.sii.lift())
    }
}

impl<T: CountOnesParity, U: CountOnesParity> CountOnesParity for RssShareGeneral<T, U> {
    type Output = RssShareGeneral<<T as CountOnesParity>::Output, <U as CountOnesParity>::Output>;
    fn count_ones_parity(&self) -> Self::Output {
        RssShareGeneral::from(self.si.count_ones_parity(), self.sii.count_ones_parity())
    }
}

pub type RssShare<F> = RssShareGeneral<F, F>;

/// A vector of [RssShare]s.
pub type RssShareVec<F> = Vec<RssShare<F>>;

pub type EmptyRssShare = RssShareGeneral<Empty, Empty>;

pub const EmptyRssShare: EmptyRssShare = RssShareGeneral {
    si: Empty,
    sii: Empty,
};

#[cfg(test)]
mod tests {
    use super::RssShareGeneral;
    use crate::rep3_core::network::NetSerializable;
    use crate::share::{Empty, mersenne61::Mersenne61};

    #[test]
    fn test_serialization() {
        let v = vec![
            RssShareGeneral::from(Empty, Mersenne61(0x1)),
            RssShareGeneral::from(Empty, Mersenne61(0xffffffffffffff)),
            RssShareGeneral::from(Empty, Mersenne61(0x3)),
            RssShareGeneral::from(Empty, Mersenne61(0x12345781234578)),
        ];
        let as_bytes = RssShareGeneral::as_byte_vec(v.iter(), 4);
        let v_new = RssShareGeneral::from_byte_vec(as_bytes, 4);
        assert_eq!(v_new, v);
    }

    #[test]
    fn test_serialization2() {
        let v = vec![
            RssShareGeneral::from(Mersenne61(0x1), Empty),
            RssShareGeneral::from(Mersenne61(0xffffffffffffff), Empty),
            RssShareGeneral::from(Mersenne61(0x3), Empty),
            RssShareGeneral::from(Mersenne61(0x12345781234578), Empty),
        ];
        let as_bytes = RssShareGeneral::as_byte_vec(v.iter(), 4);
        let v_new = RssShareGeneral::from_byte_vec(as_bytes, 4);
        assert_eq!(v_new, v);
    }

    #[test]
    fn test_serialization3() {
        let v = vec![
            RssShareGeneral::from(Mersenne61(0x1), Mersenne61(0xffffffffffffff)),
            RssShareGeneral::from(Mersenne61(0xffffffffffffff), Mersenne61(0x3)),
            RssShareGeneral::from(Mersenne61(0x3), Mersenne61(0x12345781234578)),
            RssShareGeneral::from(Mersenne61(0x12345781234578), Mersenne61(0x1)),
        ];
        let as_bytes = RssShareGeneral::as_byte_vec(v.iter(), 4);
        let v_new = RssShareGeneral::from_byte_vec(as_bytes, 4);
        assert_eq!(v_new, v);
    }
}
