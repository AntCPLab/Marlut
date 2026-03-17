//! This module implements the unsigned integer rings Z_{2^N}.
use std::{
    borrow::Borrow,
    iter::Sum,
    num::Wrapping,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{
    rep3_core::{
        network::NetSerializable,
        party::{DigestExt, RngExt},
        share::HasZero,
    },
    share::BasicFieldLike,
};
use rand::{CryptoRng, Rng};
use sha2::Digest;

use super::{Empty, Field, FieldLike};

#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct UR8(pub Wrapping<u8>);

impl UR8 {
    pub fn new(bits: u8) -> Self {
        Self(Wrapping(bits))
    }

    pub fn as_u8(&self) -> u8 {
        self.0.0
    }
}

impl BasicFieldLike for UR8 {}

impl FieldLike for UR8 {
    const NBYTES: usize = 1;
    const IS_UR: bool = true;

    fn as_raw(&self) -> usize {
        self.0.0 as usize
    }

    fn from_raw(a: usize) -> Self {
        Self::new(a as u8)
    }
}

impl Field for UR8 {
    const ONE: Self = Self(Wrapping(1));

    fn is_zero(&self) -> bool {
        self.0.0 == 0
    }
}

impl Sum for UR8 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl HasZero for UR8 {
    const ZERO: Self = Self(Wrapping(0));
}

impl NetSerializable for UR8 {
    fn serialized_size(n_elements: usize) -> usize {
        n_elements
    }

    fn as_byte_vec(it: impl IntoIterator<Item = impl Borrow<Self>>, _len: usize) -> Vec<u8> {
        it.into_iter().map(|el| el.borrow().0.0 as u8).collect()
    }

    fn as_byte_vec_slice(elements: &[Self]) -> Vec<u8> {
        elements.iter().map(|x| x.0.0).collect()
    }

    fn from_byte_vec(v: Vec<u8>, _len: usize) -> Vec<Self> {
        v.into_iter().map(Self::new).collect()
    }

    fn from_byte_slice(v: Vec<u8>, dest: &mut [Self]) {
        v.into_iter()
            .zip(dest)
            .for_each(|(byte, dst)| *dst = Self::new(byte))
    }
}

impl Neg for UR8 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Mul for UR8 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Sub for UR8 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Add for UR8 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for UR8 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for UR8 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl MulAssign for UR8 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Add<Empty> for UR8 {
    type Output = Self;
    fn add(self, _rhs: Empty) -> Self::Output {
        self
    }
}
impl Sub<Empty> for UR8 {
    type Output = Self;
    fn sub(self, _rhs: Empty) -> Self::Output {
        self
    }
}
impl Mul<Empty> for UR8 {
    type Output = Empty;
    fn mul(self, _rhs: Empty) -> Self::Output {
        Empty
    }
}

impl RngExt for UR8 {
    fn fill<R: Rng + CryptoRng>(rng: &mut R, buf: &mut [Self]) {
        let mut v = vec![0u8; buf.len()];
        rng.fill_bytes(&mut v);
        buf.iter_mut().zip(v).for_each(|(x, r)| x.0.0 = r)
    }

    fn generate<R: Rng + CryptoRng>(rng: &mut R, n: usize) -> Vec<Self> {
        let mut r = vec![0; n];
        rng.fill_bytes(&mut r);
        r.into_iter().map(Self::new).collect()
    }
}

impl DigestExt for UR8 {
    fn update<D: Digest>(digest: &mut D, message: &[Self]) {
        for x in message {
            digest.update([x.0.0]);
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct UR16(pub Wrapping<u16>);

impl UR16 {
    pub fn new(bits: u16) -> Self {
        Self(Wrapping(bits))
    }

    pub fn as_u16(&self) -> u16 {
        self.0.0
    }
}

impl BasicFieldLike for UR16 {}

impl FieldLike for UR16 {
    const NBYTES: usize = 2;
    const IS_UR: bool = true;

    fn as_raw(&self) -> usize {
        self.0.0 as usize
    }

    fn from_raw(a: usize) -> Self {
        Self::new(a as u16)
    }
}

impl Field for UR16 {
    const ONE: Self = Self(Wrapping(1));

    fn is_zero(&self) -> bool {
        self.0.0 == 0
    }
}

impl Sum for UR16 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl HasZero for UR16 {
    const ZERO: Self = Self(Wrapping(0));
}

impl NetSerializable for UR16 {
    fn serialized_size(n_elements: usize) -> usize {
        n_elements * 2
    }

    fn as_byte_vec(it: impl IntoIterator<Item = impl Borrow<Self>>, _len: usize) -> Vec<u8> {
        it.into_iter()
            .flat_map(|el| el.borrow().0.0.to_le_bytes())
            .collect()
    }

    fn as_byte_vec_slice(elements: &[Self]) -> Vec<u8> {
        elements.iter().flat_map(|x| x.0.0.to_le_bytes()).collect()
    }

    fn from_byte_vec(v: Vec<u8>, _len: usize) -> Vec<Self> {
        v.into_iter()
            .array_chunks::<2>()
            .map(|x| Self::new(u16::from_le_bytes(x)))
            .collect()
    }

    fn from_byte_slice(v: Vec<u8>, dest: &mut [Self]) {
        v.into_iter()
            .array_chunks::<2>()
            .zip(dest)
            .for_each(|(byte, dst)| *dst = Self::new(u16::from_le_bytes(byte)))
    }
}

impl Neg for UR16 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Mul for UR16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Sub for UR16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Add for UR16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for UR16 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for UR16 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl MulAssign for UR16 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Add<Empty> for UR16 {
    type Output = Self;
    fn add(self, _rhs: Empty) -> Self::Output {
        self
    }
}
impl Sub<Empty> for UR16 {
    type Output = Self;
    fn sub(self, _rhs: Empty) -> Self::Output {
        self
    }
}
impl Mul<Empty> for UR16 {
    type Output = Empty;
    fn mul(self, _rhs: Empty) -> Self::Output {
        Empty
    }
}

impl RngExt for UR16 {
    fn fill<R: Rng + CryptoRng>(rng: &mut R, buf: &mut [Self]) {
        let mut v = vec![0u8; buf.len() * 2];
        rng.fill_bytes(&mut v);
        buf.iter_mut()
            .zip(v.into_iter().array_chunks::<2>())
            .for_each(|(x, r)| *x = Self::new(u16::from_le_bytes(r)))
    }

    fn generate<R: Rng + CryptoRng>(rng: &mut R, n: usize) -> Vec<Self> {
        let mut r = vec![0; n * 2];
        rng.fill_bytes(&mut r);
        r.into_iter()
            .array_chunks::<2>()
            .map(|x| Self::new(u16::from_le_bytes(x)))
            .collect()
    }
}

impl DigestExt for UR16 {
    fn update<D: Digest>(digest: &mut D, message: &[Self]) {
        for x in message {
            digest.update(x.0.0.to_le_bytes());
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct UR32(pub Wrapping<u32>);

impl UR32 {
    pub fn new(bits: u32) -> Self {
        Self(Wrapping(bits))
    }

    pub fn as_u32(&self) -> u32 {
        self.0.0
    }
}

impl BasicFieldLike for UR32 {}

impl FieldLike for UR32 {
    const NBYTES: usize = 4;

    fn as_raw(&self) -> usize {
        self.0.0 as usize
    }

    fn from_raw(a: usize) -> Self {
        Self::new(a as u32)
    }
}

impl Field for UR32 {
    const ONE: Self = Self(Wrapping(1));

    fn is_zero(&self) -> bool {
        self.0.0 == 0
    }
}

impl Sum for UR32 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl HasZero for UR32 {
    const ZERO: Self = Self(Wrapping(0));
}

impl NetSerializable for UR32 {
    fn serialized_size(n_elements: usize) -> usize {
        n_elements * 4
    }

    fn as_byte_vec(it: impl IntoIterator<Item = impl Borrow<Self>>, _len: usize) -> Vec<u8> {
        it.into_iter()
            .flat_map(|el| el.borrow().0.0.to_le_bytes())
            .collect()
    }

    fn as_byte_vec_slice(elements: &[Self]) -> Vec<u8> {
        elements.iter().flat_map(|x| x.0.0.to_le_bytes()).collect()
    }

    fn from_byte_vec(v: Vec<u8>, _len: usize) -> Vec<Self> {
        v.into_iter()
            .array_chunks::<4>()
            .map(|x| Self::new(u32::from_le_bytes(x)))
            .collect()
    }

    fn from_byte_slice(v: Vec<u8>, dest: &mut [Self]) {
        v.into_iter()
            .array_chunks::<4>()
            .zip(dest)
            .for_each(|(byte, dst)| *dst = Self::new(u32::from_le_bytes(byte)))
    }
}

impl Neg for UR32 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Mul for UR32 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Sub for UR32 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Add for UR32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for UR32 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for UR32 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl MulAssign for UR32 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Add<Empty> for UR32 {
    type Output = Self;
    fn add(self, _rhs: Empty) -> Self::Output {
        self
    }
}
impl Sub<Empty> for UR32 {
    type Output = Self;
    fn sub(self, _rhs: Empty) -> Self::Output {
        self
    }
}
impl Mul<Empty> for UR32 {
    type Output = Empty;
    fn mul(self, _rhs: Empty) -> Self::Output {
        Empty
    }
}

impl RngExt for UR32 {
    fn fill<R: Rng + CryptoRng>(rng: &mut R, buf: &mut [Self]) {
        let mut v = vec![0u8; buf.len() * 4];
        rng.fill_bytes(&mut v);
        buf.iter_mut()
            .zip(v.into_iter().array_chunks::<4>())
            .for_each(|(x, r)| *x = Self::new(u32::from_le_bytes(r)))
    }

    fn generate<R: Rng + CryptoRng>(rng: &mut R, n: usize) -> Vec<Self> {
        let mut r = vec![0; n * 4];
        rng.fill_bytes(&mut r);
        r.into_iter()
            .array_chunks::<4>()
            .map(|x| Self::new(u32::from_le_bytes(x)))
            .collect()
    }
}

impl DigestExt for UR32 {
    fn update<D: Digest>(digest: &mut D, message: &[Self]) {
        for x in message {
            digest.update(x.0.0.to_le_bytes());
        }
    }
}
