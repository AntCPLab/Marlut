//! This module implements the 2^{61}-1 Mersenne prime field
use std::{
    fmt::Display,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign},
};

use crate::Field;

pub const MODULUS: u64 = (1 << 61) - 1;

fn modulo_mersenne61(val: u64) -> u64 {
    let val = (val & MODULUS) + (val >> 61);
    if val >= MODULUS {
        val - MODULUS
    } else {
        val
    }
}

fn modulo_mersenne61_carry(val: u64, carry: u64) -> u64 {
    let val = (val & MODULUS) + (val >> 61) + (carry << 3);
    if val >= MODULUS {
        val - MODULUS
    } else {
        val
    }
}

#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Mersenne61(pub u64);

impl Mersenne61 {
    pub const ZERO: Self = Self(0);

    pub fn new(val: u64) -> Self {
        debug_assert!(val < MODULUS);
        Self(val)
    }
}

impl Display for Mersenne61 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mersenne61({})", self.0)
    }
}

impl Sum for Mersenne61 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl Product for Mersenne61 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ONE)
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

impl From<u64> for Mersenne61 {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl Field for Mersenne61 {
    const ONE: Self = Self(1);

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

fn square_times<const N: usize>(val: Mersenne61) -> Mersenne61 {
    let mut ret = val;
    for _ in 0..N {
        ret = ret * ret;
    }
    ret
}

#[cfg(test)]
mod test {
    use super::Mersenne61;
    use crate::Field;

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
}
