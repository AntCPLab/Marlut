//! This crate implements different LUT protocols.
#![feature(bigint_helper_methods)]
#![feature(allocator_api)]
#![feature(portable_simd)]
#![feature(generic_const_exprs)]
#![feature(iter_array_chunks)]
#![allow(dead_code)]
pub mod aes;
pub mod chida;
pub mod lut256;
pub mod lut_sp;
pub mod lut_sp_malsec;
pub mod lut_sp_boolean;
pub mod lut_sp_boolean_malsec;
pub mod rep3_core;
pub mod share;
pub mod util;
pub mod wollut16;
pub mod wollut16_malsec;
