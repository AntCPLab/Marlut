#![feature(allocator_api)]
use std::vec;

use criterion::*;
use itertools::zip_eq;
use marlut::{
    rep3_core::{party::RngExt, share::HasZero},
    share::{Field, gf2p64::GF2p64, mersenne61::Mersenne61},
    util::aligned_vec::{AlignedAllocator, AlignedVec},
};
use rand::*;

fn bench_field<F: Field>(b: &mut Bencher) {
    let mut rng = thread_rng();
    b.iter_batched(
        || (F::generate(&mut rng, 102400), F::generate(&mut rng, 102400)),
        |(mut x, y)| {
            for (x, y) in zip_eq(&mut x, &y) {
                *x *= *y;
            }
            black_box(x);
        },
        criterion::BatchSize::LargeInput,
    );
}

fn alloc_aligned_blocks<T: HasZero + Clone>(size: usize) -> (AlignedVec<T, 64>, AlignedVec<T, 64>) {
    let e_si = vec::from_elem_in(T::ZERO, size, AlignedAllocator::<64>);
    let e_sii = vec::from_elem_in(T::ZERO, size, AlignedAllocator::<64>);
    (e_si, e_sii)
}

fn bench_multi(b: &mut Bencher) {
    let mut rng = thread_rng();
    b.iter_batched(
        || {
            let (mut x, mut y) = alloc_aligned_blocks(102400);
            Mersenne61::fill(&mut rng, &mut x);
            Mersenne61::fill(&mut rng, &mut y);
            (x, y)
        },
        |(mut x, y)| {
            Mersenne61::mul_assign_multiple_opt_8(&mut x, &y);
            black_box(x);
        },
        criterion::BatchSize::LargeInput,
    );
}

fn bench_multiplication(c: &mut Criterion) {
    c.bench_function("GF2p64", bench_field::<GF2p64>);
    c.bench_function("Mersenne61", bench_field::<Mersenne61>);
    c.bench_function("Mersenne61_new", bench_multi);
}

criterion_group!(field_mul_benchmark, bench_multiplication);
criterion_main!(field_mul_benchmark);
