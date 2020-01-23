use {
    criterion::{black_box, criterion_group, criterion_main, Criterion},
    rand::prelude::*,
    rayon::prelude::*,
    simd_par_bench::*,
};

const COUNT: usize = 1000;

fn scalar_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut data = Vec::<([f32; 32], [f32; 32], [f32; 32], [f32; 32])>::with_capacity(COUNT);
    for _ in 0..COUNT {
        data.push(rng.gen());
    }

    c.bench_function("scalar", |b| {
        b.iter(|| {
            data.iter().for_each(|(x1, y1, x2, y2)| unsafe {
                distance_scalar(black_box(x1), black_box(y1), black_box(x2), black_box(y2));
            })
        })
    });
}

fn par_scalar_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut data = Vec::<([f32; 32], [f32; 32], [f32; 32], [f32; 32])>::with_capacity(COUNT);
    for _ in 0..COUNT {
        data.push(rng.gen());
    }

    c.bench_function("par_scalar", |b| {
        b.iter(|| {
            data.par_iter().for_each(|(x1, y1, x2, y2)| unsafe {
                distance_scalar(black_box(x1), black_box(y1), black_box(x2), black_box(y2));
            })
        })
    });
}

fn runtime_select_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut data = Vec::<([f32; 32], [f32; 32], [f32; 32], [f32; 32])>::with_capacity(COUNT);
    for _ in 0..COUNT {
        data.push(rng.gen());
    }

    c.bench_function("runtime_select", |b| {
        b.iter(|| {
            data.iter().for_each(|(x1, y1, x2, y2)| {
                distance_runtime_select(black_box(x1), black_box(y1), black_box(x2), black_box(y2));
            })
        })
    });
}

fn par_runtime_select_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut data = Vec::<([f32; 32], [f32; 32], [f32; 32], [f32; 32])>::with_capacity(COUNT);
    for _ in 0..COUNT {
        data.push(rng.gen());
    }

    c.bench_function("par_runtime_select", |b| {
        b.iter(|| {
            data.par_iter().for_each(|(x1, y1, x2, y2)| {
                distance_runtime_select(black_box(x1), black_box(y1), black_box(x2), black_box(y2));
            })
        })
    });
}

criterion_group!(
    benches,
    scalar_benchmark,
    par_scalar_benchmark,
    runtime_select_benchmark,
    par_runtime_select_benchmark
);
criterion_main!(benches);
