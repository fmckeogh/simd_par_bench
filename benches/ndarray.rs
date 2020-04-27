use {
    criterion::{black_box, criterion_group, criterion_main, Benchmark, Criterion},
    ndarray::{Axis, Zip},
    rayon::prelude::*,
    simd_par_bench::{gen_array, NZ},
    std::sync::{Arc, Mutex},
};

fn muladd_benchmark(c: &mut Criterion) {
    c.bench(
        "ndarray",
        Benchmark::new("muladd_zip", |b| {
            let mut arr1 = gen_array();
            let arr2 = gen_array();
            let arr3 = gen_array();

            b.iter(|| {
                Zip::from(black_box(&mut arr1))
                    .and(black_box(&arr2))
                    .and(black_box(&arr3))
                    .apply(|a, b, c| *a += b * c);
            })
        })
        .sample_size(10),
    );

    c.bench(
        "ndarray",
        Benchmark::new("muladd_par_zip", |b| {
            let mut arr1 = gen_array();
            let arr2 = gen_array();
            let arr3 = gen_array();

            b.iter(|| {
                Zip::from(black_box(&mut arr1))
                    .and(black_box(&arr2))
                    .and(black_box(&arr3))
                    .par_apply(|a, b, c| *a += b * c);
            })
        })
        .sample_size(10),
    );

    c.bench(
        "ndarray",
        Benchmark::new("muladd_outer", |b| {
            let mut arr1 = gen_array();
            let arr2 = gen_array();
            let arr3 = gen_array();

            b.iter(|| {
                black_box(&mut arr1)
                    .axis_iter_mut(Axis(2))
                    .zip(black_box(&arr2).axis_iter(Axis(2)))
                    .zip(black_box(&arr3).axis_iter(Axis(2)))
                    .for_each(|((a, b), c)| {
                        Zip::from(a).and(&b).and(&c).apply(|a, b, c| *a += b * c)
                    });
            })
        })
        .sample_size(10),
    );

    c.bench(
        "ndarray",
        Benchmark::new("muladd_par_outer", |b| {
            let mut arr1 = gen_array();
            let arr2 = gen_array();
            let arr3 = gen_array();

            b.iter(|| {
                black_box(&mut arr1)
                    .axis_iter_mut(Axis(2))
                    .into_par_iter()
                    .zip(black_box(&arr2).axis_iter(Axis(2)).into_par_iter())
                    .zip(black_box(&arr3).axis_iter(Axis(2)).into_par_iter())
                    .for_each(|((a, b), c)| {
                        Zip::from(a).and(&b).and(&c).apply(|a, b, c| *a += b * c)
                    });
            })
        })
        .sample_size(10),
    );

    c.bench(
        "ndarray",
        Benchmark::new("muladd_par", |b| {
            let arr1 = Arc::new(Mutex::new(gen_array()));
            let arr2 = gen_array();
            let arr3 = gen_array();

            b.iter(|| {
                (0..=NZ).into_par_iter().for_each(|iz| {
                    Zip::from(arr1.lock().unwrap().index_axis_mut(Axis(2), iz))
                        .and(black_box(&arr2).index_axis(Axis(2), iz))
                        .and(black_box(&arr3).index_axis(Axis(2), iz))
                        .apply(|a, b, c| *a += b * c);
                });
            });
        })
        .sample_size(10),
    );
}

criterion_group!(benches, muladd_benchmark);
criterion_main!(benches);
