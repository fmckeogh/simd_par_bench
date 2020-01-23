use {
    criterion::{black_box, criterion_group, criterion_main, Criterion},
    ndarray::{Array3, Zip},
    rand::prelude::*,
    simd_par_bench::*,
};

const NG: usize = 128;
const NZ: usize = 16;

fn addsub_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut arr1 = {
        let mut data = vec![0.0; NG * NG * NZ];
        for e in data.iter_mut() {
            *e = rng.gen::<f64>();
        }
        Array3::from_shape_vec((NG, NG, NZ), data).unwrap()
    };
    let arr2 = {
        let mut data = vec![0.0; NG * NG * NZ];
        for e in data.iter_mut() {
            *e = rng.gen::<f64>();
        }
        Array3::from_shape_vec((NG, NG, NZ), data).unwrap()
    };

    c.bench_function("addsub_overload", |b| {
        b.iter(|| {
            arr1 += black_box(&arr2);
            arr1 -= black_box(&arr2);
        })
    });
    c.bench_function("addsub_zip", |b| {
        b.iter(|| {
            Zip::from(&mut arr1)
                .and(black_box(&arr2))
                .apply(|a, b| *a += b);
            Zip::from(&mut arr1)
                .and(black_box(&arr2))
                .apply(|a, b| *a -= b);
        })
    });
    c.bench_function("addsub_runtime_select", |b| {
        b.iter(|| {
            add_runtime_select(&mut arr1, black_box(arr2.view()));
            sub_runtime_select(&mut arr1, black_box(arr2.view()));
        })
    });
}

fn muldiv_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut arr1 = {
        let mut data = vec![0.0; NG * NG * NZ];
        for e in data.iter_mut() {
            *e = rng.gen::<f64>();
        }
        Array3::from_shape_vec((NG, NG, NZ), data).unwrap()
    };
    let arr2 = {
        let mut data = vec![0.0; NG * NG * NZ];
        for e in data.iter_mut() {
            *e = rng.gen::<f64>();
        }
        Array3::from_shape_vec((NG, NG, NZ), data).unwrap()
    };

    c.bench_function("muldiv_overload", |b| {
        b.iter(|| {
            arr1 *= black_box(&arr2);
            arr1 /= black_box(&arr2);
        })
    });
    c.bench_function("muldiv_zip", |b| {
        b.iter(|| {
            Zip::from(&mut arr1)
                .and(black_box(&arr2))
                .apply(|a, b| *a *= b);
            Zip::from(&mut arr1)
                .and(black_box(&arr2))
                .apply(|a, b| *a /= b);
        })
    });
}

criterion_group!(benches, addsub_benchmark, muldiv_benchmark);
criterion_main!(benches);
