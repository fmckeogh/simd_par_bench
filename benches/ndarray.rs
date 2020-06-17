use {
    criterion::{black_box, criterion_group, criterion_main, Benchmark, Criterion},
    ndarray::{Array2, Array3, Axis, ShapeBuilder, Zip},
    rand::prelude::*,
    rand_xorshift::XorShiftRng,
    rayon::prelude::*,
    std::sync::{Arc, Mutex},
};

const NG: usize = 512;
const NZ: usize = 64;
const SAMPLE_SIZE: usize = 10;

fn gen_vec_array2() -> Vec<Array2<f64>> {
    let mut result = Vec::with_capacity(NZ + 1);
    let mut rng = XorShiftRng::from_entropy();

    for _ in 0..NZ + 1 {
        let arr2 = {
            let mut data = Vec::with_capacity(NG * NG);

            for _ in 0..NG * NG {
                data.push(rng.gen());
            }

            Array2::<f64>::from_shape_vec((NG, NG).f(), data).unwrap()
        };
        result.push(arr2);
    }

    result
}

fn gen_array3() -> Array3<f64> {
    let mut rng = XorShiftRng::from_entropy();
    let mut data = Vec::with_capacity(NG * NG * (NZ + 1));

    for _ in 0..NG * NG * (NZ + 1) {
        data.push(rng.gen());
    }

    Array3::<f64>::from_shape_vec((NG, NG, NZ + 1).f(), data).unwrap()
}

fn muladd_benchmark(c: &mut Criterion) {
    c.bench(
        "ndarray",
        Benchmark::new("muladd_zip", |b| {
            let mut arr1 = gen_array3();
            let arr2 = gen_array3();
            let arr3 = gen_array3();

            b.iter(|| {
                Zip::from(black_box(&mut arr1))
                    .and(black_box(&arr2))
                    .and(black_box(&arr3))
                    .apply(|a, b, c| *a += b.exp() * c.exp());
            })
        })
        .sample_size(SAMPLE_SIZE),
    );

    c.bench(
        "ndarray",
        Benchmark::new("muladd_zip_par", |b| {
            let mut arr1 = gen_array3();
            let arr2 = gen_array3();
            let arr3 = gen_array3();

            b.iter(|| {
                Zip::from(black_box(&mut arr1))
                    .and(black_box(&arr2))
                    .and(black_box(&arr3))
                    .par_apply(|a, b, c| *a += b.exp() * c.exp());
            })
        })
        .sample_size(SAMPLE_SIZE),
    );

    c.bench(
        "ndarray",
        Benchmark::new("muladd_outer", |b| {
            let mut arr1 = gen_array3();
            let arr2 = gen_array3();
            let arr3 = gen_array3();

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
        .sample_size(SAMPLE_SIZE),
    );

    c.bench(
        "ndarray",
        Benchmark::new("muladd_par_outer", |b| {
            let mut arr1 = gen_array3();
            let arr2 = gen_array3();
            let arr3 = gen_array3();

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
        .sample_size(SAMPLE_SIZE),
    );

    c.bench(
        "ndarray",
        Benchmark::new("muladd_par", |b| {
            let arr1 = Arc::new(Mutex::new(gen_array3()));
            let arr2 = gen_array3();
            let arr3 = gen_array3();

            b.iter(|| {
                (0..=NZ).into_par_iter().for_each(|iz| {
                    Zip::from(arr1.lock().unwrap().index_axis_mut(Axis(2), iz))
                        .and(black_box(&arr2).index_axis(Axis(2), iz))
                        .and(black_box(&arr3).index_axis(Axis(2), iz))
                        .apply(|a, b, c| *a += b * c);
                });
            });
        })
        .sample_size(SAMPLE_SIZE),
    );

    c.bench(
        "ndarray",
        Benchmark::new("muladd_arr2vec", |b| {
            let mut arr1 = gen_vec_array2();
            let arr2 = gen_vec_array2();
            let arr3 = gen_vec_array2();

            b.iter(|| {
                arr1.iter_mut()
                    .zip(arr2.iter())
                    .zip(arr3.iter())
                    .for_each(|((a, b), c)| {
                        Zip::from(a).and(b).and(c).apply(|a, b, c| *a += b * c);
                    })
            })
        })
        .sample_size(SAMPLE_SIZE),
    );

    c.bench(
        "ndarray",
        Benchmark::new("muladd_arr2vec_par", |b| {
            let mut arr1 = gen_vec_array2();
            let arr2 = gen_vec_array2();
            let arr3 = gen_vec_array2();

            b.iter(|| {
                arr1.par_iter_mut()
                    .zip(arr2.par_iter())
                    .zip(arr3.par_iter())
                    .for_each(|((a, b), c)| {
                        Zip::from(a).and(b).and(c).apply(|a, b, c| *a += b * c);
                    })
            })
        })
        .sample_size(SAMPLE_SIZE),
    );

    c.bench(
        "ndarray",
        Benchmark::new("muladd_vec", |b| {
            let mut rng = XorShiftRng::from_entropy();

            let mut arr1 = Vec::<f64>::with_capacity(NG * NG * (NZ + 1));
            let mut arr2 = Vec::<f64>::with_capacity(NG * NG * (NZ + 1));
            let mut arr3 = Vec::<f64>::with_capacity(NG * NG * (NZ + 1));

            for _ in 0..NG * NG * (NZ + 1) {
                arr1.push(rng.gen());
                arr2.push(rng.gen());
                arr3.push(rng.gen());
            }

            b.iter(|| {
                arr1.iter_mut()
                    .zip(arr2.iter())
                    .zip(arr3.iter())
                    .for_each(|((a, b), c)| *a += b * c)
            })
        })
        .sample_size(SAMPLE_SIZE),
    );

    c.bench(
        "ndarray",
        Benchmark::new("muladd_vec_par", |b| {
            let mut rng = XorShiftRng::from_entropy();

            let mut arr1 = Vec::<f64>::with_capacity(NG * NG * (NZ + 1));
            let mut arr2 = Vec::<f64>::with_capacity(NG * NG * (NZ + 1));
            let mut arr3 = Vec::<f64>::with_capacity(NG * NG * (NZ + 1));

            for _ in 0..NG * NG * (NZ + 1) {
                arr1.push(rng.gen());
                arr2.push(rng.gen());
                arr3.push(rng.gen());
            }

            b.iter(|| {
                arr1.par_iter_mut()
                    .zip(arr2.par_iter())
                    .zip(arr3.par_iter())
                    .for_each(|((a, b), c)| *a += b * c)
            })
        })
        .sample_size(SAMPLE_SIZE),
    );

    c.bench(
        "ndarray",
        Benchmark::new("muladd_rayon_zip_1d", |b| {
            let arr1 = gen_array3();
            let mut v1 = vec![0f64; arr1.len()];
            let v2 = vec![0f64; arr1.len()];
            let v3 = vec![0f64; arr1.len()];

            b.iter(|| {
                v1.par_iter_mut()
                    .zip(&v2)
                    .zip(&v3)
                    .for_each(|((a, &b), &c)| *a += b * c);
            })
        })
        .sample_size(SAMPLE_SIZE),
    );

    c.bench(
        "ndarray",
        Benchmark::new("muladd_split_zip", |b| {
            let mut arr1 = gen_array3();
            let arr2 = gen_array3();
            let arr3 = gen_array3();

            b.iter(|| {
                let f = &|a: &mut f64, b: &f64, c: &f64| *a += b * c;
                let (z1, z2) = Zip::from(black_box(&mut arr1))
                    .and(black_box(&arr2))
                    .and(black_box(&arr3))
                    .split();
                rayon::join(move || z1.apply(f), move || z2.apply(f));
            })
        })
        .sample_size(SAMPLE_SIZE),
    );
}

criterion_group!(benches, muladd_benchmark);
criterion_main!(benches);
