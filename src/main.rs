use {
    arrayfire::*,
    ndarray::Array3,
    rand::prelude::*,
    rayon::prelude::*,
    simd_par_bench::*,
    std::time::{SystemTime, UNIX_EPOCH},
};

/*
fn main() {
    let mut rng = rand::thread_rng();

    let mut data = Vec::<([f32; 32], [f32; 32], [f32; 32], [f32; 32])>::with_capacity(COUNT);
    for _ in 0..COUNT {
        data.push(rng.gen());
    }

    dbg!(data
        .par_iter()
        .map(|(x1, y1, x2, y2)| distance_runtime_select(x1, y1, x2, y2))
        .flatten()
        .sum::<f32>());
}
*/
const X: usize = 1024;
const Y: usize = 1024;
const Z: usize = 80;

fn main() {
    let mut rng = rand::thread_rng();
    let a_data = {
        let mut data = Vec::<f64>::with_capacity(X * Y * Z);
        for _ in 0..X * Y * Z {
            data.push(rng.gen());
        }
        data
    };
    let b_data = {
        let mut data = Vec::<f64>::with_capacity(X * Y * Z);
        for _ in 0..X * Y * Z {
            data.push(rng.gen());
        }
        data
    };
    let c_data = {
        let mut data = Vec::<f64>::with_capacity(X * Y * Z);
        for _ in 0..X * Y * Z {
            data.push(rng.gen());
        }
        data
    };
    let d_data = {
        let mut data = Vec::<f64>::with_capacity(X * Y * Z);
        for _ in 0..X * Y * Z {
            data.push(rng.gen());
        }
        data
    };
    arrayfire_test(&a_data, &b_data);
    arrayfire_test(&c_data, &d_data);
    arrayfire_test(&a_data, &b_data);
    arrayfire_test(&c_data, &d_data);
    ndarray_test(a_data, b_data);
    ndarray_test(c_data, d_data);
}

fn arrayfire_test(a: &[f64], b: &[f64]) {
    println!("AF: Loading data...");
    let start = SystemTime::now();
    let dims = Dim4::new(&[X as u64, Y as u64, Z as u64, 1]);
    let a = Array::<f64>::new(a, dims);
    let b = Array::<f64>::new(b, dims);
    let end = SystemTime::now();
    println!(
        "AF: Loading took {}ms\n",
        end.duration_since(start).unwrap().as_millis()
    );
    println!("AF: Starting...");
    let start = SystemTime::now();

    let mut c = &a * &b;
    for _ in 0..20 {
        c = c - &a;
        c = c + &b;
        c = c * &b;
        c = &c * &c * &a;
    }
    let sum = sum_all(&c);

    let end = SystemTime::now();
    println!(
        "AF: Finished!\nTook {}ms\n",
        end.duration_since(start).unwrap().as_millis()
    );
    println!("AF sum: {:?}", sum);
}

fn ndarray_test(a: Vec<f64>, b: Vec<f64>) {
    println!("ND: Loading data...");
    let a = Array3::from_shape_vec((X, Y, Z), a).unwrap();
    let b = Array3::from_shape_vec((X, Y, Z), b).unwrap();

    println!("ND: Starting...");
    let start = SystemTime::now();
    let mut c = &a * &b;
    for _ in 0..20 {
        c = c - &a;
        c = c + &b;
        c = c * &b;
        c = &c * &c * &a;
    }
    let sum = c.sum();
    let end = SystemTime::now();
    println!(
        "ND: Finished!\nTook {}ms\n",
        end.duration_since(start).unwrap().as_millis()
    );
    println!("ND sum: {:?}", sum);
}
