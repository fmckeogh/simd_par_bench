use {
    arrayfire::*,
    ndarray::{Array3, Zip},
    rand::prelude::*,
    std::time::SystemTime,
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

fn main() {
    for &(ng, nz) in &[
        (64, 16),
        (128, 32),
        (256, 64),
        (512, 128),
        (512, 256),
        (1024, 80),
    ] {
        println!("ng: {}, nz: {}", ng, nz);

        let mut a = gen_array(ng, nz);
        let b = gen_array(ng, nz);
        let c = gen_array(ng, nz);

        {
            let start = SystemTime::now();

            /*Zip::from(&mut a)
            .and(&b)
            .and(&c)
            .apply(|a, b, c| *a += b + c);*/
            Zip::from(&mut a)
                .and(&b)
                .and(&c)
                .apply(|a, b, c| *a *= b * c);

            let end = SystemTime::now();
            println!(
                "ndarray took {}us",
                end.duration_since(start).unwrap().as_micros()
            );
        }

        {
            let start = SystemTime::now();

            /*Zip::from(&mut a)
            .and(&b)
            .and(&c)
            .par_apply(|a, b, c| *a += b + c);*/
            Zip::from(&mut a)
                .and(&b)
                .and(&c)
                .par_apply(|a, b, c| *a *= b * c);

            let end = SystemTime::now();
            println!(
                "parallel ndarray took {}us",
                end.duration_since(start).unwrap().as_micros()
            );
        }

        {
            let start = SystemTime::now();

            let dims = Dim4::new(&[ng as u64, ng as u64, nz as u64, 1]);
            let mut a_af = Array::<f32>::new(a.as_slice_memory_order().unwrap(), dims);
            let b_af = Array::<f32>::new(b.as_slice_memory_order().unwrap(), dims);
            let c_af = Array::<f32>::new(c.as_slice_memory_order().unwrap(), dims);

            a_af *= b_af * c_af;

            a_af.host(a.as_slice_memory_order_mut().unwrap());

            let end = SystemTime::now();
            println!(
                "ArrayFire took {}us",
                end.duration_since(start).unwrap().as_micros()
            );
        }

        println!();
    }
}

fn gen_array(ng: usize, nz: usize) -> Array3<f32> {
    let mut rng = rand::thread_rng();

    let mut data = Vec::with_capacity(ng * ng * nz);

    for _ in 0..ng * ng * nz {
        data.push(rng.gen());
    }

    Array3::from_shape_vec((ng, ng, nz), data).unwrap()
}
