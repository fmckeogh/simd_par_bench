use {rand::prelude::*, rayon::prelude::*, simd_par_bench::*};

const COUNT: usize = 1000000;

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
