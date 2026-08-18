use std::io;

use clap::Args;
use easy_tiger::input::VectorStore;
use half::slice::HalfFloatSliceExt;
use indicatif::ProgressIterator;
use vectors::f16;

use crate::ui::progress_bar;

#[derive(Args)]
pub struct MrlArgs {
    /// Target dimensionalities to measure. Each must be less than the dimensionality
    /// of the doc vectors.
    #[arg(short, long, value_delimiter = ',')]
    dimensionalities: Vec<usize>,
}

pub fn mrl(
    args: MrlArgs,
    vectors: &(impl VectorStore<Elem = f16> + Send + Sync),
) -> io::Result<()> {
    let dim = vectors.elem_stride();
    for &d in args.dimensionalities.iter() {
        if d > dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("dimensionality {d} must be less than the vector dimensionality {dim}"),
            ));
        }
    }

    let mut dims = args.dimensionalities.clone();
    dims.sort();

    let mut partial_norms = vec![0.0; dims.len()];
    let mut sums = vec![0.0; dims.len()];
    let mut widened = vec![0.0f32; dim];
    for v in vectors
        .iter()
        .progress_with(progress_bar(vectors.len(), "process"))
    {
        partial_norms.fill(0.0);
        v.convert_to_f32_slice(&mut widened);
        for (i, &d) in dims.iter().enumerate() {
            for &x in widened.iter().take(d) {
                partial_norms[i] = x.mul_add(x, partial_norms[i]);
            }
        }

        for (&p, s) in partial_norms.iter().zip(sums.iter_mut()) {
            *s += p.sqrt() as f64;
        }
    }

    for (&d, &s) in dims.iter().zip(sums.iter()) {
        println!("{d:4} {:.6}", s / vectors.len() as f64);
    }

    Ok(())
}
