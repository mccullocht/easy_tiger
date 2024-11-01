use quantization::binary_quantize;
use wt_mdb::{Record, Session};

mod input;
mod quantization;

pub fn binary_quantize_and_load<'a, V>(
    session: &Session,
    table_name: &str,
    vectors: V,
) -> wt_mdb::Result<()>
where
    V: Iterator<Item = &'a [f32]>,
{
    session.bulk_load(
        table_name,
        None,
        vectors
            .enumerate()
            .map(|(i, v)| Record::new(i as i64, binary_quantize(v))),
    )
}
