use std::num::NonZero;

use quantization::binary_quantize;
use wt_mdb::{RecordView, Session};

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
    // XXX maybe I can make Record contain a Cow?
    // you could have RecordView<'a> and type Record = RecordView<'static>
    // * RecordView::from_slice(s: &'a [u8]) -> RecordView<'a>
    // * RecordView::from_vec(v: Vec<u8>) -> Record
    session.bulk_load(
        table_name,
        None,
        vectors.enumerate().map(|(i, v)| {
            let b = binary_quantize(v);
            RecordView::new(i as i64, &b)
        }),
    )
}
