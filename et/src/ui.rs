use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};

pub(crate) fn progress_bar(len: usize, message: Option<&'static str>) -> ProgressBar {
    if let Some(message) = message {
        ProgressBar::new(len as u64)
            .with_style(
                ProgressStyle::default_bar()
                    .template(
                        "{msg} {wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}",
                    )
                    .unwrap(),
            )
            .with_message(message)
    } else {
        ProgressBar::new(len as u64).with_style(
            ProgressStyle::default_bar()
                .template("{wide_bar} {pos}/{len} ETA: {eta_precise} Elapsed: {elapsed_precise}")
                .unwrap(),
        )
    }
    .with_finish(ProgressFinish::AndLeave)
}

pub(crate) fn progress_spinner() -> ProgressBar {
    ProgressBar::new_spinner()
        .with_style(
            ProgressStyle::default_spinner()
                .template("{wide_bar} {pos} Elapsed: {elapsed_precise}")
                .unwrap(),
        )
        .with_finish(ProgressFinish::AndLeave)
}
