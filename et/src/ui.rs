use std::{borrow::Cow, time::Duration};

use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};

pub(crate) fn progress_bar(len: usize, message: impl Into<Cow<'static, str>>) -> ProgressBar {
    let message = message.into();
    let fmt = if !message.is_empty() {
        "{msg} {wide_bar} {pos:>8}/{len:>8} ETA: {eta_precise} Elapsed: {elapsed_precise}"
    } else {
        "{wide_bar} {pos:>8}/{len:>8} ETA: {eta_precise} Elapsed: {elapsed_precise}"
    };
    ProgressBar::new(len as u64)
        .with_style(ProgressStyle::default_bar().template(fmt).unwrap())
        .with_finish(ProgressFinish::AndLeave)
        .with_message(message)
}

pub(crate) fn progress_spinner(message: impl Into<Cow<'static, str>>) -> ProgressBar {
    let message = message.into();
    let fmt = if !message.is_empty() {
        "{spinner} {msg} {pos:>8} Elapsed: {elapsed_precise}"
    } else {
        "{spinner} {pos:>8} Elapsed: {elapsed_precise}"
    };
    let spinner = ProgressBar::new_spinner()
        .with_style(
            ProgressStyle::default_spinner()
                .tick_strings(&[
                    "▹▹▹▹▹",
                    "▸▹▹▹▹",
                    "▹▸▹▹▹",
                    "▹▹▸▹▹",
                    "▹▹▹▸▹",
                    "▹▹▹▹▸",
                    "▪▪▪▪▪",
                ])
                .template(fmt)
                .unwrap(),
        )
        .with_finish(ProgressFinish::AndLeave)
        .with_message(message);
    spinner.enable_steady_tick(Duration::from_millis(100));
    spinner
}
