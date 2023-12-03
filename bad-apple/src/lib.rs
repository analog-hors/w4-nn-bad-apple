mod decoder;

use decoder::*;
pub use decoder::{FRAME_WIDTH, FRAME_HEIGHT};

static KEYFRAME_DATA: &[u8] = include_bytes!("../../encoded_frames.bin");

pub static FRAME_COUNT: usize = (KEYFRAME_DATA.len() / EMBEDDING_DIMS - 1) * KEYFRAME_INTERVAL;

fn get_keyframe_embedding(i: usize) -> [f32; EMBEDDING_DIMS] {
    let keyframes: &[[i8; EMBEDDING_DIMS]] = bytemuck::cast_slice(KEYFRAME_DATA);
    keyframes[i].map(|n| n as f32 / FRAME_QUANT_RANGE * FRAME_CLIP_RANGE)
}

fn get_frame_embedding(i: usize) -> [f32; EMBEDDING_DIMS] {
    match i % KEYFRAME_INTERVAL {
        0 => get_keyframe_embedding(i / KEYFRAME_INTERVAL),
        p => {
            let mut frame = get_keyframe_embedding(i / KEYFRAME_INTERVAL);
            let target = get_keyframe_embedding(i / KEYFRAME_INTERVAL + 1);
            let p = p as f32 / KEYFRAME_INTERVAL as f32;
            for (f, t) in frame.iter_mut().zip(target) {
                *f = *f * (1.0 - p) + t * p;
            }
            frame
        }
    }
}

pub fn get_frame(i: usize) -> [u8; FRAME_WIDTH * FRAME_HEIGHT] {
    decoder(get_frame_embedding(i)).map(|n| (n * u8::MAX as f32).round() as u8)
}
