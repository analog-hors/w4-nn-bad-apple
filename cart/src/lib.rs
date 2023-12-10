use bad_apple::*;

const PALETTE: *mut [u32; 4] = 0x04 as _;
const FRAMEBUFFER: *mut [u8; 6400] = 0xa0 as _;
const BUFFER: *mut [u8; DECODER_BUFFER_SIZE] = 0xa0 as _;

static mut INDEX: usize = 0;

fn cubic([pn1, p0, p1, p2]: [f32; 4], t: f32) -> f32 {
    (2.0*t*t*t - 3.0*t*t + 1.0) * p0
        + (t*t*t - 2.0*t*t + t) * (p1 - pn1) / 2.0
        + (-2.0*t*t*t + 3.0*t*t) * p1
        + (t*t*t - t*t) * (p2 - p0) / 2.0
}

fn sample_indices(i: usize, target: usize, source: usize) -> [usize; 4] {
    let j = i * source / target;
    [
        j.saturating_sub(1),
        j,
        (j + 1).min(source - 1),
        (j + 2).min(source - 1),
    ]
}

fn resample(frame: &[u8; FRAME_WIDTH * FRAME_HEIGHT], fbx: usize, fby: usize) -> f32 {
    let x_samples = sample_indices(fbx, 160, FRAME_WIDTH);
    let y_samples = sample_indices(fby, 120, FRAME_HEIGHT);
    let samples = y_samples.map(|y| {
        let samples = x_samples.map(|x| {
            frame[y * FRAME_WIDTH + x] as f32 / u8::MAX as f32
        });
        cubic(samples, (fbx as f32 / 160.0 * FRAME_WIDTH as f32).fract())
    });
    cubic(samples, (fby as f32 / 120.0 * FRAME_HEIGHT as f32).fract())
}

#[no_mangle]
unsafe fn update() {
    let frame = get_frame(INDEX / 2, &mut *BUFFER);
    INDEX = (INDEX + 1) % (FRAME_COUNT * 2);

    *PALETTE = [0x000000, 0x555555, 0xAAAAAA, 0xFFFFFF];
    (*FRAMEBUFFER).fill(0);
    for fby in 0..120 {
        for fbx in 0..160 {
            let pixel = resample(&frame, fbx, fby);
            let pixel = if pixel < 0.25 {
                0
            } else if pixel < 0.50 {
                1
            } else if pixel < 0.75 {
                2
            } else {
                3
            };
            let index = (fby + 20) * 160 + fbx;
            (*FRAMEBUFFER)[index / 4] |= pixel << index % 4 * 2;
        }
    }
}
