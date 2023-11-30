include!("../decoder_nn.rs");
static FRAMES: &[u8] = include_bytes!("../encoded_frames.bin");
const KEYFRAME_INTERVAL: usize = 1;

fn linear<
    const I: usize,
    const O: usize,
>(
    input: &[f32; I],
    weight: &[[i8; I]; O],
    bias: &[f32; O]
) -> [f32; O] {
    let mut out = bias.clone();
    for (weight, n) in weight.iter().zip(&mut out) {
        for (w, i) in weight.iter().zip(input) {
            let w = *w as f32 / 127.0 * 0.5;
            *n += w * i;
        }
    }
    out
}

fn convtrans2d<
    const IC: usize,
    const OC: usize,
    const IH: usize,
    const IW: usize,
    const OH: usize,
    const OW: usize,
    const K: usize,
>(
    input: &[[[f32; IW]; IH]; IC],
    weight: &[[[[i8; K]; K]; OC]; IC],
    bias: &[f32; OC]
) -> [[[f32; OW]; OH]; OC] {
    struct AssertOutputSize<const I: usize, const O: usize, const K: usize>;
    impl<const I: usize, const O: usize, const K: usize> AssertOutputSize<I, O, K> {
        const CORRECT: () = assert!(I + K - 1 == O);
    }
    let _ = AssertOutputSize::<IW, OW, K>::CORRECT;
    let _ = AssertOutputSize::<IH, OH, K>::CORRECT;

    let mut out = [[[0.0; OW]; OH]; OC];
    for oc in 0..OC {
        out[oc] = [[bias[oc]; OW]; OH];
    }
    for ic in 0..IC {
        for oc in 0..OC {
            for iy in 0..IH {
                for ix in 0..IW {
                    for ky in 0..K {
                        for kx in 0..K {
                            out[oc][iy + ky][ix + kx] += weight[ic][oc][ky][kx] as f32 / 127.0 * 0.5 * input[ic][iy][ix];
                        }
                    }
                }
            }
        }
    }

    out
}

fn softplus(n: f32) -> f32 {
    (1.0 + n.exp()).ln()
}

fn mish(n: f32) -> f32 {
    n * softplus(n).tanh()
}

fn sigmoid(n: f32) -> f32 {
    1.0 / (1.0 + (-n).exp())
}

fn activation<T: bytemuck::Pod>(f: impl Fn(f32) -> f32, v: T) -> T {
    let mut v = [v];
    for n in bytemuck::cast_slice_mut::<_, f32>(&mut v) {
        *n = f(*n);
    }
    v[0]
}

fn decode(input: &[f32; FRAME_NUMS]) -> [u8; FRAME_WIDTH * FRAME_HEIGHT] {
    let input = linear(&input, &L0_WEIGHT, &L0_BIAS);
    let input = activation(mish, input);
    let input = convtrans2d::<
        8,
        16,
        {FRAME_HEIGHT - 15 - 3},
        {FRAME_WIDTH - 15 - 3},
        {FRAME_HEIGHT - 15},
        {FRAME_WIDTH - 15},
        4,
    >(bytemuck::cast_ref(&input), &L1_WEIGHT, &L1_BIAS);
    let input = activation(mish, input);
    let input = convtrans2d::<
        16,
        1,
        {FRAME_HEIGHT - 15},
        {FRAME_WIDTH - 15},
        {FRAME_HEIGHT},
        {FRAME_WIDTH},
        16,
    >(&input, &L2_WEIGHT, &L2_BIAS);
    let input: [f32; FRAME_WIDTH * FRAME_HEIGHT] = bytemuck::cast(input);
    input.map(|n| ((sigmoid(n) * 255.0).round() as i32).clamp(0, u8::MAX as i32) as u8)
}

static FRAME_COUNT: usize = FRAMES.len() / FRAME_NUMS;

fn get_frame(i: usize) -> [f32; FRAME_NUMS] {
    let i = i.clamp(0, FRAME_COUNT - 1);
    let frame = &FRAMES[i * FRAME_NUMS..i * FRAME_NUMS + FRAME_NUMS];
    let frame: [u8; FRAME_NUMS] = frame.try_into().unwrap();
    frame.map(|n| n as f32 / 255.0)
}

fn main() {
    for i in 0..FRAME_COUNT {
        let frame = match i % KEYFRAME_INTERVAL {
            0 => get_frame(i),
            progress => {
                let mut frame = get_frame(i - progress);
                let target = get_frame(i - progress + KEYFRAME_INTERVAL);
                for (f, t) in frame.iter_mut().zip(target) {
                    let p = progress as f32 / KEYFRAME_INTERVAL as f32;
                    *f = *f * (1.0 - p) + t * p;
                }
                frame
            }
        };

        let frame = decode(&frame);
        let image = image::GrayImage::from_raw(FRAME_WIDTH as u32, FRAME_HEIGHT as u32, frame.to_vec()).unwrap();
        image.save(format!("decoded/{}.png", i + 1)).unwrap();
    }
}
