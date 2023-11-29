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

fn softplus(n: f32) -> f32 {
    (1.0 + n.exp()).ln()
}

fn mish(n: f32) -> f32 {
    n * softplus(n).tanh()
}

fn sigmoid(n: f32) -> f32 {
    1.0 / (1.0 + (-n).exp())
}

fn decode(input: &[f32; FRAME_NUMS]) -> [u8; 20 * 20] {
    let input = linear(&input, &WEIGHT_0, &BIAS_0);
    let input = input.map(mish);
    let input = linear(&input, &WEIGHT_2, &BIAS_2);
    let input = input.map(mish);
    let input = linear(&input, &WEIGHT_4, &BIAS_4);
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
        let image = image::GrayImage::from_raw(20, 20, frame.to_vec()).unwrap();
        image.save(format!("decoded/{}.png", i + 1)).unwrap();
    }
}
