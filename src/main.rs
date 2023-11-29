include!("../decoder_nn.rs");
static FRAMES: &[u8] = include_bytes!("../encoded_frames.bin");

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

fn decode(input: &[u8; FRAME_NUMS]) -> [u8; 20 * 20] {
    let input = input.map(|n| n as f32 / 255.0);
    let input = linear(&input, &WEIGHT_0, &BIAS_0);
    let input = input.map(mish);
    let input = linear(&input, &WEIGHT_2, &BIAS_2);
    let input = input.map(mish);
    let input = linear(&input, &WEIGHT_4, &BIAS_4);
    input.map(|n| ((sigmoid(n) * 255.0).round() as i32).clamp(0, u8::MAX as i32) as u8)
}

fn main() {
    for (i, frame) in FRAMES.chunks_exact(FRAME_NUMS).enumerate() {
        let frame = frame.try_into().unwrap();
        let frame = decode(frame);
        let image = image::GrayImage::from_raw(20, 20, frame.to_vec()).unwrap();
        image.save(format!("decoded/{}.png", i + 1)).unwrap();
    }
}
