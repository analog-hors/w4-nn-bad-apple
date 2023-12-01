use bytemuck::Pod;

include!("../decoder_nn.rs");
static FRAMES: &[u8] = include_bytes!("../encoded_frames.bin");
const KEYFRAME_INTERVAL: usize = 1;

fn view<T: Pod, U: Pod>(t: &T) -> &U {
    bytemuck::cast_ref(t)
}

fn view_flat_mut<T: Pod, U: Pod>(t: &mut T) -> &mut [U] {
    bytemuck::cast_slice_mut(std::slice::from_mut(t))
}

fn map<T: Pod>(f: impl Fn(f32) -> f32, mut t: T) -> T {
    for n in view_flat_mut(&mut t) {
        *n = f(*n);
    }
    t
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

fn rescale_weight(w: i8) -> f32 {
    w as f32 / WEIGHT_QUANT_RANGE * WEIGHT_CLIP_RANGE
}

struct Linear<const I: usize, const O: usize> {
    weight: [[i8; I]; O],
    bias: [f32; O],
}

impl<const I: usize, const O: usize> Linear<I, O> {
    fn forward(&self, input: &[f32; I]) -> [f32; O] {
        let mut out = self.bias.clone();
        for (weight, n) in self.weight.iter().zip(&mut out) {
            for (&w, i) in weight.iter().zip(input) {
                *n += rescale_weight(w) * i;
            }
        }
        out
    }
}

struct ConvTrans2d<
    const IC: usize,
    const OC: usize,
    const KH: usize,
    const KW: usize,
> {
    weight: [[[[i8; KW]; KH]; OC]; IC],
    bias: [f32; OC],
}

impl<
    const IC: usize,
    const OC: usize,
    const KH: usize,
    const KW: usize,
> ConvTrans2d<IC, OC, KH, KW> {
    fn forward<
        const IH: usize,
        const IW: usize,
        const OH: usize,
        const OW: usize,
    >(&self, input: &[[[f32; IW]; IH]; IC]) -> [[[f32; OW]; OH]; OC] {
        struct AssertOutputSize<const I: usize, const O: usize, const K: usize>;
        impl<const I: usize, const O: usize, const K: usize> AssertOutputSize<I, O, K> {
            const CORRECT: () = assert!(I + K - 1 == O);
        }
        let _ = AssertOutputSize::<IW, OW, KW>::CORRECT;
        let _ = AssertOutputSize::<IH, OH, KH>::CORRECT;

        let mut out = [[[0.0; OW]; OH]; OC];
        for oc in 0..OC {
            out[oc] = [[self.bias[oc]; OW]; OH];
        }
        for ic in 0..IC {
            for oc in 0..OC {
                for iy in 0..IH {
                    for ix in 0..IW {
                        for ky in 0..KH {
                            for kx in 0..KW {
                                out[oc][iy + ky][ix + kx] += rescale_weight(self.weight[ic][oc][ky][kx]) * input[ic][iy][ix];
                            }
                        }
                    }
                }
            }
        }

        out
    }
}

type L1Input = [[[f32; FRAME_WIDTH - 15 - 3]; FRAME_HEIGHT - 15 - 3]; 16];
type L1Output = [[[f32; FRAME_WIDTH - 15]; FRAME_HEIGHT - 15]; 16];
type L2Output = [[[f32; FRAME_WIDTH]; FRAME_HEIGHT]; 1];
type NnOutput = [f32; FRAME_WIDTH * FRAME_HEIGHT];

fn decode(input: &[f32; FRAME_NUMS]) -> [u8; FRAME_WIDTH * FRAME_HEIGHT] {
    let input = L0.forward(input);
    let input = map(mish, input);
    let input: &L1Input = view(&input);
    let input: L1Output = L1.forward(input);
    let input = map(mish, input);
    let input: L2Output = L2.forward(&input);
    let input = map(sigmoid, input);
    let input: NnOutput = *view(&input);
    input.map(|n| ((n * 255.0).round() as i32).clamp(0, u8::MAX as i32) as u8)
}

static FRAME_COUNT: usize = FRAMES.len() / FRAME_NUMS;

fn get_frame(i: usize) -> [f32; FRAME_NUMS] {
    let i = i.clamp(0, FRAME_COUNT - 1);
    let frame = &FRAMES[i * FRAME_NUMS..i * FRAME_NUMS + FRAME_NUMS];
    let frame: [u8; FRAME_NUMS] = frame.try_into().unwrap();
    frame.map(|n| n as f32 / FRAME_QUANT_RANGE)
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
