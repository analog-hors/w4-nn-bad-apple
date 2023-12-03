use bytemuck::Pod;

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

fn rescale_bias(b: i16) -> f32 {
    b as f32 / BIAS_QUANT_RANGE * BIAS_CLIP_RANGE
}

pub struct Linear<const I: usize, const O: usize> {
    weight: [[i8; I]; O],
    bias: [i16; O],
}

impl<const I: usize, const O: usize> Linear<I, O> {
    fn forward(&self, input: &[f32; I]) -> [f32; O] {
        let mut out = [0.0; O];
        for o in 0..O {
            for i in 0..I {
                out[o] += rescale_weight(self.weight[o][i]) * input[i];
            }
            out[o] += rescale_bias(self.bias[o]);
        }
        out
    }
}

pub struct ConvTrans2d<
    const IC: usize,
    const OC: usize,
    const KH: usize,
    const KW: usize,
> {
    weight: [[[[i8; KW]; KH]; OC]; IC],
    bias: [i16; OC],
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
            for oy in 0..OH {
                for ox in 0..OW {
                    out[oc][oy][ox] = rescale_bias(self.bias[oc]);
                }
            }
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

include!("../../decoder_nn.rs");

type L1Input = [[[f32; FRAME_WIDTH - 15 - 3]; FRAME_HEIGHT - 15 - 3]; 16];
type L1Output = [[[f32; FRAME_WIDTH - 15]; FRAME_HEIGHT - 15]; 16];
type L2Output = [[[f32; FRAME_WIDTH]; FRAME_HEIGHT]; 1];

pub fn decoder(input: [f32; EMBEDDING_DIMS]) -> [f32; FRAME_WIDTH * FRAME_HEIGHT] {
    let input = map(f32::tanh, input);
    let input = L0.forward(&input);
    let input = map(mish, input);
    let input: &L1Input = view(&input);
    let input: L1Output = L1.forward(input);
    let input = map(mish, input);
    let input: L2Output = L2.forward(&input);
    let input = map(sigmoid, input);
    *view(&input)
}
