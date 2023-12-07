use bytemuck::Pod;
use half::f16;

fn view<T: Pod, U: Pod>(t: &T) -> &U {
    bytemuck::cast_ref(t)
}

fn view_flat_mut<T: Pod, U: Pod>(t: &mut T) -> &mut [U] {
    bytemuck::cast_slice_mut(std::slice::from_mut(t))
}

fn activation<T: Pod>(f: impl Fn(f32) -> f32, t: &mut T) {
    for n in view_flat_mut::<_, f16>(t) {
        *n = f16::from_f32(f(n.to_f32()));
    }
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

fn apply_weight(w: i8, i: f16) -> f16 {
    let w = w as f32 / WEIGHT_QUANT_RANGE * WEIGHT_CLIP_RANGE;
    let i = f32::from(i);
    f16::from_f32(w * i)
}

fn get_bias(b: i8) -> f16 {
    let b = b as f32 / BIAS_QUANT_RANGE * BIAS_CLIP_RANGE;
    f16::from_f32(b)
}

pub struct Linear<const I: usize, const O: usize> {
    weight: [[i8; I]; O],
    bias: [i8; O],
}

impl<const I: usize, const O: usize> Linear<I, O> {
    fn forward(&self, input: &[f16; I], output: &mut [f16; O]) {
        for o in 0..O {
            output[o] = get_bias(self.bias[o]);
            for i in 0..I {
                output[o] += apply_weight(self.weight[o][i], input[i]);
            }
        }
    }
}

pub struct ConvTrans2d<
    const IC: usize,
    const OC: usize,
    const KH: usize,
    const KW: usize,
> {
    weight: [[[[i8; KW]; KH]; OC]; IC],
    bias: [i8; OC],
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
    >(&self, input: &[[[f16; IW]; IH]; IC], output: &mut [[[f16; OW]; OH]; OC]) {
        struct AssertOutputSize<const I: usize, const O: usize, const K: usize>;
        impl<const I: usize, const O: usize, const K: usize> AssertOutputSize<I, O, K> {
            const CORRECT: () = assert!(I + K - 1 == O);
        }
        let _ = AssertOutputSize::<IW, OW, KW>::CORRECT;
        let _ = AssertOutputSize::<IH, OH, KH>::CORRECT;

        for oc in 0..OC {
            for oy in 0..OH {
                for ox in 0..OW {
                    output[oc][oy][ox] = get_bias(self.bias[oc]);
                }
            }
        }
        for ic in 0..IC {
            for oc in 0..OC {
                for iy in 0..IH {
                    for ix in 0..IW {
                        for ky in 0..KH {
                            for kx in 0..KW {
                                output[oc][iy + ky][ix + kx] += apply_weight(
                                    self.weight[ic][oc][ky][kx],
                                    input[ic][iy][ix],
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

include!("../../decoder_nn.rs");

pub const DECODER_BUFFER_SIZE: usize = 2700 * std::mem::size_of::<f16>();

struct LayerBuffer<'b, I: Pod> {
    buffer: &'b mut [u8],
    _phantom: std::marker::PhantomData<I>,
}

impl<'b, I: Pod> LayerBuffer<'b, I> {
    fn new(buffer: &'b mut [u8], input: &I) -> Self {
        let input = bytemuck::cast_slice(std::slice::from_ref(input));
        buffer[..input.len()].copy_from_slice(input);
        Self { buffer, _phantom: std::marker::PhantomData }
    }

    fn layer<O: Pod>(self, f: impl FnOnce(&mut I, &mut O)) -> LayerBuffer<'b, O> {
        let input_size = std::mem::size_of::<I>();
        let output_size = std::mem::size_of::<O>();

        let (input, output) = self.buffer[..input_size + output_size].split_at_mut(input_size);
        f(&mut bytemuck::cast_slice_mut(input)[0], &mut bytemuck::cast_slice_mut(output)[0]);
        
        self.buffer.copy_within(input_size..input_size + output_size, 0);
        LayerBuffer { buffer: self.buffer, _phantom: std::marker::PhantomData }
    }

    fn finish(self) -> &'b I {
        let input_size = std::mem::size_of::<I>();
        let input = &mut self.buffer[..input_size];
        &bytemuck::cast_slice(input)[0]
    }
}

type L1Input = [[[f16; FRAME_WIDTH - 15 - 3]; FRAME_HEIGHT - 15 - 3]; 4];
type L1Output = [[[f16; FRAME_WIDTH - 15]; FRAME_HEIGHT - 15]; 4];
type L2Output = [[[f16; FRAME_WIDTH]; FRAME_HEIGHT]; 1];

pub fn decoder(mut input: [f16; EMBEDDING_DIMS], buffer: &mut [u8; DECODER_BUFFER_SIZE]) -> &[f16; FRAME_WIDTH * FRAME_HEIGHT] {
    activation(f32::tanh, &mut input);

    let output = LayerBuffer::new(buffer, &input)
        .layer(|input, output| {
            L0.forward(input, output);
            activation(mish, output);
        })
        .layer(|input, output: &mut L1Output| {
            let input: &L1Input = view(input);
            L1.forward(input, output);
            activation(mish, output);        
        })
        .layer(|input, output: &mut L2Output| {
            L2.forward(input, output);
            activation(sigmoid, output);
        })
        .finish();
    
    view(output)
}

pub fn decoder_size() -> usize {
    std::mem::size_of_val(&L0) + std::mem::size_of_val(&L1) + std::mem::size_of_val(&L2)
}
