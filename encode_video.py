from typing import Iterator
import random, torch
from PIL import Image

EPOCHS = 10_000
BATCH_SIZE = 256
FRAME_WIDTH = 32
FRAME_HEIGHT = 24
FRAME_STEP = 2
FRAME_COUNT = 6572
EMBEDDING_DIMS = 16
FRAME_QUANT_RANGE = 255.0
WEIGHT_CLIP_RANGE = 0.5
WEIGHT_QUANT_RANGE = 127.0
BIAS_CLIP_RANGE = 16.0
BIAS_QUANT_RANGE = 32767.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 0xd9e
torch.manual_seed(SEED)
random.seed(SEED)

def load_frames() -> list[list[float]]:
    frames = []
    for i in range(0, FRAME_COUNT, FRAME_STEP):
        with Image.open(f"frames/{i + 1}.png") as frame:
            assert frame.size == (FRAME_WIDTH, FRAME_HEIGHT)
            raw = frame.convert("L").tobytes()
            frames.append([b / 255 for b in raw])
    return frames

def make_dataset(frames: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = list(enumerate(frames))
    random.shuffle(dataset)
    indices = torch.tensor([i for i, _ in dataset], dtype=torch.long, device=DEVICE)
    indices = torch.nn.functional.one_hot(indices, FRAME_COUNT // FRAME_STEP).float()
    targets = torch.tensor([t for _, t in dataset], dtype=torch.float32, device=DEVICE)
    return indices, targets

def iter_dataset(inputs: torch.Tensor, targets: torch.Tensor) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    total = inputs.shape[0]
    index = 0
    while index + BATCH_SIZE <= total:
        yield inputs[index:index + BATCH_SIZE], targets[index:index + BATCH_SIZE]
        index += BATCH_SIZE
    if index < total:
        yield inputs[index:], targets[index:]

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(EMBEDDING_DIMS, 16 * (FRAME_HEIGHT - 3 - 15) * (FRAME_WIDTH - 3 - 15))
        self.l1 = torch.nn.ConvTranspose2d(16, 16, 4)
        self.l2 = torch.nn.ConvTranspose2d(16, 1, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l0(x)
        x = torch.nn.functional.mish(x)
        
        x = x.reshape((*x.shape[:-1], 16, FRAME_HEIGHT - 3 - 15, FRAME_WIDTH - 3 - 15))
        x = self.l1(x)
        x = torch.nn.functional.mish(x)
        x = self.l2(x)
        x = torch.nn.functional.sigmoid(x)

        x = x.reshape((*x.shape[:-3], FRAME_HEIGHT * FRAME_WIDTH))
        return x

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(FRAME_COUNT // FRAME_STEP, 512),
            torch.nn.Mish(),
            torch.nn.Linear(512, EMBEDDING_DIMS),
            torch.nn.Sigmoid(),
        )

        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

class Clipper:
    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(-WEIGHT_CLIP_RANGE, WEIGHT_CLIP_RANGE)
            module.weight.data = w
        if hasattr(module, "bias"):
            b = module.bias.data
            b = b.clamp(-BIAS_CLIP_RANGE, BIAS_CLIP_RANGE)
            module.bias.data = b

inputs, targets = make_dataset(load_frames())

model = AutoEncoder().to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fn = lambda o, t: torch.mean(torch.abs(o - t) ** 2.2)
clipper = Clipper()
print(f"total parameters: {sum(p.numel() for p in model.decoder.parameters())}")

for epoch in range(EPOCHS):
    epoch_loss = 0
    for (input, target) in iter_dataset(inputs, targets):
        outputs = model(input)
        loss = loss_fn(outputs, target)

        optim.zero_grad()
        loss.backward()
        optim.step()
        model.decoder.apply(clipper)

        epoch_loss += loss.item() * input.shape[0]
    print(f"[{epoch + 1}/{EPOCHS}] loss: {epoch_loss / inputs.shape[0]}", flush=True)
model.eval()

print(f"final loss: {loss_fn(model(inputs), targets).item()}")
print(f"total parameters: {sum(p.numel() for p in model.decoder.parameters())}")

with open("encoded_frames.bin", "wb+") as f:
    indices = torch.arange(0, FRAME_COUNT // FRAME_STEP, device=DEVICE)
    indices = torch.nn.functional.one_hot(indices, FRAME_COUNT // FRAME_STEP).float()
    encoded_frames = model.encoder(indices).cpu().detach().numpy().tolist()
    for frame in encoded_frames:
        for n in frame:
            scaled = round(n * FRAME_QUANT_RANGE)
            byte = min(max(scaled, 0), FRAME_QUANT_RANGE)
            f.write(bytes([byte]))

with open("decoder_nn.rs", "w+") as f:
    f.write(f"const FRAME_WIDTH: usize = {FRAME_WIDTH};\n")
    f.write(f"const FRAME_HEIGHT: usize = {FRAME_HEIGHT};\n")
    f.write(f"const EMBEDDING_DIMS: usize = {EMBEDDING_DIMS};\n")
    f.write(f"const FRAME_QUANT_RANGE: f32 = {FRAME_QUANT_RANGE};\n")
    f.write(f"const WEIGHT_CLIP_RANGE: f32 = {WEIGHT_CLIP_RANGE};\n")
    f.write(f"const WEIGHT_QUANT_RANGE: f32 = {WEIGHT_QUANT_RANGE};\n")
    f.write(f"const BIAS_CLIP_RANGE: f32 = {BIAS_CLIP_RANGE};\n")
    f.write(f"const BIAS_QUANT_RANGE: f32 = {BIAS_QUANT_RANGE};\n")

    def quantized_weight_str(tensor: torch.Tensor) -> str:
        if len(tensor.shape) == 0:
            n = round(tensor.item() / WEIGHT_CLIP_RANGE * WEIGHT_QUANT_RANGE)
            n = min(max(n, -WEIGHT_QUANT_RANGE), WEIGHT_QUANT_RANGE)
            return str(n)
        return f"[{','.join(quantized_weight_str(t) for t in tensor)}]"

    def quantized_bias_str(tensor: torch.Tensor) -> str:
        if len(tensor.shape) == 0:
            n = round(tensor.item() / BIAS_CLIP_RANGE * BIAS_QUANT_RANGE)
            n = min(max(n, -BIAS_QUANT_RANGE), BIAS_QUANT_RANGE)
            return str(n)
        return f"[{','.join(quantized_bias_str(t) for t in tensor)}]"

    def struct_str(name: str, fields: dict[str, str]) -> str:
        fields_str = ",".join(f"{k}:{v}" for k, v in fields.items())
        return f"{name}{{{fields_str}}}"

    def type_str(name: str, args: list[int]) -> str:
        return f"{name}<{','.join(str(a) for a in args)}>"

    def write_linear(name: str, l: torch.nn.Linear):
        struct = struct_str("Linear", {
            "weight": quantized_weight_str(l.weight),
            "bias": quantized_bias_str(l.bias),
        })
        type = type_str("Linear", [
            l.in_features,
            l.out_features,
        ])
        f.write(f"static {name}: {type} = {struct};\n")

    def write_convtrans2d(name: str, l: torch.nn.ConvTranspose2d):
        struct = struct_str("ConvTrans2d", {
            "weight": quantized_weight_str(l.weight),
            "bias": quantized_bias_str(l.bias),
        })
        type = type_str("ConvTrans2d", [
            l.in_channels,
            l.out_channels,
            l.kernel_size[0],
            l.kernel_size[1],
        ])
        f.write(f"static {name}: {type} = {struct};\n")

    decoder = model.decoder.cpu()
    write_linear("L0", decoder.l0)
    write_convtrans2d("L1", decoder.l1)
    write_convtrans2d("L2", decoder.l2)
