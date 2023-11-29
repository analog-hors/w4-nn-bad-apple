import random, json, torch, torch.nn
from PIL import Image

EPOCHS = 15_000
FRAME_WIDTH = 20
FRAME_HEIGHT = 20
FRAME_NUMS = 8
WEIGHT_CLIP_RANGE = 0.5
WEIGHT_QUANT_RANGE = 127
FRAME_QUANT_RANGE = 255

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 0xd9e
torch.manual_seed(SEED)
random.seed(SEED)

def load_frames() -> list[list[float]]:
    frames = []
    while True:
        try:
            path = f"frames/{len(frames) + 1}.png"
            with Image.open(path) as frame:
                assert frame.size == (FRAME_WIDTH, FRAME_HEIGHT)
                raw = frame.convert("L").tobytes()
                frames.append([b / 255 for b in raw])
        except FileNotFoundError:
            break
    return frames

def make_dataset(frames: list[list[float]]) -> torch.Tensor:
    dataset = frames[:]
    random.shuffle(dataset)
    return torch.tensor(dataset, dtype=torch.float32, device=DEVICE)

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(FRAME_WIDTH * FRAME_HEIGHT, 1024),
            torch.nn.Mish(),
            torch.nn.Linear(1024, 512),
            torch.nn.Mish(),
            torch.nn.Linear(512, FRAME_NUMS),
            torch.nn.Sigmoid(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(FRAME_NUMS, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128, 64),
            torch.nn.Mish(),
            torch.nn.Linear(64, FRAME_WIDTH * FRAME_HEIGHT),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class Clipper:
    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(-WEIGHT_CLIP_RANGE, WEIGHT_CLIP_RANGE)
            module.weight.data = w

frames = load_frames()
dataset = make_dataset(frames)

model = Encoder().to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fn = lambda o, t: torch.mean(torch.abs(o - t) ** 2.2)
clipper = Clipper()

for epoch in range(EPOCHS):
    outputs = model(dataset)
    loss = loss_fn(outputs, dataset)

    optim.zero_grad()
    loss.backward()
    optim.step()
    model.decoder.apply(clipper)

    print(f"[{epoch + 1}/{EPOCHS}] loss: {loss.item()}")
print(f"total parameters: {sum(p.numel() for p in model.decoder.parameters())}")

with open("encoded_frames.bin", "wb+") as f:
    frames_tensor = torch.tensor(frames, dtype=torch.float32, device=DEVICE)
    encoded_frames = model.encoder(frames_tensor).cpu().detach().numpy().tolist()
    for frame in encoded_frames:
        for n in frame:
            scaled = round(n * FRAME_QUANT_RANGE)
            byte = min(max(scaled, 0), FRAME_QUANT_RANGE)
            f.write(bytes([byte]))

def quantize_weights(tensor: torch.Tensor):
    if len(tensor.shape) == 0:
        n = round(tensor.item() / WEIGHT_CLIP_RANGE * WEIGHT_QUANT_RANGE)
        return min(max(n, -WEIGHT_QUANT_RANGE), WEIGHT_QUANT_RANGE)
    return [quantize_weights(t) for t in tensor]

def tensor_type(tensor: torch.Tensor, num_type: str) -> str:
    tensor_type = num_type
    for size in reversed(tensor.shape):
        tensor_type = f"[{tensor_type}; {size}]"
    return tensor_type

with open("decoder_nn.rs", "w+") as f:
    f.write(f"const FRAME_WIDTH: usize = {FRAME_WIDTH};\n")
    f.write(f"const FRAME_HEIGHT: usize = {FRAME_HEIGHT};\n")
    f.write(f"const FRAME_NUMS: usize = {FRAME_NUMS};\n")
    for name, tensor in model.decoder.cpu().state_dict().items():
        index, kind = name.split(".")
        if kind == "weight":
            type = tensor_type(tensor, "i8")
            quantized = quantize_weights(tensor)
            f.write(f"static WEIGHT_{index}: {type} = {json.dumps(quantized)};\n")
        else:
            type = tensor_type(tensor, "f32")
            f.write(f"static BIAS_{index}: {type} = {json.dumps(tensor.numpy().tolist())};\n")
