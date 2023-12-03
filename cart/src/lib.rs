const PALETTE: *mut [u32; 4] = 0x04 as _;
const FRAMEBUFFER: *mut [u8; 6400] = 0xa0 as _;

static mut INDEX: usize = 0;

#[no_mangle]
unsafe fn update() {
    let mut buffer = [0; bad_apple::DECODER_BUFFER_SIZE];
    let frame = bad_apple::get_frame(INDEX / 2, &mut buffer);
    INDEX = (INDEX + 1) % (bad_apple::FRAME_COUNT * 2);

    *PALETTE = [0x000000, 0x0D0D0D, 0xF2F2F2, 0xFFFFFF];
    for fby in 0..120 {
        for fbx in 0..160 {
            let y = fby * bad_apple::FRAME_HEIGHT / 120;
            let x = fbx * bad_apple::FRAME_WIDTH / 160;
            let pixel = frame[y * bad_apple::FRAME_WIDTH + x];

            let pixel = if pixel > 128 { 3 } else { 0 };
            let index = (fby + 20) * 160 + fbx;
            (*FRAMEBUFFER)[index / 4] |= pixel << index % 4 * 2;
        }
    }
}
