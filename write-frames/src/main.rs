fn main() {
    println!("decoder size: {}", bad_apple::decoder_size());
    let mut buffer = [0; bad_apple::DECODER_BUFFER_SIZE];
    for i in 0..bad_apple::FRAME_COUNT {
        let image = image::GrayImage::from_raw(
            bad_apple::FRAME_WIDTH as u32,
            bad_apple::FRAME_HEIGHT as u32,
            bad_apple::get_frame(i, &mut buffer).to_vec(),
        ).unwrap();
        image.save(format!("decoded/{}.png", i + 1)).unwrap();
    }
}
