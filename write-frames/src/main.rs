fn main() {
    for i in 0..bad_apple::FRAME_COUNT {
        let image = image::GrayImage::from_raw(
            bad_apple::FRAME_WIDTH as u32,
            bad_apple::FRAME_HEIGHT as u32,
            bad_apple::get_frame(i).to_vec(),
        ).unwrap();
        image.save(format!("decoded/{}.png", i + 1)).unwrap();
    }
}
