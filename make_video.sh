cargo run --release
ffmpeg -framerate 30 -start_number 1 -i "decoded/%d.png" -i bad_apple.ogg bad_apple_decoded.webm -y
