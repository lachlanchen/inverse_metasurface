# Create a file list for ffmpeg using the images from epoch 1 to 281
with open("images.txt", "w") as f:
    for epoch in range(1, 282):  # 1 to 281 inclusive
        file_name = f"filter_epoch_{epoch}_with_shape.png"
        f.write(f"file '{file_name}'\n")

