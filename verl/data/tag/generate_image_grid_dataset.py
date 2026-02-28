import cv2
import argparse
import os
import math
from PIL import Image, ImageDraw, ImageFont


def get_font(size):
    """Tries to load a good font with robust fallbacks across OS."""
    font_candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "arial.ttf",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def format_timestamp(seconds):
    """Formats seconds into MM:SS.ddd"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def center_crop_to_square(image):
    """Center-crops an image to a square to preserve aspect ratio."""
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def add_timestamp_label(image, timestamp_sec, cell_size):
    """Burns the real timestamp into the top-left corner of a frame."""
    draw = ImageDraw.Draw(image)
    label = format_timestamp(timestamp_sec)

    font_size = max(12, cell_size // 10)
    font = get_font(font_size)

    padding = 3
    bbox = draw.textbbox((padding, padding), label, font=font)
    draw.rectangle(
        [bbox[0] - 2, bbox[1] - 2, bbox[2] + 4, bbox[3] + 4],
        fill=(0, 0, 0, 200),
    )
    draw.text((padding, padding), label, fill="white", font=font)
    return image


def process_video(video_path, output_dir, canvas_width=1024):
    """
    For each second of the video, extracts 12 frames (4 cols x 3 rows)
    and saves them as a single 4:3 grid image.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    rows, cols = 3, 4
    num_frames_per_grid = rows * cols  # 12
    cell_w = canvas_width // cols
    cell_h = cell_w * 3 // 4  # Each cell is also 4:3
    canvas_height = cell_h * rows

    total_seconds = int(math.ceil(duration))
    print(f"Video: {video_path}")
    print(f"Duration: {format_timestamp(duration)} ({duration:.2f}s) | FPS: {fps:.2f}")
    print(
        f"Generating {total_seconds} grid images ({cols}x{rows}, {num_frames_per_grid} frames each)..."
    )
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    for sec in range(total_seconds):
        # Calculate 12 evenly spaced timestamps within this 1-second window
        t_start = float(sec)
        t_end = min(float(sec + 1), duration)
        interval = (t_end - t_start) / num_frames_per_grid
        timestamps = [
            t_start + interval * i + interval / 2 for i in range(num_frames_per_grid)
        ]

        canvas = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))

        for i, t in enumerate(timestamps):
            t_clamped = max(0.0, min(t, duration - 0.01))
            frame_id = int(fps * t_clamped)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()

            if not ret:
                # If we can't read a frame, fill with a black placeholder
                cell = Image.new("RGB", (cell_w, cell_h), (30, 30, 30))
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cell = Image.fromarray(frame_rgb)
                cell = cell.resize((cell_w, cell_h), Image.Resampling.LANCZOS)

            cell = add_timestamp_label(cell, t, min(cell_w, cell_h))

            col = i % cols
            row = i // cols
            canvas.paste(cell, (col * cell_w, row * cell_h))

        out_path = os.path.join(output_dir, f"second_{sec:05d}.png")
        canvas.save(out_path)

        if (sec + 1) % 10 == 0 or sec == 0 or sec == total_seconds - 1:
            print(f"  [{sec + 1}/{total_seconds}] Saved {out_path}")

    cap.release()
    print(f"\nDone! {total_seconds} grid images saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert each second of a video into a 4x3 grid image (12 frames/sec)."
    )
    parser.add_argument(
        "--video", type=str, required=True, help="Path to the input video file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="grid_frames",
        help="Directory to save the per-second grid images (default: grid_frames/).",
    )
    parser.add_argument(
        "--canvas-width",
        type=int,
        default=1024,
        help="Canvas width in pixels; height is derived as 3/4 (default: 1024 → 1024x576).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
    else:
        process_video(args.video, args.output_dir, args.canvas_width)
