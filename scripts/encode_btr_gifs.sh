#!/bin/bash
# mp4 (scripts/record_btr.py) -> GIF under GitHub's 10 MB inline limit.
cd "$(dirname "$0")/.."
for t in ds1 ds2 ds3; do
  ffmpeg -y -loglevel error -i videos/btr/btr_$t.mp4 -vf "fps=10,scale=320:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" videos/btr/btr_$t.gif
done
ls -la videos/btr/*.gif
