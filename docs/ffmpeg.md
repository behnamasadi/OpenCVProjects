# FFmpeg Command Reference

A practical cheat-sheet for FFmpeg, ffprobe, and ffplay. Commands are grouped
by task. `INPUT`/`OUTPUT` are placeholders for real file names.

FFmpeg's pipeline is always: **demux → decode → filter → encode → mux**. Most
options below are just knobs on one of those stages.

## Available encoders and decoders

FFmpeg ships hundreds of codecs. A *codec* is the compression format (e.g.
H.264); an *encoder*/*decoder* is a specific implementation of it (e.g.
`libx264`, `h264_nvenc`).

```bash
# List every encoder (video V, audio A, subtitle S)
ffmpeg -encoders

# List every decoder
ffmpeg -decoders

# List codecs and whether they can be encoded/decoded (D.....=decode, .E....=encode)
ffmpeg -codecs

# List container formats that can be muxed/demuxed
ffmpeg -formats

# Show help for one specific encoder (its private options)
ffmpeg -h encoder=libx264
ffmpeg -h decoder=h264

# Filter the list with grep
ffmpeg -encoders | grep -i h264
ffmpeg -decoders | grep -i hevc
```

In `ffmpeg -codecs` output the leading flags read as: `D` decode, `E` encode,
`V/A/S` type, `I` intra-frame-only, `L` lossy, `S` lossless.

## 1. FFmpeg Common Options

```bash
# General form: global opts -> input opts -> -i INPUT -> output opts -> OUTPUT
ffmpeg [global] [input opts] -i INPUT [output opts] OUTPUT
```

| Option        | Meaning                                                        |
|---------------|----------------------------------------------------------------|
| `-i INPUT`    | Input file (repeatable for multiple inputs)                    |
| `-c:v CODEC`  | Video codec/encoder (`-vcodec` alias); `copy` = remux, no re-encode |
| `-c:a CODEC`  | Audio codec/encoder (`-acodec` alias); `copy` = passthrough    |
| `-b:v 4M`     | Target video bitrate (`-b:a` for audio)                        |
| `-crf 23`     | Constant Rate Factor (x264/x265 quality; lower = better)       |
| `-r 30`       | Output frame rate (fps)                                        |
| `-s 1280x720` | Output frame size (shorthand for scale)                        |
| `-ss 00:00:10`| Seek to start position (before `-i` = fast/keyframe seek)      |
| `-t 30`       | Limit duration to 30 seconds                                   |
| `-to 00:01:00`| Stop at this timestamp                                         |
| `-vf`         | Video filtergraph                                              |
| `-af`         | Audio filtergraph                                              |
| `-y` / `-n`   | Overwrite output without asking / never overwrite             |
| `-an` / `-vn` | Drop audio / drop video                                        |
| `-sn`         | Drop subtitles                                                 |

```bash
# Re-encode to H.264 + AAC
ffmpeg -i input.mov -c:v libx264 -crf 23 -c:a aac -b:a 128k output.mp4

# Remux (change container only, no quality loss, very fast)
ffmpeg -i input.mkv -c copy output.mp4

# Trim 30s starting at 00:01:00 (fast seek before -i)
ffmpeg -ss 00:01:00 -i input.mp4 -t 30 -c copy clip.mp4

# Extract audio only / strip audio
ffmpeg -i input.mp4 -vn -c:a copy audio.aac
ffmpeg -i input.mp4 -an -c:v copy noaudio.mp4

# Resize and change frame rate, overwrite existing output
ffmpeg -y -i input.mp4 -s 1280x720 -r 25 output.mp4
```

## 2. FFmpeg Filters

Filters transform frames between decode and encode. Use `-vf` for a simple
video chain, `-af` for audio, and `-filter_complex` for graphs with multiple
inputs/outputs.

### 2.1 Available Filters

```bash
# List all filters (T=timeline, S=slice threading, C=command support)
ffmpeg -filters

# Only video filters
ffmpeg -filters | grep -i video

# Help / options for a specific filter
ffmpeg -h filter=scale
ffmpeg -h filter=drawtext
```

### 2.2 Send Output of FFmpeg Directly to FFplay

Pipe raw or muxed frames from ffmpeg into ffplay to preview a filter chain
without writing a file.

```bash
# Encode to a stream on stdout (-) and view it live
ffmpeg -i input.mp4 -vf "scale=640:-1" -f matroska - | ffplay -

# Preview raw video (must tell ffplay the format/size/rate)
ffmpeg -i input.mp4 -vf hflip -f rawvideo -pix_fmt rgb24 - \
  | ffplay -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -

# ffplay can also apply filters itself for quick previews
ffplay -i input.mp4 -vf "transpose=1,scale=480:-1"
```

### 2.3 Apply a Filter

```bash
# Scale to 640px wide, keep aspect ratio (-1 = auto, -2 = auto even number)
ffmpeg -i input.mp4 -vf "scale=640:-2" output.mp4

# Crop a 640x480 region starting at x=100,y=50
ffmpeg -i input.mp4 -vf "crop=640:480:100:50" cropped.mp4

# Rotate 90 degrees clockwise
ffmpeg -i input.mp4 -vf "transpose=1" rotated.mp4

# Chain filters with commas (linear order)
ffmpeg -i input.mp4 -vf "crop=640:480:0:0,scale=320:240,hflip" out.mp4
```

### 2.4 Filter Graph

A **filtergraph** passed via `-vf`/`-af` is a single linear chain: one input
pad, filters separated by `,`, one output pad. Semicolons `;` separate named
chains even within `-vf`.

```bash
# Linear chain: decode -> crop -> scale -> encode
ffmpeg -i input.mp4 -vf "crop=in_w:in_h-100:0:50,scale=1280:-2" output.mp4

# Named links with [labels] and ; between sub-chains
ffmpeg -i input.mp4 -vf "split[a][b];[a]hflip[left];[b]negate[right]" out.mp4
```

Syntax rules:
- `,` connects filters into a chain (output of one feeds the next).
- `;` separates independent chains.
- `[label]` names a pad so it can be referenced elsewhere in the graph.

### 2.5 Filter Complex

`-filter_complex` is required when a graph has **multiple inputs or multiple
outputs**. Input pads are referenced as `[0:v]`, `[1:a]` (input index : stream
type), and outputs are captured with named labels then routed via `-map`.

```bash
# Overlay a logo (input 1) onto a video (input 0) at 10,10
ffmpeg -i video.mp4 -i logo.png \
  -filter_complex "[0:v][1:v]overlay=10:10[outv]" \
  -map "[outv]" -map 0:a -c:a copy output.mp4

# Concatenate two clips (video+audio) into one
ffmpeg -i a.mp4 -i b.mp4 \
  -filter_complex "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" output.mp4

# Side-by-side (hstack) of two inputs
ffmpeg -i left.mp4 -i right.mp4 \
  -filter_complex "[0:v][1:v]hstack=inputs=2[out]" \
  -map "[out]" stacked.mp4
```

### 2.6 Filter Complex vs Filter Graph

| Aspect            | `-vf` / `-af` (simple filtergraph) | `-filter_complex`                 |
|-------------------|------------------------------------|-----------------------------------|
| Inputs            | Exactly one                        | One or more (`[0:v]`, `[1:v]`...) |
| Outputs           | Exactly one                        | One or more (named labels)        |
| Typical use       | scale, crop, fps, single-stream    | overlay, concat, stack, split/mix |
| Output routing    | Implicit                           | Explicit via `-map "[label]"`     |

Rule of thumb: if you need to combine two sources (overlay, concat, mix) or
split one source into several outputs, use `-filter_complex`. Otherwise `-vf`
is simpler.

### 2.7 Setting Number of Threads

```bash
# Codec-level decode/encode threads (0 = auto/all cores)
ffmpeg -threads 4 -i input.mp4 -c:v libx264 -threads 8 output.mp4

# Threads dedicated to the filtergraph
ffmpeg -i input.mp4 -filter_threads 4 -vf "scale=1280:-2" output.mp4

# Slice threading inside a single filter that supports it
ffmpeg -i input.mp4 -filter_complex "scale=1280:-2:threads=4" output.mp4

# x264 as an encoder option
ffmpeg -i input.mp4 -c:v libx264 -x264-params "threads=8" output.mp4
```

Note `-threads` placed **before `-i`** affects decoding; placed **after** it
affects the encoder. `0` lets FFmpeg pick based on CPU count.

### 2.8 Commonly Used FFmpeg Filters

```bash
# scale — resize (-1 keep AR, -2 keep AR and force even dimension)
-vf "scale=1280:720"          # exact
-vf "scale=1280:-2"           # width fixed, height auto-even

# crop=w:h:x:y — cut a region
-vf "crop=640:480:100:50"

# fps — change/normalize frame rate
-vf "fps=30"

# transpose — rotate (0=90CCW+vflip,1=90CW,2=90CCW,3=90CW+vflip)
-vf "transpose=1"

# overlay=x:y — composite second input over first (filter_complex)
-filter_complex "[0:v][1:v]overlay=W-w-10:H-h-10"   # bottom-right

# drawtext — burn text onto video
-vf "drawtext=text='Hello':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5"

# format — force a pixel format mid-graph
-vf "format=yuv420p"

# hflip / vflip — mirror horizontally / vertically
-vf "hflip"
-vf "vflip"
```

```bash
# A realistic combined example
ffmpeg -i input.mp4 -vf \
  "fps=30,scale=1280:-2,drawtext=text='%{pts\\:hms}':x=10:y=10:fontcolor=yellow,format=yuv420p" \
  -c:v libx264 -crf 20 output.mp4
```

## 3. ffprobe

`ffprobe` inspects media without decoding it fully.

```bash
# Full dump of streams and container/format info
ffprobe -show_streams -show_format input.mp4

# Human-readable summary (also printed by ffmpeg -i)
ffprobe -hide_banner input.mp4

# Just the resolution of the first video stream
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height -of csv=p=0 input.mp4

# Duration in seconds
ffprobe -v error -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 input.mp4

# Codec name of the video stream
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name -of default=nk=1:nw=1 input.mp4

# Frame rate as a fraction (e.g. 30000/1001)
ffprobe -v error -select_streams v:0 \
  -show_entries stream=r_frame_rate -of csv=p=0 input.mp4

# JSON output for scripting
ffprobe -v error -print_format json -show_streams input.mp4
```

## 4. ffmpeg metadata

```bash
# Read metadata (title, artist, comment, etc.)
ffprobe -show_format -show_entries format_tags input.mp4

# Set global metadata
ffmpeg -i input.mp4 -c copy \
  -metadata title="My Clip" -metadata comment="captured 2026" output.mp4

# Set per-stream metadata (here on the first audio stream)
ffmpeg -i input.mp4 -c copy -metadata:s:a:0 language=eng output.mp4

# Copy all metadata from input 0 to the output
ffmpeg -i input.mp4 -map_metadata 0 -c copy output.mp4

# Strip all metadata
ffmpeg -i input.mp4 -map_metadata -1 -c copy clean.mp4
```

## 5. Setting encoder for a specific codec

One codec (e.g. H.264) can have several encoders. Pick the encoder explicitly
with `-c:v`.

```bash
# Software H.264 (libx264) — high quality, CPU bound
ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4

# NVIDIA GPU H.264 (NVENC) — fast, hardware accelerated
ffmpeg -i input.mp4 -c:v h264_nvenc -preset p5 -b:v 5M output.mp4

# Intel Quick Sync H.264
ffmpeg -i input.mp4 -c:v h264_qsv -b:v 5M output.mp4

# VA-API (Linux, AMD/Intel)
ffmpeg -vaapi_device /dev/dri/renderD128 -i input.mp4 \
  -vf "format=nv12,hwupload" -c:v h264_vaapi output.mp4

# HEVC/H.265 with libx265
ffmpeg -i input.mp4 -c:v libx265 -crf 28 output.mp4

# See which encoders exist for a codec
ffmpeg -encoders | grep -i 264
```

## 6. Set the format (container) and codec for the output

The **container** (format) is the wrapper file; the **codec** is how the
streams inside are compressed. `-f` forces the container (useful for pipes or
when the extension is ambiguous); `-c:v`/`-c:a` set the codecs.

```bash
# Force MP4 container explicitly (normally inferred from .mp4 extension)
ffmpeg -i input.mkv -f mp4 -c:v libx264 -c:a aac output.mp4

# Container required when writing to a pipe (no filename to infer from)
ffmpeg -i input.mp4 -f mpegts - | someprogram

# Same codecs, different container (remux)
ffmpeg -i input.mkv -c copy -f matroska output.mkv

# List available containers
ffmpeg -formats
```

Container and codec are independent: an `.mkv` can hold H.264/HEVC/VP9, and
H.264 can live in `.mp4`, `.mkv`, `.ts`, etc. Not every codec is valid in
every container (e.g. Opus in MP4 needs recent muxers).

## 7. map

`-map` selects exactly which input streams go into the output. Without it,
FFmpeg picks one stream per type automatically; with it, you control
everything. Syntax: `-map input_index:stream_specifier`.

```bash
# Take video from input 0 and audio from input 1
ffmpeg -i video.mp4 -i music.mp3 \
  -map 0:v -map 1:a -c copy output.mp4

# Keep all streams from input 0 (video, all audio, subtitles)
ffmpeg -i input.mkv -map 0 -c copy output.mkv

# Select the second audio track only (0-indexed)
ffmpeg -i input.mkv -map 0:a:1 -c:a copy track2.aac

# Exclude a stream: everything except subtitles
ffmpeg -i input.mkv -map 0 -map -0:s -c copy nosubs.mkv

# Map a labeled filtergraph output (see filter_complex)
ffmpeg -i a.mp4 -i b.png \
  -filter_complex "[0:v][1:v]overlay[outv]" \
  -map "[outv]" -map 0:a output.mp4
```

## graph2dot

`graph2dot` is a tool shipped in the FFmpeg source (`tools/`) that converts a
filtergraph description into Graphviz DOT so you can visualize it. FFmpeg can
also dump the DOT of a running graph.

```bash
# Build the tool from the FFmpeg source tree
make tools/graph2dot

# Describe a filtergraph and render it to a PNG via Graphviz (dot)
echo "scale=640:480,hflip,overlay" | tools/graph2dot -o graph.dot
dot -Tpng graph.dot -o graph.png

# One-liner: filtergraph -> image
echo "movie=in.mp4,scale=1280:-2[v]" | tools/graph2dot | dot -Tpng -o graph.png
```

You can also get the internal graph FFmpeg actually builds by setting a high
log level; complex `-filter_complex` graphs print their pad connections when
run with `-v verbose`/`-v debug`.

## 8. Determining Pixel Format

The pixel format defines how color/chroma is laid out (e.g. `yuv420p` = planar
Y/U/V with 4:2:0 chroma subsampling, the most compatible for H.264/web).

```bash
# List every pixel format (flags: I=input, O=output, H=hw, P=paletted, B=bitstream)
ffmpeg -pix_fmts

# Show the pixel format of an existing file
ffprobe -v error -select_streams v:0 \
  -show_entries stream=pix_fmt -of default=nk=1:nw=1 input.mp4

# Force output pixel format (yuv420p = broadest player/browser support)
ffmpeg -i input.mov -c:v libx264 -pix_fmt yuv420p output.mp4

# Convert to 10-bit 4:2:2 for higher color fidelity
ffmpeg -i input.mov -c:v libx265 -pix_fmt yuv422p10le output.mkv

# Set pixel format inside a filtergraph
ffmpeg -i input.mp4 -vf "format=yuv420p" output.mp4
```

If playback shows green frames or a player refuses the file, forcing
`-pix_fmt yuv420p` is the usual fix — many players only support 4:2:0 8-bit.
