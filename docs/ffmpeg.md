# 1. FFmpeg Common Options

- `-f` or `fmt`:              force format
- `-c` or `codec`:            codec name
- `-ss`:              set the start time offset
- `-vframes`:      set the number of video frames to output
- `-y`:                  overwrite output files
- `-filter_threads`:     number of non-complex filter threads
- `-codecs`:            show available codecs
- `-decoders`:           show available decoders
- `-encoders`:           show available encoders
- `-filters` :          show available filters- 
- `-pix_fmts`:          show available pixel formats
- `-hwaccels`:           show available HW acceleration methods
- `-vf` or `filter_graph`:    set video filters



# 2. FFmpeg Filters 
## 2.1 Available Filters

List all filters:
```
ffmpeg -filters
```

for instance
```
ffmpeg -filters | grep crop
```

to find help about specific filter:
```
ffmpeg -h filter=crop
```
## 2.2 Send Output of FFmpeg Directly to FFplay

```
ffmpeg -i Big_Buck_Bunny_1080_10s_2MB.mp4 -f nut - | ffplay -
```

`-f nut`: The NUT format is chosen here because it's a simple container format designed for streaming, which works well with pipes. You might also use 'mpegts' or others depending on the situation.

To have better quality:

```
ffmpeg -i Big_Buck_Bunny_1080_10s_2MB.mp4  -c:v hevc  -f   nut - | ffplay -
```

or 

```
ffmpeg -i Big_Buck_Bunny_1080_10s_2MB.mp4 -f matroska - | ffplay -
```

## 2.3 Apply a Filter
to apply a filter:

```
ffmpeg -filter:v filter_name=param1=value:param2:value
```
or short version:

```
ffmpeg -vf filter_name=param1=value:param2:value
```
for instance:

```
ffmpeg -i Big_Buck_Bunny_1080_10s_2MB.mp4   -vf crop=out_w=640:out_h=480 out.mp4
```

## 2.4 Filter Graph
A filter graph refers to a sequence or chain of filters that are applied to audio or video streams during processing. 

filters in a chain are separated by commas "," chains by a semicolon ";" :

```
ffmpeg -vf filter_name1=param1=value:param2:value, filter_name2=param1=value:param2:value
```
for example:

```
ffmpeg -i Big_Buck_Bunny_1080_10s_2MB.mp4  -vf crop=out_w=640:x=100,rotate=angle=1 out.mp4
```

if an input or output is not specified it is assumed to come from the preceding or sent to the following item in the chain.

## 2.5 Filter Complex
if you have multiple input you should use `-filter_complex`:

```
ffmpeg \
  -i Big_Buck_Bunny_1080_10s_2MB.mp4  \
  -i colmap.mp4  \
  -filter_complex \
"[0] scale=640:-1   [a]; \
 [1] scale=640:-1, nlmeans    [b]; \
 [a] [b] hstack=inputs=2 [out]" \
 -c:v hevc  -map "[out]" -f nut - | ffplay - 
```




## 2.6 Filter Complex vs Filter Graph


Both `filter_complex` and `-vf` (or `-af` for audio) are used to apply filters to media streams. However, they serve different purposes and are used in different scenarios:

1. `-vf` (Video Filter):

Usage: Used for simple video filtering tasks on single streams.
Scenario: You have one input video and you want to scale it, crop it, or apply other basic filters.
Example: To scale a video to a width of 1280 pixels while maintaining the aspect ratio, you'd use:

```
ffmpeg -i input.mp4 -vf "scale=1280:-1" output.mp4
```

2. `filter_complex`:

Usage: Used for more complex filtering tasks that involve multiple streams or require chaining multiple filters.
Scenario: You want to overlay one video on top of another, combine audio streams, split streams, or any other task that involves manipulating more than one stream or chaining multiple filters together.
Example: To overlay a watermark image on a video, you'd use:

```
ffmpeg -i video.mp4 -i watermark.png -filter_complex "[0:v][1:v] overlay=W-w-10:H-h-10" output.mp4
```
In this command, the [0:v][1:v] notation refers to the video stream of the first input and the video (image) stream of the second input, respectively. They are then processed by the overlay filter.
When to use which:

If you're dealing with a single stream (like just video or just audio) and applying simple filters, -vf or -af is likely sufficient.
If you're working with multiple streams, chaining filters, or combining/separating different inputs and outputs, filter_complex is the way to go.

## 2.7 Setting Number of threads
set number of threads:
```
-threads <N>
```

```
ffmpeg -threads 18 -i colmap.mp4 -threads 18 -vf "format=gray" out.mp4
```

```
-filter_threads
```


## 2.8 Commonly Used FFmpeg Filters

### 2.8.1 format

turning a video into gray:
```
format=gray
```
changing scale:

### split

split filter can be used to route a single input to multiple outputs. This is useful when you want to apply different filters to the same input without decoding it multiple times.
basic explanation of how the split filter works:

```
split=[outputs]
```
Where [outputs] is the number of output streams you want. If not specified, it defaults to 2.

Let's say you have a video, and you want to generate two outputs from it: one in grayscale and another with a vignette effect. You can use the split filter to achieve this:


```
ffmpeg -i Big_Buck_Bunny_1080_10s_2MB.mp4 -filter_complex "split=2[v1][v2];[v1]vignette[vout1];[v2]edgedetect[vout2];[vout1] [vout2]  hstack=inputs=2 [out]" -c:v hevc -map "[out]" -f nut - | ffplay -
```

- split=2[v1][v2]: This splits the input video stream into two identical outputs, [v1] and [v2].


### scale

```
scale=w:h
```
Note: The scale filter can also automatically calculate a dimension while preserving the aspect ratio: `scale=320:-1`, or `scale=-1:240`
```
ffmpeg -i input.mp4 -vf scale=192:-1 out%d.jpg
```

### select

#### Selecting Key Frames
For instance to select key frames (I-frames):
```
ffmpeg -i input.mp4   -vf "select='eq(pict_type,I)', scale=640:-1"    -vsync vfr -frame_pts true     out-%03d.jpg
```

In video compression, they use the so-called IPB frames

 - I frames (Intra picture): a complete picture
 - P frames (predictive picture): p frames stores difference between current frame and previous frame.
 - B frames (Bidirectionally predicted picture): b-frame stores difference between current frame and previous frame and later frame.

`-vsync` : The -vsync parameter in FFmpeg controls how video frames are synchronized when being processed. It has several options, and 'vfr' is one of them.
`-vsync vfr` stands for variable frame rate. When you set -vsync to vfr, it allows FFmpeg to adjust the video's timestamp in order to avoid duplicating or dropping frames. This can be useful when you are working with sources that have a variable frame rate.
In other words, it instructs FFmpeg to not duplicate frames and produce a video output with a variable frame rate matching the input as closely as possible.
It's often used in combination with the -copyts (copy timestamps) flag to ensure the timestamps are preserved as accurately as possible in the output video.


Also we can use:
```
ffmpeg -skip_frame nokey -i test.mp4 -vsync vfr -frame_pts true out-%03d.jpg
```
- -frame_pts true: use the frame index for image names, otherwise, the index starts from 1 and

In ffmpeg, the -skip_frame nokey option and select='eq(pict_type,I)' filter serve similar purposes of filtering and processing specific frames in a video, but they use different criteria to determine which frames to include or skip.

1. -skip_frame nokey: This option skips frames that are not keyframes. When using -skip_frame nokey, ffmpeg will only process and output keyframes.

2. select='eq(pict_type,I)': This filter expression is used to select frames based on their picture type. The eq(pict_type,I) condition checks if the current frame's picture type is equal to "I". Frames that meet this condition will be included in the output, while frames with other picture types, such as "P" (predicted) or "B" (bidirectional), will be excluded.


#### Extracting Scene-changing Frames
If we only want to retain enough info from the video, extracting I-frames only may not be enough. The extracted key frames may still exhibit too much information redundancy. For example, if you have a slow-changing video, the difference between a frame and its subsequent frames will be negligible. To further reduce the number of images generated, we can also use scene filter to select frames that are likely to be a scene-changing frame.


```
ffmpeg -i input.mp4 -vf "select=gt(scene\,0.1), scale=640:-1"  -vsync vfr -frame_pts true out%03d.jpg
```
- select: the frame selection filter
- gt: greater than (>)
- scene: the scene change detection score, values in [0-1]. In order to extract suitable number of frames from the video, for videos with fast-changing frames, we should set this value high, and for videos with mostly still frames, we should set this value low (maybe 0.1 or even less depending on the video content).
- "select='gt(scene,0.02)',showinfo": The filtergraph consists of two filters separated by a comma.

`select='gt(scene,0.02)'`: The select filter with scene option performs scene-change detection. The 0.02 value is a threshold that determines the sensitivity of scene-change detection. Higher values make it less sensitive, and lower values make it more sensitive. 
- `showinfo`: This filter shows information about each frame, including whether it was selected as a scene-change frame. This is useful for debugging and understanding the detection process.
`-vsync vfr`: This option sets the output frame rate to variable frame rate (VFR), as scene-change detection may produce frames with irregular intervals.


#### Select Every Nth Frame

To select every 10th frame from a video using FFmpeg, you can use the select filter in combination with the `not mod` option. 

```
ffmpeg -i input_video.mp4 -vf "select='not(mod(n\,10))'"  -vsync vfr -frame_pts true out%03d.jpg
```


### histeq

### nlmeans

### median

```
median=radius=5
```
or 

```
ffmpeg -i Big_Buck_Bunny_1080_10s_2MB.mp4 -vf "scale=640:-1,median=radius=5,convolution='0 -1 0 -1 5 -1 0 -1 0:0 -1 0 -1 5 -1 0 -1 0:0 -1 0 -1 5 -1 0 -1 0:0 -1 0 -1 5 -1 0 -1 0'" -c:v rawvideo  -f nut - | ffplay -
```

or 

```
ffmpeg -i Big_Buck_Bunny_1080_10s_2MB.mp4 -filter_complex "split=2[v1][v2];[v1]median=radius=5[vout1];[v2]convolution='0 -1 0 -1 5 -1 0 -1 0:0 -1 0 -1 5 -1 0 -1 0:0 -1 0 -1 5 -1 0 -1 0:0 -1 0 -1 5 -1 0 -1 0'[vout2];[vout1] [vout2]  hstack=inputs=2 [out]" -c:v rawvideo -map "[out]" -f nut - | ffplay -
```




### blurdetect

### hqdn3d

### smooth

### convolution
```
convolution='0 -1 0 -1 5 -1 0 -1 0:0 -1 0 -1 5 -1 0 -1 0:0 -1 0 -1 5 -1 0 -1 0:0 -1 0 -1 5 -1 0 -1 0'
```

### transpose
The following will rotate the video
```
ffmpeg -i input.mp4 -vf "transpose=2,transpose=2" output.mp4
```

- 0 = 90CounterCLockwise and Vertical Flip (default)
- 1 = 90Clockwise
- 2 = 90CounterClockwise
- 3 = 90Clockwise and Vertical Flip
Use -vf "transpose=2,transpose=2" for 180 degrees.

If you don't want to re-encode your video AND your player can handle rotation metadata you (rotation in the metadata):

```
ffmpeg -i input.mp4 -map_metadata 0 -metadata:s:v rotate="180" -codec copy output.mp4
```

### fps
fps filter is used here to say that we need 1 frame every 5 seconds
```
ffmpeg -i colmap.mp4 -vf "fps=1/5" out%d.jpg
```


# ffprobe
ffprobe report:

```
ffprobe Big_Buck_Bunny_1080_10s_2MB.mp4
```

Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'Big_Buck_Bunny_1080_10s_2MB.mp4':
  Metadata:
    major_brand     : isom
    minor_version   : 512
    compatible_brands: isomiso2avc1mp41
  Duration: 00:00:10.00, start: 0.000000, bitrate: 1677 kb/s
    Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, 1920x1080 [SAR 1:1 DAR 16:9], 1672 kb/s, 60 fps, 60 tbr, 15360 tbn, 120 tbc (default)
    Metadata:
      handler_name    : VideoHandler



SAR: Storage aspect ratio (SAR): the dimensions of the video frame, expressed as a ratio. For a 576i video, this is 5:4 (720×576).	
DAR: Display aspect ratio (DAR): the aspect ratio the video should be played back at. For SD video, this is 4:3.


tbr: time based rate
tbn = the time base in AVStream that has come from the container
tbc = the time base in AVCodecContext for the codec used for a particular stream
tbr = tbr is guessed from the video stream and is the value users want to see when they look for the video frame rate

```
ffprobe -show_format  Big_Buck_Bunny_1080_10s_2MB.mp4
```

```
ffprobe -show_streams  Big_Buck_Bunny_1080_10s_2MB.mp4
```



# ffmpeg metadata
Extract metadata:

```
ffmpeg -i Big_Buck_Bunny_1080_10s_2MB.mp4 -f ffmetadata metadata.txt
exiftool Big_Buck_Bunny_1080_10s_2MB.mp4 >metadata.txt
```


## Using ffmpeg to copy metadata


ffmpeg, by default, makes all metadata from the first input file available, to the output file muxer, for writing. -map_metadata allows to override that, by either pointing to a different input, or by telling ffmpeg to discard input global metadata (value of -1).


use:
`-movflags use_metadata_tags`

For instance:

```
ffmpeg -i $input_file -movflags use_metadata_tags -crf 22 $output_file
```


The `-metadata` option is for manipulating the metadata. If you just want to copy the metadata from an input file to an output file, you should use the `-map_metadata` option:	
	
```	
ffmpeg -i input.mp4 -map_metadata 0 -c copy output.mp4
```
## Image metadata exif
```
sudo apt install exif
exif image.jpg
```





# Set the format (container) and codec for the output 

When converting audio and video files with ffmpeg, you do not have to specify the input and output formats. The input file format is auto-detected, and the output format is guessed from the file extension.

```
ffmpeg -i input.mp4 output.webm
```

To set the format (container) and codec for the output file in FFmpeg, you would use the `-f` option for the format and `-c:v / -c:a` for the video/audio codec.


For example, if you want to convert an input video to the MOV format with the H.264 video codec and AAC audio codec, you could use the following command:
```
ffmpeg -i input.mp4 -f mov -c:v libx264 -c:a aac output.mov
```
- `-i` input.mp4 specifies the input file.
- `-f` mov sets the output container format to MOV.
- `-c:v` libx264 sets the video codec to H.264. The "libx264" is the encoder library FFmpeg uses to produce H.264 video.
- `-c:a` aac sets the audio codec to AAC.
-  output.mov is the output file.


In the following example, FFmpeg is converting the video file input.mp4 into output.mp4 using the `HEVC` codec for the video stream.

HEVC offers better data compression than H.264 while maintaining similar or better video quality, but it requires more computational resources for encoding and decoding. It's commonly used for 4K and 8K Ultra High Definition video.

```
ffmpeg -i input.mp4 -c:v hevc output.mp4
```
list of all available encoders and 
```
ffmpeg -encoders
```
list of all available decoders:
```
ffmpeg -decoders
```

list of all codecs:
```
ffmpeg -codecs
```
list of all formats:
```
ffmpeg -formats
```
## x265 vs HEVC

HEVC (High Efficiency Video Coding): 

1. Also Known As: H.265, is a video compression standard, the successor to H.264 (also known as AVC). HEVC is designed to achieve roughly double the data compression of H.264 at the same level of video quality, or substantially improved video quality at the same bit rate. It supports resolutions up to 8192×4320, including 8K UHD.
Purpose: Just like H.264, HEVC defines the method to compress and decompress video content. However, it doesn't implement it; that's the job of software or hardware encoders and decoders.

2. x265:

is a free software library and application developed to encode video streams into the HEVC compression format. In essence, x265 is a software encoder that turns raw video into the compressed HEVC format.
Relationship with HEVC: x265 is an implementation of the HEVC standard. It's not the only one, but it's one of the most popular and efficient open-source implementations available.
Purpose: To provide a way for software to produce video files that are compliant with the HEVC standard.

**Analogy:**
Think of HEVC (or H.265) as a recipe for making a particular dish, while x265 is a chef that follows this recipe to cook the dish. There could be many chefs (software implementations) capable of making the same dish (encoding video into the HEVC format), but they all follow the same recipe (the HEVC standard).


If you want to encode a video using the HEVC codec through FFmpeg using the x265 encoder, you would use a command like:

```
ffmpeg -i input.mp4 -c:v libx265 -crf 28 output.mp4
```

Here, `-c:v libx265` tells FFmpeg to use the x265 library to encode the video stream in the HEVC format. The `-crf` option adjusts the quality, with lower values providing higher quality (and larger file sizes) and higher values providing lower quality (and smaller file sizes).

## codec copy

the `-codec copy` or its equivalents (-c copy, -c:v copy, -c:a copy, etc.)  option tells FFmpeg to copy the input streams (audio, video, and possibly other streams like subtitles) directly to the output without re-encoding them. It is a way to change the container format without altering the actual content of the streams. This operation is often referred to as "remuxing" or "stream copying."

```
ffmpeg -i input.mkv -codec copy output.mp4
```

When you use `-codec copy`, you're not changing the quality or properties of the streams, which means the operation is:

1. Fast: Because you're not re-encoding, the process is much quicker than converting the video or audio to a different codec.
2. Lossless: No quality degradation since there's no re-encoding involved.

You can also specify which codecs to copy. For example, if you only wanted to copy the video stream:

```
ffmpeg -i input.mkv -c:v copy -an output.mp4
```

```
 _______              ______________            ________
|       |            |              |          |        |
| input |  demuxer   | encoded data |  muxer   | output |
| file  | ---------> | packets      | -------> | file   |
|_______|            |______________|          |________|

```

Filtering requires the input video to be fully decoded into raw video, then the raw video is processed by the filter(s):

```
 _______              ______________
|       |            |              |
| input |  demuxer   | encoded data |   decoder
| file  | ---------> | packets      | -----+
|_______|            |______________|      |
                                           v
                                       _________
                                      |         |
                                      | decoded |
                                      | frames  |
                                      |_________|
                                           |
                                           v
                                       __________
                                      |          |
                                      | filtered |
                                      | frames   |
                                      |__________|
 ________             ______________       |
|        |           |              |      |
| output | <-------- | encoded data | <----+
| file   |   muxer   | packets      |   encoder
|________|           |______________|
```


# map

In FFmpeg, the -map option is used to specify which streams from the input files should be included in the output file.

By default, FFmpeg includes only one stream of each type (video, audio, subtitle) from the input file in the output file. The `-map` option allows you to override this behavior and define exactly what streams you want to include in the output.

The syntax for the `-map` option is `-map i:s`, where `i` is the index number of the input file (starting from 0), and `s` is the index number of the stream in that file. Also you can use `-map 0:v` or `-map 0:a` to  refer video or audio channel.

The following command tells FFmpeg to take the first and second streams from the first input file (input.mp4) and include them in the output file (output.mkv). This could be used, for example, to ensure that both the video and audio from the input file are included in the output.

```
ffmpeg -i input.mp4 -map 0:0 -map 0:1 output.mkv
```

This command takes all streams from the first input file (video.mp4) and all streams from the second input file (audio.wav) and includes them in the output file (output.mkv). This could be used to add an audio track to a video file.


```
ffmpeg -i video.mp4 -i audio.wav -map 0 -map 1 output.mkv
```

This command includes all video (0:v) and audio (0:a) streams from the first input file (input.mkv) in the output file (output.mp4). This could be used to ensure that all video and audio streams, not just one of each, are included in the output.
```
ffmpeg -i input.mkv -map 0:v -map 0:a output.mp4
```






# graph2dot


The graph2dot tool in FFmpeg allows you to create a visual representation (in the DOT format) of a filtergraph. The DOT format can then be used with software like Graphviz to generate graphical visualizations.

Here's a step-by-step guide to use graph2dot:

Create your filtergraph: You must first define the filtergraph you want to visualize. For this example, let's take a simple filter chain: a split filter followed by two separate scale filters.

Filtergraph:

```
split[s0][s1]; [s0]scale=w=640:h=360[s0out]; [s1]scale=w=320:h=240[s1out]
```

Generate the DOT file using graph2dot: You can utilize FFmpeg's graph2dot by running:

```
echo "split[s0][s1]; [s0]scale=w=640:h=360[s0out]; [s1]scale=w=320:h=240[s1out]" | ffmpeg -filter_complex_script - -y -f null - 2>&1 | graph2dot -o filtergraph.dot
```
Visualize the filtergraph: After obtaining the filtergraph.dot file, you can use Graphviz to visualize it. If you have Graphviz installed, you can run:

```
dot -Tpng -o filtergraph.png filtergraph.dot
```

This command will generate a PNG image (filtergraph.png) that represents the filtergraph visually.

Please note that graph2dot is a utility provided with FFmpeg's source code and might not be installed by default when you install FFmpeg using some package managers. You might need to obtain it from FFmpeg's source and possibly make it executable.


# Determining Pixel Format

```
ffprobe -v error  -show_entries stream=pix_fmt  input_video.mp4
```
to list all supported 
```
ffmpeg pix_fmts of a video 
```


| NAME     | NB_COMPONENTS |    BITS_PER_PIXEL |
|:---------|:-------------:|:----- |
| yuv420p  | 3             | 12    |
| yuyv422  | 3             | 16    |
| yuv444p  | 3             | 24    |
| rgb24    | 3             | 24    |



the difference between `yuv420p` and `yuv420`:


In the context of video color spaces and chroma subsampling, `yuv420p` and `yuv420` both refer to a similar concept, but their notation can sometimes be a little different depending on the context. Here's an explanation:

**YUV:**
is a color space used in video compression. It separates image intensity (Y) from color information (UV), making it easier to compress.
Components:
Y: Luma (brightness)
U and V: Chrominance (color) components.

**420 (or 4:2:0):**
This refers to chroma subsampling, which is a way to encode color information at lower resolutions than the luma data. 
What It Means:

- The "4" in `4:2:0` means that for every `4x2` block of 8 pixels, there are 8 Y values (every pixel has its own Y value).
- The "2" means that for the same `4x2` block, there are only 2 U values.
- The "0" means that for the same block, there are only 2 V values, the same as U.
So, in total, a `4x2` block of 8 pixels will have `8` Y values, `2` U values, and `2` V values. The U and V values are shared among multiple pixels.

**yuv420p vs. yuv420:**

1. yuv420p:
The "p" in yuv420p stands for "planar."
In this format, all Y values are stored consecutively, followed by all U values, then all V values. Each of these is called a plane, hence "planar."

2. yuv420 (without the 'p'):
In many contexts, "yuv420" and "yuv420p" can mean the same thing, especially when "planar" is the default or assumed format.
However, in some contexts, omitting the "p" might mean that the specific storage layout isn't being defined, or it's assumed based on context.
Additional Notes:

There are other formats like yuv420sp, where "sp" stands for "semi-planar." In such formats, Y values are in one plane, and the U and V values are interleaved in a second plane.
Understanding the difference between planar and other formats is essential when working at a low level with video data, like in video decoders or specific image processing tasks.
In general, if you're working with tools like FFmpeg and you see yuv420p, it's referring to the 4:2:0 chroma subsampling with the data stored in a planar format. If you just see yuv420, it's often safe to assume it means the same thing unless the context suggests otherwise.







Refs: [1](https://linuxize.com/post/how-to-install-ffmpeg-on-ubuntu-20-04/), [2](https://ffmpeg.org/ffmpeg-filters.html#scdet-1)






