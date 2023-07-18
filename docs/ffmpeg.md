- [Available encoders and decoders](#available-encoders-and-decoders)
- [Conversion between codecs](#conversion-between-codecs)
- [Change the output resolution](#change-the-output-resolution)
- [ffmpeg metadata](#ffmpeg-metadata)
  * [Using ffmpeg to copy metadata](#using-ffmpeg-to-copy-metadata)
  * [Image metadata exif](#image-metadata-exif)
- [Extracting key frames](#extracting-key-frames)
- [Extracting scene-changing frames](#extracting-scene-changing-frames)
- [Rotating video](#rotating-video)

# Available encoders and decoders
```
ffmpeg -encoders
ffmpeg -decoders
```
for instance
```
ffmpeg -filters | grep crop
```
to find help about specific filter:
```
ffmpeg -h filter=crop
```
to apply a filter:

```
-filter:v filter_name=param1=value:param2:value
```
or short version:

```
-vf filter_name=param1=value:param2:value
```
for instance:

```
ffmpeg -i Big_Buck_Bunny_1080_10s_2MB.mp4   -vf crop=out_w=640:out_h=480 out.mp4
```

if you have multiple input you should use `-filter_complex`:

```
ffmpeg -i first.flac -i second.flac -filter_complex acrossfade=d=10:c1=exp:c2=exp output.flac
```

filters in a chain are separated by commas "," chains by a semicolon ";" :

```
-vf filter_name1=param1=value:param2:value, filter_name2=param1=value:param2:value
```
for example:

```
ffmpeg -i Big_Buck_Bunny_1080_10s_2MB.mp4  -vf crop=out_w=640:x=100,rotate=angle=1 out.mp4
```

if an input or output is not specified it is assumed to come from the preceding or sent to the following item in the chain.


set number of threads:
```
-threads <N>
```

```
ffmpeg -threads 18 -i colmap.mp4 -threads 18 -vf "format=gray" out.mp4
```

turning a video into gray:
```
format=gray
```
changing scale:

```
scale=w:h
```

for instance:
```
scale=640:-1
```




# Conversion between codecs 

When converting audio and video files with ffmpeg, you do not have to specify the input and output formats. The input file format is auto-detected, and the output format is guessed from the file extension.

```
ffmpeg -i input.mp4 output.webm
```



When converting files, use the `-c:v` and `c:a` option to specify the codecs:
```
ffmpeg -i input.mp4 -c:v libvpx -c:a libvorbis output.webm
```


# Change the output resolution
use flag: `-s <width>x<height>`
for example:	
	
```
ffmpeg -i input.mp4 -s 192x168 out%d.jpg
```
	
	
or with the `-vf` option:

```
ffmpeg -i input.mp4 -vf scale=192:168 out%d.jpg
```

Note: The scale filter can also automatically calculate a dimension while preserving the aspect ratio: `scale=320:-1`, or `scale=-1:240`
```
ffmpeg -i input.mp4 -vf scale=192:-1 out%d.jpg
```



# ffmpeg metadata
Extract metadata:

```
ffmpeg -i input.mp4 -f ffmetadata metadata.txt
exiftool input.mp4 >metadata.txt
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


# Extracting key frames


In video compression, they use the so-called IPB frames

 - I frames (Intra picture): a complete picture
 - P frames (predictive picture): p frames stores difference between current frame and previous frame.
 - B frames (Bidirectionally predicted picture): b-frame stores difference between current frame and previous frame and later frame.

To extract I-frames:
```
ffmpeg -skip_frame nokey -i test.mp4 -vsync vfr -frame_pts true out-%03d.jpg
```
- -vsync vfr: discard the unused frames
- -frame_pts true: use the frame index for image names, otherwise, the index starts from 1 and


with scaling:

```
ffmpeg -i input.mp4   -vf "select='eq(pict_type,I)', scale=640:-1"    -vsync vfr -frame_pts true     out-%03d.jpg
```


We can also use the filter syntax to extract keyframes:
```
ffmpeg -i input.mp4   -vf "select='eq(pict_type,I)'" -vsync vfr -frame_pts true     out-%03d.jpg
```

- -vf filter_graph: set video filters





# Extracting scene-changing frames

If we only want to retain enough info from the video, extracting I-frames only may not be enough. The extracted key frames may still exhibit too much information redundancy. For example, if you have a slow-changing video, the difference between a frame and its subsequent frames will be negligible. To further reduce the number of images generated, we can also use scene filter to select frames that are likely to be a scene-changing frame.



```
ffmpeg -i input.mp4 -vf "select=gt(scene\,0.1), scale=640:-1"  -vsync vfr -frame_pts true out%03d.jpg
```


- select: the frame selection filter
- gt: greater than (>)
- scene: the scene change detection score, values in [0-1]. In order to extract suitable number of frames from the video, for videos with fast-changing frames, we should set this value high, and for videos with mostly still frames, we should set this value low (maybe 0.1 or even less depending on the video content).


In FFmpeg, scene-change detection refers to the process of identifying significant changes in video frames. Scene-change frames are those frames that mark a transition from one scene to another in a video. These frames are essential for various video processing tasks, such as video editing, compression, and analysis.

To perform scene-change detection in FFmpeg, you can use the select filter with the scene option. This filter analyzes the video frames and marks those frames that are considered scene-change frames.

Here's an example of how to detect scene-change frames in FFmpeg:


Let's break down the command:

-i input_video.mp4: Specifies the input video file.

-vf: Sets the video filtergraph, where we apply the scene-change detection filter.

"select='gt(scene,0.02)',showinfo": The filtergraph consists of two filters separated by a comma.

select='gt(scene,0.02)': The select filter with scene option performs scene-change detection. The 0.4 value is a threshold that determines the sensitivity of scene-change detection. Higher values make it less sensitive, and lower values make it more sensitive. You may need to adjust this value depending on your video.
showinfo: This filter shows information about each frame, including whether it was selected as a scene-change frame. This is useful for debugging and understanding the detection process.
-vsync vfr: This option sets the output frame rate to variable frame rate (VFR), as scene-change detection may produce frames with irregular intervals.

output_frames%d.jpg: Specifies the output frame filenames, which will be saved as JPEG images with sequential numbers.

After running the command, FFmpeg will process the input video and save the scene-change frames as JPEG images in the specified output format. Each scene-change frame will be saved as a separate image, which you can later use for further analysis or video editing purposes.

Keep in mind that scene-change detection is not a perfect process and may produce false positives or miss some scene changes, depending on the complexity of the video content and the chosen threshold. Fine-tuning the threshold value may be necessary to achieve more accurate results for specific videos.



# select every 10th frame


To select every 10th frame from a video using FFmpeg, you can use the select filter in combination with the notmod option. This filter allows you to skip frames based on a specified modulo value.

Here's the FFmpeg command to achieve this:
ffmpeg -i input_video.mp4 -vf "select='not(mod(n\,10))'" -vsync vfr output_every_10th_frame.mp4


-i input_video.mp4: Specifies the input video file.

-vf: Sets the video filtergraph, where we apply the frame selection filter.

"select='not(mod(n\,10))'": The select filter with the notmod option selects frames based on a modulo operation. The expression not(mod(n,10)) ensures that only frames with a frame number (n) that is not divisible by 10 will be selected. In other words, every 10th frame will be skipped, and the remaining frames will be selected.

-vsync vfr: This option sets the output frame rate to variable frame rate (VFR) as the frame selection might result in irregular frame intervals.

output_every_10th_frame.mp4: Specifies the output file name, where every 10th frame will be saved as a new video file.

# Rotating video

```
ffmpeg -i <input.mp4> -vf "transpose=2,transpose=2" <output.mp4>
```

- 0 = 90CounterCLockwise and Vertical Flip (default)
- 1 = 90Clockwise
- 2 = 90CounterClockwise
- 3 = 90Clockwise and Vertical Flip
Use -vf "transpose=2,transpose=2" for 180 degrees.


If you don't want to re-encode your video AND your player can handle rotation metadata you (rotation in the metadata):


```
ffmpeg -i <input.mp4> -map_metadata 0 -metadata:s:v rotate="180" -codec copy <output.mp4>
```



Refs: [1](https://linuxize.com/post/how-to-install-ffmpeg-on-ubuntu-20-04/), [2](https://ffmpeg.org/ffmpeg-filters.html#scdet-1)






