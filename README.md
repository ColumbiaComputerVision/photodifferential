# Minimal Sensing for Orienting a Solar Panel
### [[Preprint]](https://cave.cs.columbia.edu/Statics/publications/pdfs/Klotz_Preprint25.pdf) [[Project Page]](https://cave.cs.columbia.edu/projects/categories/project?cid=Computational%20Imaging&pid=Minimal%20Sensing%20for%20Orienting%20a%20Solar%20Panel) [[Video]](https://cave.cs.columbia.edu/old/projects/photodiff/videos/video.mp4)

Code and data for the preprint "Minimal Sensing for Orienting a Solar Panel."

[Jeremy Klotz](https://cs.columbia.edu/~jklotz) and [Shree K. Nayar](https://www.cs.columbia.edu/~nayar/)

### UrbanSky
Download the HDR panoramas in the UrbanSky dataset [here](https://cave.cs.columbia.edu/repository/UrbanSky/download).

### Python environment
Install the python environment:
`conda create env -f env.yml`

### Run simulation
Set the path to the downloaded images in UrbanSky in urbansky.yml. This path should point to the parent folder container all the .exr files.


To run the full simulation, use `python main.py --config ../data/urbansky.yml`

Set `multiprocessing: True` in urbansky.yml to use all available CPU cores (this is necessary for reasonable run-times, but debugging is easier with multiprocessing disabled).

### Compute results
Use `python compute_results.py --config ../data/urbansky.yml` to compute the average harvested energy per scene, separated for unimodal and multimodal scenes. This produces Figure 8(a,b) in the paper.