# ros_neuralrecon

## Description
This ROS node is a wrapper for the [NeuralRecon](https://github.com/zju3dv/NeuralRecon) project for real-time reconstruction.

## Getting started

Pull the image:
```
docker pull smellslikeml/neuralrecon:ros_noetic
```

Or build an image, following instructions in `../docker/README.md`. 

```
docker build -f docker/Dockerfile -t neuralrecon:ros_noetic .
```

Then run a container with:

```
docker run -it neuralrecon:ros_noetic /bin/bash
```
Finally, launch the node:
```
roslaunch ros_neuralrecon neural_recon.launch
```

