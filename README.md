# This extension project is for the sawyer robot.

It will use its camera along with a YOLO model to recognise and move an object to a target location. If the object is covered by an obstacle, the sawyer will move the obstacle out of the way.

Build the package and launch 

```bash
roslaunch extension_project main_scene.launch
```

## Important notes: 
Ensure you have the ultraltyics pip package installed. The program tries to install it but it makes the whole process easier if you pre-install

## Current status: 
Loads the sawyer arm successfully
Publishes joint states and camera information
Image recognition partially working. With more training on specific objects, this can be much more reliable

