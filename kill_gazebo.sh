#!/bin/bash

pkill -f gazebo || true
pkill -f gzserver || true
pkill -f gzclient || true
echo "Done killing Gazebo processes."
