#! /bin/bash
XSOCK=/tmp/.X11-unix
XAUTH=/tmp.docker.xauth
touch $XAUTH

docker run -it --volume=$XSOCK:$XSOCK:rw \
	--volume=$XAUTH:$XAUTH:rw \
	--env=XAUTHORITY=${XAUTH} \
	--env="DISPLAY" --user=rl \
    mujoco-humanoid-sac:v1