MOUNTED_PATH="/home/cvpr-pu/sungpil/posetron"
NAME="gtc"
docker run --runtime=nvidia -it --name ${NAME} -v /dev/snd:/dev/snd -v ${MOUNTED_PATH}:/${NAME} -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY \
            --cap-add SYS_PTRACE \
            --ip host khosungpil/gtc:3.0
