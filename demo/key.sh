FILE_NAME="meterialCheck_2"
rm -rf ../output/${FILE_NAME}

python demo.py --config-file ../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
		--video-input ../input/${FILE_NAME}.mp4 \
		--output ../output/${FILE_NAME}/${FILE_NAME}.mp4 \
		--parallel 1 \
		--opts MODEL.WEIGHTS ../model/model_final_997cc7.pkl \
