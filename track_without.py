import cv2
from imutils.video import FPS

from numpy import *

OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"mil": cv2.TrackerMIL_create,

	}




# Measurement matrices



tracker = OPENCV_OBJECT_TRACKERS["csrt" ]()

initBB = None

vs = cv2.VideoCapture("drone2.mp4")
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH) )
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT) )
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('without.mp4', fourcc, 20.0, (width,  height))
fps = None
count=0


bb_x=0
bb_y=0
sum_vel_x=0
sum_vel_y=0
while True:
    frame = vs.read()
    count+=1
    frame = frame[1]
    if frame is None:
        break
    (H, W) = frame.shape[:2]
    if initBB is not None:
            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                velx=x-bb_x
                vely=y-bb_y
                print(velx, vely)
                bb_x=x
                bb_y=y
                sum_vel_x+=velx
                sum_vel_y+=vely
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            fps.update()
            fps.stop()
            info = [
                ("Tracker", "kfc"),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    cv2.imshow("Frame", frame)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF
                # if the 's' key is selected, we are going to "select" a bounding
                # box to track
    if key == ord("s"):
                    # select the bounding box of the object we want to track (make
                    # sure you press ENTER or SPACE after selecting the ROI)
                initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                        showCrosshair=True)

                    # start OpenCV object tracker using the supplied bounding box
                    # coordinates, then start the FPS throughput estimator as well

                tracker.init(frame, initBB)
                print(initBB)
                (x, y, w, h) = initBB
                bb_x = x
                bb_y = y

                fps = FPS().start()

    elif key == ord("q"):
        break
vs.release()
out.release()
cv2.destroyAllWindows()

print("x=" ,sum_vel_x/count, "y=",sum_vel_y/count , "count=", count)