import cv2
from imutils.video import FPS
import kalman/
from numpy import *

OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"mil": cv2.TrackerMIL_create,

	}

dt = 1 #1 frame
vx=-1.35
vy=-0.2476635
# Initialization of state matrices
X = array([[0.0], [0.0],[vx], [vy]])
Y=array([[0], [0], [0], [0]])
P = diag((0.01, 0.01,0.01, 0.01))
B=diag((dt,dt))
print(X)
A = A = array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
U = array([[vx], [vy]])
Q= eye(X.shape[0])
oldx=0
oldy=0
dx=0
dy=0
w=0
h=0
# Measurement matrices

R = diag((1,1,1,1))
# Number of iterations in Kalman Filter
N_iter = 50


counter=0
tracker = OPENCV_OBJECT_TRACKERS["csrt" ]()

initBB = None

vs = cv2.VideoCapture("ISS.mp4")
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH) )
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT) )
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width,  height))

fps = None
while True:
    frame = vs.read()

    counter+=1
    frame = frame[1]
    if frame is None:
        break
    (H, W) = frame.shape[:2]
    if initBB is not None:
            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [v for v in box]
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                dx = x - oldx
                dy = y - oldy
                oldx = x
                oldy = y

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


            #Kalman filter:
            # if (counter %5 ==0):
            #     dx = x - oldx
            #     dy = y - oldy
            #     oldx = x
            #     oldy = y


            Y=array([[box[0]], [box[1]], [dx], [dy]])
            (X, P) = kalman.kf_predict(X, P, A, B, U, Q)
            # if(abs(dx) <0.5):
            #     Y=array([ [X[0]], [X[1]], [vx], [vy]])
            #     print(abs(dx))
            (X, P, K, IM, IS, LH) = kalman.kf_update(X, P, Y, A, R)
    else:


            (X, P) = kalman.kf_predict(X, P, A, B, U,Q)
            # if(abs(dx) <0.5):
            print(X[2][0],X[3][0])

            Y=array([ [oldx+vx], [oldy+vy], [X[2][0]], [X[3][0]]])
            oldx=oldx+vx
            oldy=oldy+vy
            #     print(abs(dx))
            (X, P, K, IM, IS,LH) = kalman.kf_update(X, P, Y, A, R)
            #print(X[0],X[1])
    cv2.rectangle(frame, (int(X[0])-5, int(X[1])-5), (int(X[0] + w-5), int(X[1] + h-5)), (0, 255, 255), 3)


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
                X[0]=initBB[0]
                X[1]=initBB[1]
                oldx=initBB[0]
                oldy=initBB[1]
                    # start OpenCV object tracker using the supplied bounding box
                    # coordinates, then start the FPS throughput estimator as well
                tracker.init(frame, initBB)
                fps = FPS().start()
    if key == ord("c"):
        initBB=None

    elif key == ord("q"):
        break
vs.release()
out.release()
cv2.destroyAllWindows()