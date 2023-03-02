import cv2, time

video=cv2.VideoCapture(0)
if not video.isOpened():
    print("Cannot open camera")
    exit()
a=1
while True:
    a=a+1
    check, frame=video.read()
    print(check)
    print(frame)

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #time.sleep(1)
    cv2.imshow("Capturing",gray)


    key=cv2.waitKey(1)
    if key==ord('q'):
        break
print("total number of frames generated: ", a)
video.release()
cv2.destroyAllWindows()
