import cv2, time, pandas
from datetime import datetime

first_frame=None
status_list=[None, None]
times=[]
df=pandas.DataFrame(columns=["Start", "End"])

video=cv2.VideoCapture(0)

if not video.isOpened():
    print("Cannot open camera")
    exit()

while True:
    check, frame=video.read()
    status=0
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (21,21), 0)

    if first_frame is None:
        first_frame=gray
        continue

#compare the difference from the first frame to detect motion
    delta_frame=cv2.absdiff(first_frame, gray)

    thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
#30 is the absolute difference, 255 is the color we want (white),
#threshold binary allows us to have either black or white on the screen
#threshold function returns a tuple and we want to access only the index 1 element
#first: value for threshold second: actual frame

#removing the blackholes and smoothening the frame
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)
#none is for the array (we are not passing any) to make it more sophisticated
#no. of times the loop will run to to smoothen the frame

#find contours in the image and store it in a tuple
#contours help us identify the shapes present in an image
#Contour is a the line joining all the points along the boundary of an image that have the same intensity
    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#method to retreive the external contours of the shape/objects
#approximation method used to retreive the contours

#we will filter out to get only the contours that have area greater than 1000 pixels
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status=1

        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

    status_list.append(status)
    if status_list[-1] ==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1] ==0 and status_list[-2]==1:
        times.append(datetime.now())

    cv2.imshow("Gray frame",gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    key=cv2.waitKey(1)
    # print(gray)
    # print(delta_frame)

    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break

print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i], "End":times[i+1]}, ignore_index=True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()
