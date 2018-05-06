import cv2

# Create the video capture
videoCapture = cv2.VideoCapture(0)

# Use the face detection with smile detection together, otherwise, smiles EVERYWHERE.

# Create the haar cascades
cascadeClassifierFaces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
  # Read the frame
  ret, frame = videoCapture.read()

  # Convert the frame to gray scale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Use the classifier to detect
  detectFacesResults = cascadeClassifierFaces.detectMultiScale(
   gray,
   scaleFactor=1.3,
   minNeighbors=5,
   # Larger minSize seems to avoid false positives in background
   minSize=(70, 70),
   # NOTE: FIND_BIGGEST_OBJECT and DO_ROUGH_SEARCH flags not in cv v3
#    flags = (cv2.CASCADE_SCALE_IMAGE, cv2.FIND_BIGGEST_OBJECT, cv2.DO_ROUGH_SEARCH)
   flags = (cv2.CASCADE_SCALE_IMAGE)
  )
  
  print("Detected {0} detectFacesResults".format(len(detectFacesResults)))

  # Draw a rectangle around the detectFacesResults
  for (x, y, w, h) in detectFacesResults:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#     cv2.putText(frame,"Face Detected",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    
  # Display the resulting frame
  cv2.imshow('frame', frame)
  
  # If escape key is pressed, exit while loop
  if 0xFF & cv2.waitKey(1) == 27:
    break

# When everything done, release the capture
videoCapture.release()
cv2.destroyAllWindows()
