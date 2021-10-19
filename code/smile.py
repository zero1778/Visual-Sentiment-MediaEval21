import cv2

cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml') 
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')

def detection(grayscale, img):
    face = cascade_face.detectMultiScale(grayscale, 1.3, 5)
    for (x_face, y_face, w_face, h_face) in face:
        cv2.rectangle(img, (x_face, y_face), (x_face+w_face, y_face+h_face), (255, 130, 0), 2)
        ri_grayscale = grayscale[y_face:y_face+h_face, x_face:x_face+w_face]
        ri_color = img[y_face:y_face+h_face, x_face:x_face+w_face] 
        eye = cascade_eye.detectMultiScale(ri_grayscale, 1.2, 18) 
        for (x_eye, y_eye, w_eye, h_eye) in eye:
            cv2.rectangle(ri_color,(x_eye, y_eye),(x_eye+w_eye, y_eye+h_eye), (0, 180, 60), 2) 
        smile = cascade_smile.detectMultiScale(ri_grayscale, 1.7, 20)
        for (x_smile, y_smile, w_smile, h_smile) in smile: 
            cv2.rectangle(ri_color,(x_smile, y_smile),(x_smile+w_smile, y_smile+h_smile), (255, 0, 130), 2)
    return img 

# vc = cv2.VideoCapture(0) 

# while True: 
img = cv2.imread("data/images/0c080be4-462b-4445-8791-0e5b65c08651.jpg")
scale_percent = 400 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
final = detection(grayscale, img) 
cv2.imwrite("demo_smile.jpg",final)
# cv2.imshow('Video', final) 
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break 

# vc.release() 
# cv2.destroyAllWindows() 