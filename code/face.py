from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
import cv2
# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=224)

# Create an inception resnet (in eval mode):
# resnet = InceptionResnetV1(pretrained='vggface2').eval()

from PIL import Image, ImageDraw

img = cv2.imread("/home/ubuntu/pbdang/CONTEST/MediaEval21/VisualSentiment/data/images/8929f487-6442-4476-98a1-66a1022d89fa.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)
# Get cropped and prewhitened image' tensor
boxes, probs, points = mtcnn.detect(img, landmarks=True)
boxes, probs, points = mtcnn.select_boxes(
                boxes, probs, points, img, method='largest_over_threshold',
                threshold=0.998
            )  

img_draw = img.copy()
draw = ImageDraw.Draw(img_draw)
for i, (box, point) in enumerate(zip(boxes, points)):
    draw.rectangle(box.tolist(), width=5)
    # for p in point:
    #     draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
    tmp = extract_face(img, box, save_path='/home/ubuntu/pbdang/CONTEST/MediaEval21/VisualSentiment/Smile-Detection/demo/detected_face_{}.png'.format(i))
    # print(tmp.shape)
img_draw.save('/home/ubuntu/pbdang/CONTEST/MediaEval21/VisualSentiment/Smile-Detection/demo/annotated_faces.png')

# Calculate embedding (unsqueeze to add batch dimension)
# img_embedding = resnet(img_cropped.unsqueeze(0))

# Or, if using for VGGFace2 classification
# resnet.classify = True
# img_probs = resnet(img_cropped.unsqueeze(0))
