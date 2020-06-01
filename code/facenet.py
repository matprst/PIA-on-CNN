from facenet_pytorch import MTCNN, InceptionResnetV1

# If required, create a face detection pipeline using MTCNN:
# mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2')
print(resnet)

pytorch_total_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
print(pytorch_total_params)
