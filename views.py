from django.shortcuts import render, redirect
from django.views.generic import FormView,TemplateView,DetailView
from .models import Post
from .forms import PostForm,ImageForm  # Assuming TensorFlow backend
import numpy as np
from PIL import Image
from inference_sdk import InferenceHTTPClient
import torch
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from django.http import JsonResponse
import base64
import os
import base64
import io
import cv2
import supervision as sv
import matplotlib.pyplot as plt
import seaborn_image as isimage
import json
from django.core.files.base import ContentFile





Disease= ['Tomato___Bacterial_spot', 'Tomato___Early_blight','Tomato_Healthy','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato_Mosaic_Virus', 'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Septoria_Leaf']
Rem=['''BS =  Bacterial Spot:
   - Pesticide Used: Copper-based fungicides
   - Quantity: Mix 2-4 tablespoons per gallon of water
   - About the Disease: Wide occurrence in India, particularly in areas with frequent rainfall.
   - Precautions:
     - Avoid overhead irrigation to minimize water splashing.
     - Remove and destroy infected plant material promptly.''',
     '''EB = Early Blight:
   - Pesticide Used: Mancozeb
   - Quantity: Mix 2-4 tablespoons per gallon of water
   - About the Disease: Widespread in various regions of India.
   - Precautions:
     - Practice crop rotation with non-related plants.
     - Mulch around plants to prevent soil splashing.''',
     '''LB =  Late Blight:
   - Pesticide Used: Mancozeb
   - Quantity: Mix 2-4 tablespoons per gallon of water
   - About the Disease: Found in cooler and moist regions of India.
   - Precautions:
     - Apply fungicides preventively during periods of high humidity.
     - Remove and destroy infected plant material to reduce the source of spores.''',
     '''LM = Leaf Mold:
   - Pesticide Used: Copper-based fungicides
   - Quantity: Dilute according to the manufacturer's instructions
   - About the Disease: Common in areas with high humidity and moderate temperatures.
   - Precautions:
     - Provide adequate spacing for air circulation.
     - Avoid overhead irrigation.''',
     '''SLS = Septoria Leaf Spot:
   - Pesticide Used: Chlorothalonil
   - Quantity: Follow the manufacturer's instructions for dilution
   - About the Disease: Common in regions with moderate temperatures and high humidity.
   - Precautions:
     - Water the soil, not the foliage, to minimize the spread of spores.
     - Remove and destroy infected leaves promptly.''',
     
    '''SM = Spider Mites (Two-Spotted Spider Mite):
      - Pesticide Used: Insecticidal soap
      - Quantity: Follow the product instructions for dilution
      - About the Pest: Found in dry and hot regions of India.
      - Precautions:
        - Maintain proper humidity to discourage mite infestations.
        - Regularly spray plants with water to reduce mite populations.''',
     '''TS = Target Spot:
   - Pesticide Used: Chlorothalonil
   - Quantity: Follow the manufacturer's instructions for dilution
   - About the Disease: Common in warm and humid regions of India.
   - Precautions:
     - Rotate crops to reduce the risk of disease buildup.
     - Ensure proper spacing between plants for good air circulation.''',
     '''TYL = Tomato Yellow Leaf Curl Virus:
   - Pesticide Used: Imidacloprid
   - Quantity: Follow the recommended dosage on the product label
   - About the Disease: Prevalent in tropical and subtropical regions of India.
   - Precautions:
     - Use virus-resistant tomato varieties.
     - Control whitefly populations through insecticides or reflective mulches.''',
     ''' TMS=Tomato Mosaic Virus:
   - Pesticide Used: Neem oil
   - Quantity: Mix 1-2 tablespoons of neem oil with a gallon of water
   - About the Disease: Found in various locations in India, especially in areas with high humidity.
   - Precautions:
     - Use virus-free seeds and seedlings.
     - Control and manage insect vectors like aphids and whiteflies.''',
     '''TH = Tomato - Healthy:
   - No specific pesticides needed for healthy plants.
   - Preventive Measures:
     - Practice good garden hygiene.
     - Monitor plants regularly for any signs of diseases or pests.''',
   ]

CLIENT = InferenceHTTPClient(
api_url="https://detect.roboflow.com",
api_key="GUIXwvQ9d1FIoKHb3wWl"
)

class CapsuleNetwork(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CapsuleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.primary_capsules = nn.Conv2d(256, 32*8, kernel_size=3, stride=2, padding=1)
        self.digit_capsules = nn.Linear(32*8 * 56 * 56, num_classes * 16)  # Adjusted input size

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.primary_capsules(x))
        x = x.view(x.size(0), -1)
        x = self.digit_capsules(x)
        return x
      
      
class Diseaseview(TemplateView):
  template_name='prediction_result.html'
  
class MainPage(TemplateView):
  template_name='main.html'
  
  
class HomePageView(TemplateView):#returning everything. display page
    template_name='home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['Post'] = Post.objects.all()
        return context

class PostDetailView(DetailView):#maynot need
    template_name="detail.html"
    model = Post
    


def all_posts(request):
  posts = Post.objects.all()  # Retrieve all Post objects
  context = {'posts': posts}
  return render(request, 'all_posts.html', context)


class PredictImage(FormView):
  template_name = 'predict.html'
  form_class = PostForm
  success_url = '/'
  
  def form_valid(self, form):
    new_object = form.save(commit=False)  # Don't save initially

    # Image prediction logic
    image_file = form.cleaned_data['image']
    img = self.preprocess_image(image_file)  # Replace with your preprocessing function

    # Load the model (replace 'your_model.h5' with your actual model filename)
    model_path = 'CPNET.PTH'
    model = CapsuleNetwork(input_shape=(3, 224, 224), num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    with torch.no_grad():
      prediction=model(img)
      
    predicted_class=torch.argmax(prediction).item()
    print(predicted_class)
    pred=Disease[predicted_class-1]
    remedy=Rem[predicted_class-1]
  
    # Update post object with prediction and save
    new_object.prediction = pred
    new_object.save()

    context = {'new_object': new_object,'Rem':remedy}  # Adjusted context dictionary
    return render(self.request, 'prediction_result.html', context)

  def preprocess_image(self, image_file):
    # Implement your image preprocessing logic here (resize, normalize etc.)
    # This example resizes to 224x224 (common for image classification)
    img = Image.open(image_file)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to fit model input
        transforms.ToTensor(),           # Convert image to PyTorch tensor
        transforms.Normalize(            # Normalize the image
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # Apply the transformations
    img = transform(img)
    # Add batch dimension
    img = img.unsqueeze(0)
    return img
  
  
            
            
            
            
        
       
def get_top_two_images(folder_path):
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
                    if filename.endswith(('.jpg', '.jpeg', '.png'))]
    image_paths.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time (newest first)
    return image_paths[:1]
  
def hello_view(request):
    if request.method == 'POST':
        top_two_images = get_top_two_images('D:\Project\imageshow\crop_results')
        hello=[]
        base64_encoded_images = []
        processed_image=[]

        img = preprocess_image("D:\\Project\\imageshow\\crop_results\\Leaf Mold_1.jpg")
        print(img.shape)
        processed_image.append(img)
        model_path = 'CPNET.PTH'
        model = CapsuleNetwork(input_shape=(3, 224, 224), num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        with torch.no_grad():
            prediction=model(img)
        _,predicted_class=torch.max(prediction,1)
        img_array = img.squeeze().permute(1, 2, 0).numpy()
        img_pil = Image.fromarray((img_array*255).astype('uint8'))
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG')

            
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

            
            
            
            
            
        return render(request, 'predict_result.html', {
            'base64_encoded_images': img_str,
            'predicted_classes': Disease[predicted_class.item()],
        })
    return render(request, 'hello.html')
def preprocess_image(image_file):
    # Implement your image preprocessing logic here (resize, normalize etc.)
    # This example resizes to 224x224 (common for image classification)
    img = Image.open(io.BytesIO(image_file))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to fit model input
        transforms.ToTensor(),
        # Convert image to PyTorch tensor
        
    ])
    # Apply the transformations
    img = transform(img)
    # Add batch dimension
    img = img.unsqueeze(0)
    return img

def predict_image(request):
    if request.method == 'POST':
        processed_image=[]

        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']
            image_bytes = image.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            if image_base64:
        # Send the base64-encoded image for inference using the Roboflow client
                result = CLIENT.infer(image_base64, model_id="tomata-leaf-disease/3")

        # Process the result to generate annotated image
                labels = [item["class"] for item in result["predictions"]]
                detections = sv.Detections.from_inference(result)
                label_annotator = sv.LabelAnnotator()
                bounding_box_annotator = sv.BoundingBoxAnnotator()
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
                annotated_image = bounding_box_annotator.annotate(
                scene=image, detections=detections)
                annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels)

        # Convert the annotated image to a base64-encoded string
                imagecrop=crop_and_save_bounding_boxes(annotated_image, json.dumps(result["predictions"]), 'D:\\Project\\imageshow\\crop_results')
                top_two_images = get_top_two_images('D:\Project\imageshow\crop_results')
                path=top_two_images[0]
                img = preprocess_image(imagecrop)
                print(img.shape)
                processed_image.append(img)
                model_path = 'CPNET.PTH'
                model = CapsuleNetwork(input_shape=(3, 224, 224), num_classes=10)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                with torch.no_grad():
                  prediction=model(img)
                  predicted_class=torch.argmax(prediction).item()
                img_array = img.squeeze().permute(1, 2, 0).numpy()
                img_pil = Image.fromarray((img_array*255).astype('uint8'))
                buffer = io.BytesIO()
                img_pil.save(buffer, format='JPEG')

            
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                remedy=Rem[predicted_class-1]
                image_data = ContentFile(base64.b64decode(img_str), name=f'{predicted_class}.jpg')  # Customize filename


                

            
            
            post = Post.objects.create(
                text='new try',  # Assuming predicted_class holds the text data
                image=image_data,
                prediction=Disease[predicted_class-1]  # Store prediction if needed
)
            
            
            return render(request, 'predict_result.html', {
            'base64_encoded_images': img_str,
            'predicted_classes': Disease[predicted_class-1],
            'rem' : remedy,
        })
        return render(request, 'hello.html')

        
        # Render the HTML template with the annotated image
        # Inside the /upload route
                
                
            

    

            # Preprocess the image (resize, normalize, etc.)
            

    else:
        form = ImageForm()
        context = {'form': form}
        return render(request, 'pred.html', context)

def crop_and_save_bounding_boxes(image, decoded_data, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    decoded_data = decoded_data.replace("'", '"')
    decoded_data = json.loads(decoded_data)
    print(decoded_data)
    print(type(image))
    print("Image shape:", image.shape)
    sorted_list = sorted(decoded_data, key=lambda x: x['confidence'], reverse=True)

    # Storing the top 2 detections with highest confidence in another list
    top_2_confidence = sorted_list[:1]
    # Crop each bounding box
    for i, prediction in enumerate(top_2_confidence):
        print(type(prediction))
        #x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        class_name = prediction['class']
        # Crop the bounding box

        roi_x = int(prediction['x'] - prediction['width'] / 2)
        roi_y = int(prediction['y'] - prediction['height'] / 2)
        roi_width = int(prediction['width'])
        roi_height = int(prediction['height'])

        roi = image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        # Save the cropped image
        jpeg_bytes=convert_roi_to_jpeg(roi)
        filename = f"{class_name}_{i+1}.jpg"  # Example: "Leaf_Mold_1.jpg"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, roi)
        print(f"Saved {filename} in {output_folder}")
        return jpeg_bytes

def convert_roi_to_jpeg(roi):
    # Convert numpy array to BGR image
    bgr_image = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)

    # Encode BGR image to JPEG format in memory
    _, jpeg_image = cv2.imencode('.jpg', bgr_image)

    # Convert JPEG image data to bytes
    jpeg_bytes = jpeg_image.tobytes()

    return jpeg_bytes
  
  
def save_image():
    # Get the base64-encoded annotated image from the request
    request_data = request.json
    
    # Extract annotated image base64 and predictions from the request data
    annotated_image_base64 = request_data.get('annotated_image_base64')
    predictions = request_data.get('predictions')

    # Decode the base64-encoded image
    annotated_image_bytes = base64.b64decode(annotated_image_base64)
    
    # Convert the image bytes to a numpy array
    annotated_image_np = np.frombuffer(annotated_image_bytes, dtype=np.uint8)
    
    # Decode the image using OpenCV
    annotated_image = cv2.imdecode(annotated_image_np, cv2.IMREAD_COLOR)

    # Save the annotated image
    output_folder = 'annotated_images'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'annotated_image.jpg')
    cv2.imwrite(output_path, annotated_image)
    decoded_string = html.unescape(predictions)
    # Parse the JSON string
    decoded_data = ast.literal_eval(decoded_string)
    #decoded_data = json.loads(decoded_string)
    output_folder = 'crop_results'
    crop_and_save_bounding_boxes(annotated_image, decoded_data, output_folder)
    return "image saved"