from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import FileSystemStorage
from ultralytics import YOLO
import numpy as np
import cv2
import os
import json
import tempfile
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
REPO_ID = "AManan05/PlantAnalytica-Models"
#Plant vs Non-Plant Model
nonplant_path = hf_hub_download(repo_id=REPO_ID,filename="Plant_NonPlant_Model1.h5")
plant_nonplant_model = load_model(nonplant_path)
#Plant Name Detection Model
name_path = hf_hub_download(repo_id=REPO_ID,filename="Plant_Name_Detection.h5")
plant_name_model = load_model(name_path)
#Plant Health / Disease Model
health_path = hf_hub_download(repo_id=REPO_ID,filename="Plant_Health_detection_Model3.h5")
health_model = load_model(health_path)
#YOLO Model
yolo_path = hf_hub_download(repo_id=REPO_ID,filename="best.pt")
yolo_model = YOLO(yolo_path)
def load_class_mapping():
    try:
        mapping_path = r'D:\PlantDetectionApp\myproject\plant\static\class_mapping.json'
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        return mapping
    except Exception as e:
        print(f"Error Loading Mapping File: {e}")
        return {}
class_mapping = load_class_mapping()
def home(request): 
    return render(request, 'home.html')
def detect_and_select_best_leaf(img_path):
    try:
        results = yolo_model(img_path)
        orig_image = cv2.imread(img_path)
        if orig_image is None:
            return None, None, []
        vis_image = orig_image.copy()
        all_detections = []
        best_crop = None
        best_score = -1
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    crop = orig_image[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    area = (x2 - x1) * (y2 - y1)
                    aspect_ratio = crop.shape[1] / crop.shape[0] 
                    ideal_aspect_ratio = 1.2
                    aspect_score = 1.0 - min(abs(aspect_ratio - ideal_aspect_ratio) / ideal_aspect_ratio, 1.0)
                    img_area = orig_image.shape[0] * orig_image.shape[1]
                    size_ratio = area / img_area
                    size_score = min(size_ratio * 10, 1.0) 
                    quality_score = (confidence * 0.5 + size_score * 0.3 + aspect_score * 0.2)
                    detection_info = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'area': area,
                        'quality_score': quality_score,
                        'crop': crop
                    }
                    all_detections.append(detection_info)
                    color = (0, 255, 0)  
                    if quality_score > best_score:
                        color = (0, 0, 255)  
                    
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis_image, f'Leaf {i+1}: {confidence:.2f}', 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if quality_score > best_score:
                        best_score = quality_score
                        best_crop = crop
        cv2.putText(vis_image, f'Detected {len(all_detections)} leaves', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, 'Red box = Selected leaf', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return best_crop, vis_image, all_detections
        
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        return None, None, []

def save_temp_image(image, suffix='.jpg'):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    cv2.imwrite(temp_file.name, image)
    return temp_file.name
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)
def predict_plant_detection(img_array):
    pred = plant_nonplant_model.predict(img_array, verbose=0)[0][0]
    is_plant = pred > 0.5
    confidence = pred if is_plant else (1 - pred)
    return is_plant, float(confidence)
def predict_plant_species(img_array):
    preds = plant_name_model.predict(img_array, verbose=0)[0]
    top5_idx = preds.argsort()[-5:][::-1]  
    top5_conf = preds[top5_idx] * 100
    species_list = []
    for idx, conf in zip(top5_idx, top5_conf):
        class_name = class_mapping.get(str(idx), f"Unknown_{idx}")
        species_list.append({
            "index": int(idx),
            "name": class_name,
            "confidence": float(conf)
        })
    return species_list

def predict_plant_health(img_array):
    pred = health_model.predict(img_array, verbose=0)[0][0]
    is_healthy = pred > 0.5
    confidence = pred if is_healthy else (1 - pred)
    return {
        'status': 'healthy' if is_healthy else 'diseased',
        'confidence': float(confidence),
        'color': 'green' if is_healthy else 'red'
    }

def load_prescriptions():
    try:
        prescription_path = r'D:\PlantDetectionApp\myproject\plant\static\prescriptions.json'
        with open(prescription_path, 'r') as f:
            prescriptions = json.load(f)
        return prescriptions
    except Exception as e:
        print(f"Error Loading Prescriptions File: {e}")
        return {}

prescriptions_data = load_prescriptions()

def get_prescription(health_status, species_name):
    if health_status == 'healthy':
        return "No treatment needed. Plant is healthy. Maintain proper watering and sunlight."
    
    if species_name in prescriptions_data:
        return prescriptions_data[species_name]
    
    species_lower = species_name.lower()
    if 'bacterial' in species_lower:
        return "Apply copper-based bactericide. Remove infected leaves. Improve air circulation and avoid overhead watering."
    if 'fungal' in species_lower or 'mold' in species_lower:
        return "Apply fungicide spray. Remove affected leaves. Ensure proper spacing and avoid wet foliage."
    if 'virus' in species_lower or 'curl' in species_lower:
        return "Remove and destroy infected plants. Control insect vectors (aphids, whiteflies). Use virus-resistant varieties."
    if 'blight' in species_lower:
        return "Apply appropriate fungicide. Remove infected parts immediately. Improve air circulation and reduce humidity."
    if 'mite' in species_lower:
        return "Apply miticide. Increase humidity. Remove heavily infested leaves. Introduce beneficial insects."
    if 'spot' in species_lower or 'blotch' in species_lower:
        return "Apply fungicide. Remove spotted leaves. Avoid overhead watering. Improve air circulation."
    if 'powdery' in species_lower:
        return "Apply sulfur-based fungicide. Improve air circulation. Avoid overhead watering. Remove infected leaves."
    return "Apply appropriate treatment based on symptoms. Remove infected leaves. Ensure proper growing conditions. Consult agricultural expert for specific diagnosis."

class PlantDetectionView(APIView):
    def post(self, request):
        temp_files = []  
        try:
            if 'image' not in request.FILES:
                return Response({'error': 'No image provided'}, status=400)
            file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            original_image_path = fs.path(filename)
            temp_files.append(original_image_path)
            #YOLO Detection
            best_crop, visualization_image, all_detections = detect_and_select_best_leaf(original_image_path)
            if best_crop is None:
                cleanup_files(temp_files)
                return Response({
                    'is_plant': False,
                    'message': "No leaves detected in the image. Please upload an image with clear plant leaves."
                })
            vis_image_path = save_temp_image(visualization_image, '_visualization.jpg')
            temp_files.append(vis_image_path)
            best_crop_path = save_temp_image(best_crop, '_best_crop.jpg')
            temp_files.append(best_crop_path)
            img_array = preprocess_image(best_crop_path)
            if img_array is None:
                cleanup_files(temp_files)
                return Response({'error': 'Could not process cropped image'}, status=400)
            
            is_plant, plant_conf = predict_plant_detection(img_array)
            if not is_plant:
                cleanup_files(temp_files)
                return Response({
                    'is_plant': False,
                    'plant_confidence': plant_conf,
                    'message': "The detected region doesn't appear to be a plant leaf."
                })
            species_top5 = predict_plant_species(img_array)
            health_result = predict_plant_health(img_array)
            top_species_name = species_top5[0]['name'] if species_top5 else "Unknown"
            prescription = get_prescription(health_result['status'], top_species_name)
            response_data = {
                'is_plant': True,
                'plant_confidence': plant_conf,
                'species_top5': species_top5,
                'health_assessment': health_result,
                'prescription': prescription,
                'detection_info': {
                    'leaves_detected': len(all_detections),
                    'selected_leaf_confidence': all_detections[0]['confidence'] if all_detections else 0,
                    'selected_leaf_quality_score': all_detections[0]['quality_score'] if all_detections else 0
                }
            }
            _, buffer = cv2.imencode('.jpg', visualization_image)
            import base64
            visualization_base64 = base64.b64encode(buffer).decode('utf-8')
            response_data['visualization_image'] = f"data:image/jpeg;base64,{visualization_base64}"
            _, crop_buffer = cv2.imencode('.jpg', best_crop)
            crop_base64 = base64.b64encode(crop_buffer).decode('utf-8')
            response_data['selected_crop_image'] = f"data:image/jpeg;base64,{crop_base64}"
            cleanup_files(temp_files)
            return Response(response_data)
        except Exception as e:
            cleanup_files(temp_files)
            return Response({'error': str(e)}, status=500)
def cleanup_files(file_paths):
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error Cleaning Up File {file_path}: {e}")