# PlantAnalytica 🌿

**PlantAnalytica** is an advanced AI-powered application designed to identify plant species and detect diseases instantly. By leveraging state-of-the-art deep learning models, it empowers farmers, gardeners, and botanists to maintain healthy crops and plants through timely diagnosis and expert treatment prescriptions.

![PlantAnalytica Hero](https://images.unsplash.com/photo-1501004318641-b39e6451bec6?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80)

## 🚀 Features

-   **Instant Plant Identification**: Identify plant species with high accuracy using our trained AI models.
-   **Disease Detection**: Detect over 77 different plant diseases and health conditions.
-   **Expert Prescriptions**: Get tailored treatment advice and care tips for every identified disease.
-   **Visual Analysis**: View the analyzed image with bounding boxes showing detected leaves and areas of interest.
-   **User-Friendly Interface**: A clean, modern web interface inspired by top citizen science platforms.

## 🛠️ How It Works

PlantAnalytica uses a multi-stage AI pipeline to process and analyze plant images:

1.  **Leaf Detection (YOLO)**: The system first uses a YOLO (You Only Look Once) model to detect and locate leaves within the uploaded image. This ensures that the analysis focuses on the relevant plant parts.
2.  **Preprocessing**: The detected leaf area is cropped and preprocessed (resized, normalized) to match the input requirements of the classification models.
3.  **Species & Disease Classification**:
    -   **Plant vs. Non-Plant**: Verifies if the image contains a plant.
    -   **Species Identification**: Determines the specific plant species.
    -   **Health Assessment**: Analyzes the leaf for signs of disease.
4.  **Prescription Generation**: Based on the identified disease class, the system retrieves a specific treatment plan from its comprehensive database.

![AI Analysis](https://images.unsplash.com/photo-1550989460-0adf9ea622e2?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80)

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/MananAli05/PlantAnalytica.git
    cd PlantAnalytica
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Model Files**:
    
    Due to GitHub's file size limits, the trained model files (`.h5` and `.pt`) are not included in this repository. You need to download them separately:
    
    - `Plant_NonPlant_Model1.h5` (231 MB)
    - `Plant_Name_Detection.h5` (63 MB)
    - `Plant_Health_detection_Model3.h5` (155 MB)
    - `best.pt` (6 MB)
    
    Place these files in the `plant/static/` directory.
    
    > **Note**: Contact the repository owner or check the releases section for model file downloads.

4.  **Run the application**:
    ```bash
    python manage.py runserver
    ```

5.  **Access the app**:
    Open your browser and go to `http://127.0.0.1:8000/`.

## 🧠 Models Used

-   **YOLOv8**: For robust object detection (leaf localization).
-   **CNN (Convolutional Neural Networks)**: Custom trained models for:
    -   Plant/Non-Plant Binary Classification
    -   Species Classification (Multi-class)
    -   Health/Disease Classification

## 📸 Screenshots

### Home Page
![Home Page](static/home.png)

### Analysis Results
![Results](static/result.png)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
