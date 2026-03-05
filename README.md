# AI Image Analyze

## Overview

AI Image Analyze is a desktop Python application with a Graphical User Interface (GUI) designed for comprehensive image analysis. It integrates advanced machine learning models to perform tasks such as object detection, image classification, semantic segmentation, and contextual analysis. The application provides an intuitive interface for uploading images and obtaining detailed analysis results.

## Features

*   **Object Detection**: Utilizes the YOLOv10m model to identify and localize various objects within images.
*   **Image Classification**: Determines the type of image using the Sightengine API.
*   **Semantic Segmentation**: Highlights pixels belonging to specific object categories using the DeepLabV3 (ResNet101) model.
*   **Contextual Analysis (CLIP)**: Analyzes the overall context of an image using the OpenAI CLIP model, matching it against a predefined set of categories.
*   **Intuitive GUI**: An easy-to-use graphical interface built with Tkinter for interacting with the application.
*   **Save Results**: Ability to save detailed analysis results to a text file.

## Screenshots

### Screenshot 1: Object Detection and Basic Analysis

This screenshot demonstrates the application's main interface after an image has been loaded and basic object detection has been performed. On the left, the loaded image is displayed with bounding boxes around detected objects. On the right, a summary of detected objects and their counts is presented.

![Object Detection and Basic Analysis](assets/screenshot1.png)

### Screenshot 2: Advanced Analysis with Classification and Detailed Counts

This screenshot shows the results of an advanced analysis. In addition to object detection, the application has performed image classification (e.g., "stove") and provided more detailed object counts. The image on the left still shows bounding boxes, while the right panel offers a comprehensive textual breakdown of the analysis, including classification and individual object counts.

![Advanced Analysis with Classification and Detailed Counts](assets/screenshot2.png)

## Technologies

The project leverages the following key technologies and libraries:

*   **Python**: The primary development language.
*   **Tkinter**: For creating the graphical user interface.
*   **Pillow (PIL)**: For image processing.
*   **PyTorch**: Machine learning framework used for segmentation and CLIP models.
*   **ultralytics (YOLOv10)**: For object detection.
*   **CLIP (OpenAI)**: For contextual image analysis.
*   **Sightengine API**: For image classification (requires an API key).

## Installation

To run AIImageAnalyze on your local machine, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/6MrCrazy6/AIImageAnalyze.git
    cd AIImageAnalyze
    ```

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/macOS
    # venv\Scripts\activate  # For Windows
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note*: Installing `clip` from GitHub might take some time.

4.  **Sightengine API Configuration**:
    For the image classification feature, you will need a Sightengine API key. Register at [Sightengine](https://sightengine.com/) to obtain your `API User` and `API Secret`.
    Open the `detector.py` file and replace the placeholder `"API User"` and `"API Secret"` with your actual keys:
    ```python
    # detector.py
    class Classifier:
        def __init__(self):
            self.api_user = "YOUR_API_USER"
            self.api_secret = "YOUR_API_SECRET"
            self.url = "https://api.sightengine.com/1.0/check.json"
    ```

## Running the Application

After installing all dependencies and configuring the API, you can launch the application:

```bash
python gui_app.py
```

Upon first launch, the application will automatically download the necessary machine learning models (YOLOv10m and DeepLabV3), which may take some time depending on your internet connection speed.

## Usage

1.  **Select Image**: Click the "Choose Image" button to load an image for analysis.
2.  **Basic Analysis**: After selecting an image, the application will display detected objects.
3.  **Advanced Analysis**: Click the "Advanced Analysis" button to perform classification, segmentation, and contextual analysis. Results will be shown in text format, and detected objects will be highlighted with bounding boxes on the image.
4.  **Save Results**: Use the "Save to File" button to save the textual results of the advanced analysis.

## Project Structure

```
AIImageAnalyze/
├── README.md
├── LICENSE
├── assets/
│   ├── screenshot1.png
│   └── screenshot2.png
├── UI/
│   ├── advanced_analysis_button.png
│   ├── choose_img_button.png
│   ├── exit_button.png
│   ├── icon.ico
│   └── save_to_file_button.png
├── detector.py
├── download_assets.py
├── gui_app.py
└── requirements.txt
```

*   `README.md`: This file.
*   `LICENSE`: The MIT License file.
*   `assets/`: Contains screenshots and other media assets.
*   `UI/`: Contains graphical assets for buttons and the application icon.
*   `detector.py`: Contains the logic for object detection, classification, segmentation, and contextual analysis.
*   `download_assets.py`: Script for downloading pre-trained models.
*   `gui_app.py`: The main application file with the graphical interface.
*   `requirements.txt`: List of all required Python libraries.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

Author: **Inna Varchenko**

*   **GitHub**: [6MrCrazy6](https://github.com/6MrCrazy6)
*   **Email**: [devilyumeko42@gmail.com](mailto:devilyumeko42@gmail.com)

If you have any questions or suggestions, please feel free to reach out.
