# Plant Disease Detection

## Overview
This project implements a deep learning model for detecting plant diseases using image classification techniques. The model is designed to classify rice plant leaves as either healthy or unhealthy and further classify unhealthy leaves into specific disease categories.

## Files
- `plant_disease_detection.ipynb`: This Jupyter Notebook contains the implementation of the plant disease detection model. It includes data preparation, model training, and image classification.
- `evaluation.ipynb`: This Jupyter Notebook evaluates the performance of the trained models, providing validation metrics and confusion matrix visualizations.

## Requirements
To run this project, you need the following libraries:
- PyTorch
- torchvision
- transformers
- PIL
- seaborn
- matplotlib
- scikit-learn

You can install the required libraries using pip:
```bash
pip install torch torchvision transformers pillow seaborn matplotlib scikit-learn
```

## Usage
1. **Data Preparation**: Ensure that your dataset is organized in the following structure:
   ```
   Rice 2/
       ├── Healthy/
       └── Unhealthy/
           ├── Brown_Spot/
           ├── Leaf_Blast/
           └── Neck_Blast/
   ```

2. **Training the Model**: Open the `plant_disease_detection.ipynb` notebook and run the cells to train the model. The model will be trained in two stages:
   - Stage 1: Classifies leaves as healthy or unhealthy.
   - Stage 2: Classifies unhealthy leaves into specific diseases.

3. **Evaluating the Model**: After training, open the `evaluation.ipynb` notebook to evaluate the model's performance. This notebook will provide validation loss, accuracy, and confusion matrix visualizations for both stages of classification.

## Results
The evaluation notebook will display the validation loss and accuracy for both the healthy/unhealthy classification and the disease classification stages. Confusion matrices will also be generated to visualize the model's performance.

## Conclusion
This project demonstrates the application of deep learning techniques for plant disease detection, providing a framework for further research and development in agricultural technology.

## License
This project is licensed under the MIT License.
