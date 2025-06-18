# Flower Classification Dashboard

This is a Streamlit dashboard that demonstrates a flower classification model trained on the Oxford 102 Flowers dataset. The model can classify 102 different types of flowers with high accuracy.

## Features

- **Upload and classify** your own flower images
- View **model performance metrics**
- Explore the **learning curves** from model training
- Learn more **about the model architecture**

## Installation

1. Clone this repository or download the files.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Model Information

- Architecture: RexNet 150
- Dataset: Oxford 102 Flowers
- Accuracy: 95.4%
- Number of classes: 102

## Usage

1. Navigate to the "Upload & Classify" page.
2. Upload an image of a flower.
3. View the model's predictions and the class activation map.
4. Explore other pages to learn more about the model's performance and training process.

## Files

- `app.py`: The main Streamlit application
- `flowers_best_model.pth`: The trained model weights
- `cat_to_name.json`: Mapping from class indices to flower names
- `requirements.txt`: Required Python packages
- `validation_plot.png`: Learning curves visualization
# Flowers-Classifier-Web-Dashboard
