# Malaria Cell Detection (Computer Vision)

This project uses ResNet50 deep learning architecture to automatically classify microscopic blood cell images as malaria-infected or uninfected. The system aims to automate traditional microscopic diagnosis which requires trained professionals and is time-consuming.

The model was trained on the NIH Malaria Cell Images Dataset containing 27,558 blood smear images with equal distribution of parasitized and uninfected cells. The dataset was split into training, validation and test sets with data augmentation including rotation, flipping, brightness adjustments and noise injection.

The training approach used transfer learning with a two-phase process: first freezing ResNet50 base layers to train the classifier head, then fine-tuning the entire model with a lower learning rate. The system achieved 96.54% test accuracy with 0.993 AUC score, 96.30% precision and 96.81% recall.

The model uses an optimal decision threshold of 0.561 and shows strong discriminative power suitable for real-time inference.

Dataset source: [NIH Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

Project documentation: [View full project documentation and demo](https://adittomahmood.vercel.app/project-docs/2)

Live app: [Try the malaria detection system](https://malaria-cell-detection-cv.streamlit.app/)

Connect: [LinkedIn](https://linkedin.com/in/adittomahmood)
