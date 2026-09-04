# Explainable AI for Neural Network Interpretability

## Project Overview

This project focuses on the application of Explainable Artificial Intelligence (XAI) techniques to the interpretation of Deep Learning models, with a particular focus on model transparency, error analysis, and interpretability in regulated environments such as banking and finance.

The project uses a DenseNet-121 neural network, pre-trained on ImageNet and adapted to classify handwritten digits from the MNIST dataset.

The main objective is not only to evaluate the predictive performance of the model, but also to investigate how the model reaches its decisions. For this purpose, several XAI techniques are applied to generate saliency maps and identify the image regions that most influence the model's predictions.

Although MNIST is not a financial dataset, it provides a controlled environment in which the interpretability of a neural network can be studied and the methodology can then be conceptually related to AI systems used in banking.

## Objectives

The main objectives of the project are:

- Use a pre-trained Deep Learning model for an image classification task.
- Adapt DenseNet-121 to classify the ten classes of the MNIST dataset.
- Evaluate the predictive performance of the model.
- Apply Explainable AI techniques to the trained model.
- Generate saliency maps showing the regions that influence model predictions.
- Compare different XAI approaches.
- Analyze correctly classified samples.
- Analyze incorrectly classified samples.
- Investigate possible causes of classification errors.
- Identify potential weaknesses in the model's decision-making process.
- Explore how XAI can support model validation and continuous improvement.
- Discuss the relevance of explainability in regulated environments such as banking and finance.

## Dataset

The project uses the MNIST handwritten digit dataset through the `torchvision` library.

MNIST is a dataset of handwritten digits ranging from `0` to `9`.

The dataset contains:

- 60,000 training images.
- 10,000 test images.
- 10 different classes.
- Grayscale images.
- Original image resolution of 28×28 pixels.

The dataset is automatically downloaded using:

```python
torchvision.datasets.MNIST
```

## Data Preprocessing

The original MNIST images are grayscale images with a resolution of 28×28 pixels, while DenseNet-121 was originally designed for RGB images with a larger input resolution.

Therefore, the project applies the following preprocessing pipeline:

1. Resize the images from 28×28 to 224×224 pixels.
2. Convert the grayscale images to three channels.
3. Convert the images to PyTorch tensors.
4. Normalize the images using the ImageNet mean and standard deviation.

The preprocessing pipeline is implemented as follows:

```python
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
```

The normalization values correspond to the standard ImageNet preprocessing commonly used with ImageNet-pre-trained models.

This preprocessing makes the MNIST images compatible with the DenseNet-121 architecture.

## Model

The project uses DenseNet-121 from the `torchvision` library.

The model is initialized with the default ImageNet pre-trained weights:

```python
model = models.densenet121(
    weights=models.DenseNet121_Weights.DEFAULT
)
```

Since the original DenseNet-121 classifier is designed for 1,000 ImageNet classes, it is replaced with a new linear classification layer containing 10 output classes:

```python
model.classifier = nn.Linear(
    model.classifier.in_features,
    10
)
```

The new classifier is therefore adapted to the ten MNIST digit classes.

## Training

The model is trained using the MNIST training set.

The main training parameters are:

| Parameter | Value |
|---|---|
| Model | DenseNet-121 |
| Pre-training | ImageNet |
| Dataset | MNIST |
| Loss Function | Cross Entropy Loss |
| Optimizer | SGD |
| Learning Rate | 0.001 |
| Momentum | 0.9 |
| Batch Size | 64 |
| Number of Epochs | 5 |
| Number of Classes | 10 |

The training device is automatically selected depending on hardware availability:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

This allows the notebook to run either on a CUDA-enabled GPU or on a CPU.

After training, the model weights are saved to:

```text
DenseNet_MNIST.pt
```

## Model Evaluation

After the training phase, the model is evaluated on the MNIST test set.

The evaluation considers:

- Test loss.
- Test accuracy.

The trained model achieves approximately 95% test accuracy.

This result indicates that the model has good predictive performance on the selected classification task.

However, accuracy alone does not explain how the neural network reaches its predictions.

For this reason, the project combines quantitative model evaluation with Explainable AI techniques.

## Explainable AI Techniques

Three XAI techniques are implemented in the project:

- LIME.
- Integrated Gradients.
- Grad-CAM.

Each technique provides a different perspective on the model's decision-making process.

SHAP and Occlusion Maps were also considered as possible XAI approaches, but they were not included in the final implementation in order to limit computational requirements and maintain a manageable notebook complexity.

## LIME

LIME stands for Local Interpretable Model-agnostic Explanations.

It is a model-agnostic technique that provides local explanations for individual predictions.

LIME works by generating perturbed versions of an input and observing how the model's prediction changes. Based on these perturbations, it identifies regions of the image that are particularly relevant to the prediction.

In the first analysis, one example for each digit from `0` to `9` is selected from the test set.

The corresponding LIME explanations are saved in:

```text
lime_results/
```

The generated files follow the naming convention:

```text
lime_digit_0.png
lime_digit_1.png
lime_digit_2.png
...
lime_digit_9.png
```

A more detailed LIME analysis is also performed on an example of the digit `5`.

In this second analysis, both positive and negative contributions are considered by setting:

```python
positive_only=False
```

A larger number of superpixels is also used:

```python
num_features=50
```

This produces a more detailed explanation of the regions that influence the prediction.

## Integrated Gradients

Integrated Gradients is an attribution-based XAI method that estimates how individual input features contribute to a model prediction.

In this project, Integrated Gradients is implemented using the `Captum` library:

```python
from captum.attr import IntegratedGradients
```

The attribution values are calculated for the target class and transformed into a two-dimensional saliency map.

The resulting map is normalized and displayed as a heatmap over the original image.

Integrated Gradients is mainly used in the error analysis phase to investigate which image regions contributed to the model's decision.

The implementation uses:

```python
ig = IntegratedGradients(model)
```

and calculates the attribution using:

```python
ig_attr = ig.attribute(
    input_img,
    target=input_label,
    n_steps=50
)
```

## Grad-CAM

Grad-CAM stands for Gradient-weighted Class Activation Mapping.

It is used to identify spatial regions of an image that are associated with the activations of convolutional layers in a neural network.

The project uses the `LayerGradCam` implementation provided by Captum:

```python
from captum.attr import LayerGradCam
```

The analysis is performed on a deep convolutional layer of DenseNet-121:

```python
model.features.denseblock4.denselayer16.conv2
```

The resulting activation map is resized to 224×224 pixels and overlaid on the original input image.

The Grad-CAM implementation is:

```python
layer_gradcam = LayerGradCam(
    model,
    model.features.denseblock4.denselayer16.conv2
)
```

The resulting heatmap provides a visual indication of the spatial regions associated with the model's internal activations for the selected class.

## Analysis of Correct Predictions

The project includes a qualitative analysis of correctly classified images.

Several examples of the digit `5` are analyzed using:

- The original image.
- Integrated Gradients.
- Grad-CAM.

The objective is to determine whether the regions highlighted by the XAI techniques correspond to meaningful parts of the handwritten digit.

The analysis shows that a correct classification does not necessarily imply that the model is relying exclusively on the most intuitive or semantically meaningful features.

In some cases, the model assigns importance to regions that do not appear to be directly related to the digit itself.

This observation is important because it shows that high predictive accuracy and interpretability are distinct aspects of model evaluation.

Potential future improvements could include:

- Further fine-tuning.
- Data augmentation.
- Additional training data analysis.
- Investigation of potentially misleading visual features.
- More systematic analysis of the model's behavior.

## Error Analysis

A central part of the project is the analysis of incorrectly classified samples.

For each digit from `0` to `9`, the notebook searches for the first test sample that is incorrectly classified.

For each identified error, the following information is analyzed:

- True class.
- Predicted class.
- Prediction probability.
- Original image.
- Integrated Gradients map.
- Grad-CAM map.

The objective is not only to identify that the model made an error, but also to investigate the possible reasons behind the incorrect classification.

The saliency maps can reveal cases where:

- The model focuses on regions that are not representative of the digit.
- Important visual features are ignored.
- The handwritten digit contains ambiguous patterns.
- The model relies on features that may not be optimal.
- The model's attention differs significantly from the expected structure of the digit.

This type of analysis can help identify weaknesses in the model and provide directions for future improvements.

## Interpretation of Saliency Maps

The generated saliency maps should be interpreted as diagnostic and interpretability tools.

They provide information about which input regions are associated with a model's prediction, but they should not automatically be interpreted as a complete causal explanation of the neural network's internal reasoning.

The three techniques used in this project operate according to different principles:

| Technique | Main Principle | Type of Explanation |
|---|---|---|
| LIME | Local perturbations | Local and model-agnostic |
| Integrated Gradients | Feature attribution | Pixel-level attribution |
| Grad-CAM | Layer activations and gradients | Spatial activation map |

Comparing multiple XAI techniques can provide a more complete view of the model's behavior.

If different techniques consistently highlight similar regions, this may provide stronger qualitative evidence that those regions are relevant to the prediction.

Conversely, significant differences between explanations can indicate that the interpretation is sensitive to the selected XAI method.

## Comparison of XAI Techniques

### LIME

LIME provides a local and model-agnostic explanation.

Its main advantage is that it does not require access to the internal structure of the model.

In this project, LIME identifies image regions whose perturbation has a significant effect on the model's prediction.

### Integrated Gradients

Integrated Gradients provides feature-level attribution by evaluating the contribution of input features along a path from a baseline to the input.

It can provide a detailed visualization of the contribution of individual pixels.

### Grad-CAM

Grad-CAM uses gradients and feature activations from a convolutional layer.

It produces a spatial heatmap that highlights regions associated with the activation of the selected layer.

### Overall Comparison

The combination of the three techniques is useful because each method provides a different type of explanation.

LIME focuses on local perturbation-based behavior, Integrated Gradients provides input-level attribution, and Grad-CAM provides a higher-level spatial interpretation based on convolutional activations.

## Results

The trained DenseNet-121 achieves approximately 95% accuracy on the MNIST test set.

The quantitative results indicate that the model performs well on the classification task.

However, the main purpose of the project is to complement this performance evaluation with an interpretability analysis.

The XAI techniques provide additional information about:

- Which regions of the image influence the prediction.
- Which visual features appear to be relevant.
- How the model behaves on correct predictions.
- How the model behaves on incorrect predictions.
- Which potential weaknesses can be identified through the explanations.

The error analysis indicates that, in some cases, the model focuses on regions that are not strongly representative of the target digit or fails to sufficiently consider relevant visual characteristics.

These observations suggest that additional fine-tuning, data augmentation, or a more systematic dataset analysis could potentially improve model robustness.

## Connection to the Banking Context

The original motivation for the project is related to the need for transparency and interpretability in AI systems used in regulated environments, particularly banking and finance.

MNIST is used as a simplified experimental dataset because it allows the XAI methodology to be investigated in a controlled setting.

The same general methodology could conceptually be applied to more complex AI systems used in banking.

Examples of applications where Machine Learning models may be used include:

- Credit risk assessment.
- Fraud detection.
- Anomaly detection.
- Financial document classification.
- Customer-related decision support.
- Transaction classification.
- Risk monitoring.

In such applications, simply producing a prediction may not be sufficient.

An organization may also need to understand:

- Which input features influenced the decision.
- Why a particular prediction was produced.
- Where the model may be making systematic errors.
- Whether the model is relying on unexpected patterns.
- How the model behaves under different input conditions.

XAI techniques can therefore support:

- Model transparency.
- Model validation.
- Model monitoring.
- Error investigation.
- Model governance.
- Risk management.
- Communication with technical and non-technical stakeholders.

However, XAI techniques alone do not guarantee regulatory compliance.

Explainability should be considered as one component of a broader framework that also includes model governance, documentation, validation, risk management, monitoring, and compliance processes.

## Fully Explainable Alternative System

As an optional extension, the project considers a completely interpretable alternative to the Deep Learning model.

Instead of using a complex neural network, handwritten digits could be represented using explicitly interpretable geometric characteristics.

Possible features include:

- Number of segments.
- Number of endpoints.
- Presence of curves.
- Number and position of strokes.
- Area occupied by the digit.
- Height-to-width ratio.
- Relative position of the strokes.
- Presence of specific geometric structures.

These features could be provided to an interpretable classifier such as a Decision Tree or a rule-based system.

A decision could then be represented through explicit rules.

For example, the classifier could determine a digit based on a combination of geometric conditions.

This approach would make the reasoning process much easier to inspect and explain.

However, such an interpretable system could potentially achieve lower predictive performance than a deep neural network.

This highlights a fundamental challenge in Explainable AI: the trade-off between predictive performance and interpretability.

## Project Structure

The repository can be organized as follows:

```text
Explainable-AI-for-Neural-Network-Interpretability/
│
├── XAI_project.ipynb
├── DenseNet_MNIST.pt
│
├── lime_results/
│   ├── lime_digit_0.png
│   ├── lime_digit_1.png
│   ├── lime_digit_2.png
│   ├── ...
│   └── lime_digit_9.png
│
├── data/
│
└── README.md
```

### Main Files

#### `XAI_project.ipynb`

The main Jupyter/Google Colab notebook containing the complete project implementation.

It includes:

- Dataset loading.
- Data preprocessing.
- DenseNet-121 configuration.
- Model training.
- Model evaluation.
- LIME explanations.
- Integrated Gradients.
- Grad-CAM.
- Correct prediction analysis.
- Classification error analysis.

#### `DenseNet_MNIST.pt`

The trained DenseNet-121 model weights generated after the training phase.

#### `lime_results/`

Directory containing the LIME visualizations generated for the ten digit classes.

#### `data/`

Directory used by `torchvision` to store the downloaded MNIST dataset.

## Requirements

The project is implemented in Python and uses the following main libraries:

- Python.
- PyTorch.
- Torchvision.
- NumPy.
- Matplotlib.
- Scikit-learn.
- Scikit-image.
- LIME.
- Captum.

The main dependencies can be installed with:

```bash
pip install torch torchvision numpy matplotlib scikit-learn scikit-image lime captum
```

## Running the Project

Clone the repository:

```bash
git clone https://github.com/andrealuigipala-del/Explainable-AI-for-Neural-Network-Interpretability.git
```

Move into the project directory:

```bash
cd Explainable-AI-for-Neural-Network-Interpretability
```

Install the required dependencies:

```bash
pip install torch torchvision numpy matplotlib scikit-learn scikit-image lime captum
```

Open the notebook:

```text
XAI_project.ipynb
```

The notebook can be executed using Jupyter Notebook, JupyterLab, or Google Colab.

The original notebook was developed in Google Colab.

During execution, the notebook performs the following steps:

1. Downloads the MNIST dataset.
2. Applies the preprocessing pipeline.
3. Loads DenseNet-121 with ImageNet pre-trained weights.
4. Replaces the original classifier with a 10-class classifier.
5. Trains the model for five epochs.
6. Evaluates the model on the MNIST test set.
7. Saves the trained model weights.
8. Generates LIME explanations.
9. Analyzes selected correctly classified samples.
10. Generates Integrated Gradients maps.
11. Generates Grad-CAM maps.
12. Identifies incorrectly classified samples.
13. Generates explanations for the identified errors.

## Hardware

The project can run on both CPU and GPU.

The notebook automatically checks whether CUDA is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

A CUDA-enabled GPU is recommended because DenseNet-121 training and XAI explanation generation can be computationally demanding.

In particular, LIME can require significant computational resources because it relies on repeated model evaluations on perturbed inputs.

## Limitations

The project has several limitations.

### Dataset Limitation

MNIST is a relatively simple image dataset and does not represent the complexity of real-world banking or financial data.

Therefore, the results obtained in this project should be interpreted primarily as a demonstration of the XAI methodology rather than as a direct validation of an AI system used in banking.

### Model Limitation

DenseNet-121 was originally pre-trained on ImageNet, while MNIST consists of handwritten grayscale digits.

Although the images are converted to three channels and resized to 224×224 pixels, there remains a significant domain difference between ImageNet and MNIST.

### XAI Limitation

Saliency maps are useful for interpretation and diagnosis, but they do not necessarily provide a complete causal explanation of a neural network's behavior.

Different XAI methods may also produce different explanations for the same prediction.

Therefore, interpretations should ideally consider multiple techniques rather than relying on a single saliency method.

### Computational Limitation

LIME can be computationally expensive because it requires multiple evaluations of the model on perturbed versions of the input.

For this reason, SHAP and Occlusion Maps were not included in the final implementation.

## Future Developments

The project can be extended in several directions.

Possible future developments include:

- Implement SHAP.
- Implement Occlusion Maps.
- Perform a quantitative comparison between XAI techniques.
- Analyze a larger number of correctly classified samples.
- Analyze a larger number of incorrectly classified samples.
- Perform a systematic error analysis across all digit classes.
- Investigate the stability of explanations.
- Compare explanations generated by different XAI methods.
- Apply data augmentation.
- Perform additional fine-tuning of DenseNet-121.
- Compare DenseNet-121 with other neural network architectures.
- Use datasets that are more representative of financial applications.
- Develop a fully interpretable rule-based classifier.
- Compare the performance and interpretability of neural and symbolic approaches.
- Investigate the integration of XAI into a broader model governance framework.

## Conclusion

This project demonstrates how Explainable AI techniques can complement traditional performance evaluation by providing additional insight into the behavior of a Deep Learning model.

DenseNet-121 achieves approximately 95% accuracy on the MNIST classification task, demonstrating good predictive performance.

However, the XAI analysis shows that predictive accuracy alone is not sufficient to understand the behavior of a neural network.

LIME, Integrated Gradients, and Grad-CAM provide different perspectives on the model's decisions and allow relevant image regions to be visualized.

The analysis of both correct and incorrect predictions makes it possible to investigate how the model responds to different visual patterns and identify cases where it may rely on unexpected or non-optimal features.

From a banking perspective, this methodology illustrates how explainability can support model validation, error analysis, transparency, governance, and risk management.

The project therefore highlights the importance of combining model performance with interpretability when developing AI systems intended for environments where transparency and accountability are particularly important.

## Repository

GitHub repository:

https://github.com/andrealuigipala-del/Explainable-AI-for-Neural-Network-Interpretability
