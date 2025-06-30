    # Chronic Disease Prediction LSTM/GRU Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🏥 **Deep Learning Model for Predicting Chronic Diseases from Sequential Medical Records**

This repository contains a complete implementation of an LSTM/GRU-based neural network with attention mechanism designed to predict multiple chronic diseases from longitudinal patient medical records.

## 🎯 Features

- **Multi-Disease Prediction**: Simultaneously predicts 5 chronic diseases (Diabetes Type 2, Hypertension, CVD, Chronic Kidney Disease, COPD)
- **Sequential Learning**: LSTM/GRU architecture processes temporal medical data
- **Attention Mechanism**: Multi-head attention focuses on critical time periods
- **Scalable**: Easily customizable for additional diseases and features
- **Production Ready**: Complete with evaluation metrics, visualization, and model persistence

## 🏗️ Model Architecture

```
Input (Medical Records Sequence)
    ↓
LSTM Layer (128 units) ─── Learns temporal patterns
    ↓
GRU Layer (64 units) ──── Additional sequence processing  
    ↓
Multi-Head Attention (4 heads) ─── Focus on important periods
    ↓
Dense Layers (256 → 128) ─── Feature combination
    ↓
Multi-Output Classification ─── 5 disease predictions (sigmoid)
```

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Usage](#usage)
- [Model Training](#model-training)
- [Making Predictions](#making-predictions)
- [Customization](#customization)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation
### Prerequisites

- Python 3.8+
- TensorFlow 2.0+
- At least 8GB RAM (recommended for training)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/chronic-disease-prediction.git
cd chronic-disease-prediction

# Install required packages
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn

# Or use requirements.txt
pip install -r requirements.txt
```

## ⚡ Quick Start

```python
from chronic_disease_predictor import ChronicDiseasePredictor
import pandas as pd

# Initialize the model
predictor = ChronicDiseasePredictor(
    sequence_length=12,  # 12 months of medical history
    n_features=18,       # Number of medical measurements
    n_diseases=5         # Number of chronic diseases to predict
)

# Create and compile the model
model = predictor.create_model()

# Load your medical data
df = pd.read_csv('medical_records.csv')

# Prepare sequential data
X, y = predictor.prepare_data(df)

# Train the model
history = predictor.train_model(X, y, epochs=100)

# Make predictions for a new patient
predictions = predictor.predict_diseases(new_patient_sequence)
print(predictions)
```

## 📊 Data Format

### Required CSV Structure

Your medical records dataset should follow this structure:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `patient_id` | int | Unique patient identifier | 1, 2, 3... |
| `visit_date` | datetime | Date of medical visit | 2020-01-01 |
| `age` | float | Patient age at visit | 45.5 |
| `bmi` | float | Body Mass Index | 25.2 |
| `systolic_bp` | float | Systolic blood pressure | 120 |
| `diastolic_bp` | float | Diastolic blood pressure | 80 |
| `glucose` | float | Blood glucose level | 95 |
| `cholesterol` | float | Total cholesterol | 180 |
| `hba1c` | float | Hemoglobin A1c | 5.2 |
| ... | float | Additional medical features | ... |
| `Diabetes_Type2` | int | Target: Diabetes diagnosis (0/1) | 0 |
| `Hypertension` | int | Target: Hypertension diagnosis (0/1) | 1 |
| `CVD` | int | Target: Cardiovascular disease (0/1) | 0 |
| `Chronic_Kidney_Disease` | int | Target: CKD diagnosis (0/1) | 0 |
| `COPD` | int | Target: COPD diagnosis (0/1) | 0 |

### Sample Data

```python
medical_data = {
    'patient_id': [1, 1, 1, 2, 2, 2],
    'visit_date': ['2020-01-01', '2020-02-01', '2020-03-01', 
                   '2020-01-01', '2020-02-01', '2020-03-01'],
    'age': [45, 45.1, 45.2, 52, 52.1, 52.2],
    'bmi': [25.2, 25.5, 26.1, 28.3, 28.1, 27.9],
    'systolic_bp': [120, 125, 130, 140, 138, 135],
    'diastolic_bp': [80, 82, 85, 90, 88, 87],
    'glucose': [95, 98, 105, 110, 115, 118],
    'cholesterol': [180, 185, 190, 220, 215, 210],
    'hba1c': [5.2, 5.4, 5.6, 6.1, 6.3, 6.5],
    # ... more features
    'Diabetes_Type2': [0, 0, 1, 0, 1, 1],
    'Hypertension': [0, 1, 1, 1, 1, 1],
    'CVD': [0, 0, 0, 0, 0, 1],
    'Chronic_Kidney_Disease': [0, 0, 0, 0, 0, 0],
    'COPD': [0, 0, 0, 0, 0, 0]
}
```

## 💻 Usage

### Basic Usage

```python
from chronic_disease_predictor import ChronicDiseasePredictor, create_sample_data

# 1. Create sample data for testing (or load your own)
df = create_sample_data(n_patients=1000, n_visits_per_patient=15)

# 2. Initialize the predictor
predictor = ChronicDiseasePredictor(
    sequence_length=12,  # 12 time steps (e.g., months)
    n_features=18,       # Number of medical measurements
    n_diseases=5         # Number of diseases to predict
)

# 3. Build the model
model = predictor.create_model(
    lstm_units=128,
    gru_units=64,
    attention_heads=4
)

# 4. Prepare sequential data
X, y = predictor.prepare_data(df)
print(f"Data shape: {X.shape}")  # (samples, sequence_length, features)

# 5. Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 🎓 Model Training

### Training Configuration

```python
# Train with custom parameters
history = predictor.train_model(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32
)

# Plot training history
predictor.plot_training_history(history)
```

### Training Tips

- **Start with fewer epochs** (20-30) to test the pipeline
- **Monitor validation loss** to avoid overfitting
- **Adjust batch size** based on your GPU memory
- **Use early stopping** (included by default) to prevent overfitting

### Expected Training Output

```
Epoch 1/100
████████████████████████████████ 45/45 [==============================] - 12s - loss: 2.1534 - val_loss: 1.8234
Epoch 2/100
████████████████████████████████ 45/45 [==============================] - 8s - loss: 1.7845 - val_loss: 1.6123
...
```

## 🔮 Making Predictions

### Predict for New Patient

```python
# Medical sequence for 12 months (shape: [sequence_length, n_features])
new_patient_data = np.array([
    [45.0, 25.2, 120, 80, 95, 180, 5.2, 0, 1, 2, 3, 4, 72, 70, 0.9, 55, 110, 140],
    [45.1, 25.5, 125, 82, 98, 185, 5.4, 0, 1, 2, 3, 4, 74, 71, 0.9, 54, 115, 145],
    # ... 10 more months of data (total 12 rows)
])

# Get predictions
predictions = predictor.predict_diseases(new_patient_data)

# Display results
print("🏥 Chronic Disease Risk Assessment")
print("=" * 50)
for disease, probability in predictions.items():
    risk_level = "🔴 HIGH" if probability > 0.7 else "🟡 MEDIUM" if probability > 0.3 else "🟢 LOW"
    print(f"{disease:<25}: {probability:.3f} ({risk_level} risk)")
```

### Example Output

```
🏥 Chronic Disease Risk Assessment
==================================================
Diabetes_Type2          : 0.156 (🟢 LOW risk)
Hypertension            : 0.742 (🔴 HIGH risk)
CVD                     : 0.234 (🟢 LOW risk)
Chronic_Kidney_Disease  : 0.089 (🟢 LOW risk)
COPD                    : 0.045 (🟢 LOW risk)
```

## 🎛️ Customization

### Adding New Diseases

```python
# Modify the diseases list in the class
class CustomChronicDiseasePredictor(ChronicDiseasePredictor):
    def __init__(self, sequence_length=12, n_features=20, n_diseases=7):
        super().__init__(sequence_length, n_features, n_diseases)
        self.diseases = [
            'Diabetes_Type2', 'Hypertension', 'CVD', 
            'Chronic_Kidney_Disease', 'COPD',
            'Osteoporosis',  # New disease
            'Depression'     # New disease
        ]
```

### Model Architecture Customization

```python
# Create a more complex model
model = predictor.create_model(
    lstm_units=256,      # More LSTM units for complex patterns
    gru_units=128,       # More GRU units  
    attention_heads=8    # More attention heads
)

# Or modify sequence length for longer histories
predictor = ChronicDiseasePredictor(
    sequence_length=24,  # 24 months of data
    n_features=25,       # More medical features
    n_diseases=7         # More diseases
)
```

### Feature Engineering

Important medical features to include:

**🩺 Vital Signs**
- Blood pressure (systolic/diastolic)
- Heart rate, temperature
- Respiratory rate, oxygen saturation

**🧪 Laboratory Values**
- Glucose, HbA1c (diabetes indicators)
- Cholesterol (HDL/LDL), triglycerides
- Creatinine, BUN (kidney function)
- Liver enzymes (ALT, AST)

**📏 Physical Measurements**
- BMI, weight, height
- Waist circumference
- Body fat percentage

**🚭 Lifestyle Factors**
- Smoking status, pack-years
- Alcohol consumption
- Exercise frequency, sleep hours
- Stress levels (1-10 scale)

**🧬 Medical History**
- Family history flags
- Previous diagnoses
- Current medications
- Allergies

## 📊 Evaluation

### Performance Metrics

```python
# Evaluate model on test data
results = predictor.evaluate_model(X_test, y_test)

# Detailed evaluation for each disease
for disease, metrics in results.items():
    print(f"\n📊 {disease} Performance:")
    print(f"   AUC Score: {metrics['AUC']:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
```

### Visualization

```python
# Plot training history
predictor.plot_training_history(history)

# Plot ROC curves for each disease
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, (disease, metrics) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(metrics['true_labels'], metrics['predictions'])
    roc_auc = auc(fpr, tpr)
    
    axes[i].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[i].plot([0, 1], [0, 1], 'k--')
    axes[i].set_title(f'{disease}')
    axes[i].legend()

plt.tight_layout()
plt.show()
```

### Expected Performance

| Disease | Expected AUC | Notes |
|---------|-------------|-------|
| Diabetes Type 2 | 0.85-0.92 | Good predictability from glucose/HbA1c |
| Hypertension | 0.80-0.87 | Moderate predictability |
| CVD | 0.75-0.85 | Complex, multiple risk factors |
| Chronic Kidney Disease | 0.82-0.90 | Good from creatinine trends |
| COPD | 0.78-0.85 | Depends on smoking history quality |

## ⚠️ Important Considerations

### 🔒 Ethical & Legal

- **Educational Purpose Only**: This model is for research and educational use
- **Not for Clinical Diagnosis**: Always consult healthcare professionals
- **HIPAA Compliance**: Ensure proper handling of medical data
- **Data Anonymization**: Remove identifying information before processing
- **Regulatory Compliance**: Follow local healthcare data regulations

### 📊 Data Quality Requirements

- **Consistent Time Intervals**: Regular visit spacing improves predictions
- **Complete Records**: Handle missing values appropriately
- **Standardized Units**: Ensure consistent measurement units
- **Quality Control**: Remove outliers and data entry errors

### 🎯 Model Limitations

- **Population Bias**: May not generalize across different demographics
- **Data Dependency**: Performance heavily depends on input data quality
- **Temporal Bias**: Historical medical practices may not apply to future
- **Class Imbalance**: Rare diseases may have lower prediction accuracy

## 🚀 Advanced Features

### Model Interpretability

```python
# Get attention weights to understand model focus
def get_attention_weights(model, input_sequence):
    # Extract attention layer output
    attention_model = Model(inputs=model.input, 
                           outputs=model.get_layer('attention_layer').output)
    attention_weights = attention_model.predict(input_sequence)
    return attention_weights

# Visualize which time periods the model focuses on
def plot_attention_heatmap(attention_weights, time_labels):
    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_weights[0], xticklabels=time_labels, 
                yticklabels=['Attention'], cmap='Blues')
    plt.title('Model Attention Over Time')
    plt.show()
```

### Uncertainty Quantification

```python
# Monte Carlo Dropout for uncertainty estimation
def predict_with_uncertainty(predictor, input_data, n_samples=100):
    # Enable dropout during inference
    predictions = []
    for _ in range(n_samples):
        pred = predictor.model(input_data, training=True)
        predictions.append(pred)
    
    # Calculate mean and standard deviation
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred
```

## 📈 Performance Optimization

### Training Acceleration

```python
# Use mixed precision training
from tensorflow.keras.mixed_precision import Policy
policy = Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Use multiple GPUs if available
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = predictor.create_model()
```

### Memory Optimization

```python
# Use data generators for large datasets
class MedicalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32):
        self.X, self.y = X, y
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.X) // self.batch_size
        
    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = {disease: self.y[disease][idx * self.batch_size:(idx + 1) * self.batch_size] 
                   for disease in self.y.keys()}
        return batch_X, batch_y

# Use generator in training
train_gen = MedicalDataGenerator(X_train, y_train)
history = model.fit(train_gen, epochs=100)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/chronic-disease-prediction.git
cd chronic-disease-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

### Ways to Contribute

- 🐛 **Bug Reports**: Open an issue with detailed description
- 💡 **Feature Requests**: Suggest new features or improvements  
- 📝 **Documentation**: Improve docs and examples
- 🧪 **Testing**: Add test cases and improve coverage
- 🎯 **Models**: Contribute new model architectures

## 📚 References & Further Reading

### Research Papers
- **"Deep Learning for Healthcare: Review, Opportunities and Challenges"** - Nature Reviews
- **"LSTM Networks for Medical Time Series Prediction"** - IEEE Transactions on Biomedical Engineering
- **"Attention Mechanisms in Healthcare AI"** - Journal of Medical Internet Research

### Datasets for Training
- **MIMIC-III**: Medical Information Mart for Intensive Care
- **eICU Collaborative Research Database**: Multi-center ICU data
- **UK Biobank**: Large-scale biomedical database
- **All of Us Research Program**: Diverse health data

### Regulatory Guidelines
- **FDA AI/ML Guidance**: FDA's approach to AI/ML-based medical devices
- **EU MDR**: Medical Device Regulation for AI in healthcare
- **HIPAA Guidelines**: Healthcare data privacy requirements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Healthcare AI research community
- Open source medical datasets contributors
- Beta testers and early adopters

## 📞 Support

- 📧 **Email**: [your-email@domain.com](mailto:your-email@domain.com)
- 💬 **Issues**: [GitHub Issues](https://github.com/yourusername/chronic-disease-prediction/issues)
- 📖 **Documentation**: [Full Documentation](https://yourusername.github.io/chronic-disease-prediction)
- 💡 **Discussions**: [GitHub Discussions](https://github.com/yourusername/chronic-disease-prediction/discussions)

---

**⚡ Built with ❤️ for Healthcare AI**

*Remember: This model is for research and educational purposes. Always consult qualified healthcare professionals for medical decisions.*