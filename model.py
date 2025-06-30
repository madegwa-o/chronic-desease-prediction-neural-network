import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout, Attention,
    MultiHeadAttention, LayerNormalization, Concatenate,
    Embedding, TimeDistributed, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class ChronicDiseasePredictor:
    def __init__(self, sequence_length=12, n_features=20, n_diseases=5):
        """
        Initialize the Chronic Disease Prediction Model

        Args:
            sequence_length: Number of time steps (e.g., 12 months of records)
            n_features: Number of medical features per time step
            n_diseases: Number of chronic diseases to predict
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_diseases = n_diseases
        self.model = None
        self.scaler = StandardScaler()

        # Define chronic diseases to predict
        self.diseases = [
            'Diabetes_Type2', 'Hypertension', 'CVD',
            'Chronic_Kidney_Disease', 'COPD'
        ]

    def create_model(self, lstm_units=128, gru_units=64, attention_heads=4):
        """
        Create the LSTM/GRU model with attention mechanism
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features), name='medical_records')

        # LSTM layers for temporal pattern learning
        lstm_out = LSTM(lstm_units, return_sequences=True, dropout=0.2,
                        recurrent_dropout=0.2, name='lstm_layer')(inputs)
        lstm_out = BatchNormalization()(lstm_out)

        # GRU layer for additional temporal processing
        gru_out = GRU(gru_units, return_sequences=True, dropout=0.2,
                      recurrent_dropout=0.2, name='gru_layer')(lstm_out)
        gru_out = BatchNormalization()(gru_out)

        # Multi-head attention mechanism
        attention_out = MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=gru_units // attention_heads,
            name='attention_layer'
        )(gru_out, gru_out)

        # Add & Norm
        attention_out = LayerNormalization()(attention_out + gru_out)

        # Global average pooling to get final representation
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_out)

        # Dense layers for final processing
        dense1 = Dense(256, activation='relu', name='dense_1')(pooled)
        dense1 = Dropout(0.3)(dense1)
        dense1 = BatchNormalization()(dense1)

        dense2 = Dense(128, activation='relu', name='dense_2')(dense1)
        dense2 = Dropout(0.2)(dense2)

        # Multi-output classification (one output per disease)
        disease_outputs = []
        for i, disease in enumerate(self.diseases):
            output = Dense(1, activation='sigmoid', name=f'output_{disease}')(dense2)
            disease_outputs.append(output)

        # Create model
        self.model = Model(inputs=inputs, outputs=disease_outputs, name='ChronicDiseasePredictor')

        # Compile with different loss weights if needed
        loss_weights = {f'output_{disease}': 1.0 for disease in self.diseases}

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={f'output_{disease}': 'binary_crossentropy' for disease in self.diseases},
            loss_weights=loss_weights,
            metrics=['accuracy', 'precision', 'recall']
        )

        return self.model

    def prepare_data(self, df):
        """
        Prepare sequential medical data for training

        Expected DataFrame format:
        - patient_id: Patient identifier
        - visit_date: Date of medical visit
        - Features: Various medical measurements/indicators
        - Target diseases: Binary indicators for each chronic disease
        """

        # Sort by patient and date
        df_sorted = df.sort_values(['patient_id', 'visit_date'])

        # Extract feature columns (exclude patient_id, visit_date, and target diseases)
        feature_cols = [col for col in df.columns
                        if col not in ['patient_id', 'visit_date'] + self.diseases]

        # Scale features
        df_scaled = df_sorted.copy()
        df_scaled[feature_cols] = self.scaler.fit_transform(df_sorted[feature_cols])

        # Create sequences
        X, y = [], {disease: [] for disease in self.diseases}

        for patient_id in df_scaled['patient_id'].unique():
            patient_data = df_scaled[df_scaled['patient_id'] == patient_id]

            if len(patient_data) >= self.sequence_length:
                # Create sliding windows
                for i in range(len(patient_data) - self.sequence_length + 1):
                    # Input sequence (features)
                    sequence = patient_data[feature_cols].iloc[i:i + self.sequence_length].values
                    X.append(sequence)

                    # Target (diseases at the end of sequence)
                    target_row = patient_data.iloc[i + self.sequence_length - 1]
                    for disease in self.diseases:
                        if disease in patient_data.columns:
                            y[disease].append(target_row[disease])
                        else:
                            y[disease].append(0)  # Default to no disease if not present

        X = np.array(X)
        y = {disease: np.array(y[disease]) for disease in self.diseases}

        return X, y

    def train_model(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the model
        """
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint('best_chronic_disease_model.h5', save_best_only=True, monitor='val_loss')
        ]

        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict_diseases(self, medical_sequence):
        """
        Predict chronic diseases for a new patient's medical sequence

        Args:
            medical_sequence: Array of shape (sequence_length, n_features)

        Returns:
            Dictionary with disease probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Reshape for prediction
        if len(medical_sequence.shape) == 2:
            medical_sequence = medical_sequence.reshape(1, self.sequence_length, self.n_features)

        # Scale the input
        sequence_scaled = medical_sequence.copy()
        for i in range(sequence_scaled.shape[0]):
            sequence_scaled[i] = self.scaler.transform(sequence_scaled[i])

        # Predict
        predictions = self.model.predict(sequence_scaled)

        # Format results
        results = {}
        for i, disease in enumerate(self.diseases):
            results[disease] = float(predictions[i][0])

        return results

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        """
        predictions = self.model.predict(X_test)

        results = {}
        for i, disease in enumerate(self.diseases):
            y_true = y_test[disease]
            y_pred = predictions[i].flatten()
            y_pred_binary = (y_pred > 0.5).astype(int)

            # Calculate metrics
            auc = roc_auc_score(y_true, y_pred)

            results[disease] = {
                'AUC': auc,
                'predictions': y_pred,
                'true_labels': y_true
            }

            print(f"\n{disease} Results:")
            print(f"AUC: {auc:.4f}")
            print(classification_report(y_true, y_pred_binary))

        return results

    def plot_training_history(self, history):
        """
        Plot training history
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].legend()

        # Sample disease accuracy (first disease)
        disease_key = f'output_{self.diseases[0]}_accuracy'
        val_disease_key = f'val_output_{self.diseases[0]}_accuracy'

        if disease_key in history.history:
            axes[0, 1].plot(history.history[disease_key], label=f'{self.diseases[0]} Accuracy')
            axes[0, 1].plot(history.history[val_disease_key], label=f'Val {self.diseases[0]} Accuracy')
            axes[0, 1].set_title(f'{self.diseases[0]} Accuracy')
            axes[0, 1].legend()

        plt.tight_layout()
        plt.show()


# Example usage and data preparation
def create_sample_data(n_patients=1000, n_visits_per_patient=15):
    """
    Create sample medical data for demonstration
    """
    np.random.seed(42)

    data = []

    for patient_id in range(n_patients):
        n_visits = np.random.randint(12, n_visits_per_patient + 1)

        # Simulate progressive health decline for some patients
        risk_factor = np.random.random()

        for visit in range(n_visits):
            # Simulate medical features that change over time
            age = 45 + visit * 0.5  # Age progression
            bmi = 25 + risk_factor * 5 + np.random.normal(0, 2)
            blood_pressure_sys = 120 + risk_factor * 20 + visit * 2 + np.random.normal(0, 10)
            blood_pressure_dia = 80 + risk_factor * 15 + visit * 1 + np.random.normal(0, 5)
            glucose = 90 + risk_factor * 30 + visit * 3 + np.random.normal(0, 15)
            cholesterol = 180 + risk_factor * 60 + visit * 2 + np.random.normal(0, 20)
            hba1c = 5.0 + risk_factor * 2 + visit * 0.1 + np.random.normal(0, 0.5)

            # Additional features
            features = {
                'patient_id': patient_id,
                'visit_date': f'2020-{visit + 1:02d}-01',
                'age': age,
                'bmi': bmi,
                'systolic_bp': blood_pressure_sys,
                'diastolic_bp': blood_pressure_dia,
                'glucose': glucose,
                'cholesterol': cholesterol,
                'hba1c': hba1c,
                'smoking': np.random.choice([0, 1], p=[0.7, 0.3]),
                'family_history': np.random.choice([0, 1], p=[0.6, 0.4]),
                'exercise_hours': np.random.exponential(2),
                'alcohol_units': np.random.poisson(3),
                'stress_level': np.random.randint(1, 11),
                'sleep_hours': np.random.normal(7, 1.5),
                'heart_rate': np.random.normal(72, 10),
                'weight': bmi * 1.75 ** 2,  # Approximate weight from BMI
                'creatinine': 0.8 + risk_factor * 0.5 + np.random.normal(0, 0.2),
                'hdl_cholesterol': 50 - risk_factor * 10 + np.random.normal(0, 5),
                'ldl_cholesterol': cholesterol * 0.6 + np.random.normal(0, 10),
                'triglycerides': 150 + risk_factor * 100 + np.random.normal(0, 30)
            }

            # Define disease outcomes based on risk factors
            diabetes_risk = (glucose > 126) * 0.3 + (hba1c > 6.5) * 0.4 + risk_factor * 0.3
            hypertension_risk = (blood_pressure_sys > 140) * 0.4 + (blood_pressure_dia > 90) * 0.3 + risk_factor * 0.3
            cvd_risk = (cholesterol > 240) * 0.2 + (blood_pressure_sys > 140) * 0.2 + risk_factor * 0.6
            ckd_risk = (features['creatinine'] > 1.2) * 0.5 + risk_factor * 0.5
            copd_risk = features['smoking'] * 0.6 + risk_factor * 0.4

            # Convert to binary outcomes with some randomness
            features.update({
                'Diabetes_Type2': int(diabetes_risk > 0.5),
                'Hypertension': int(hypertension_risk > 0.4),
                'CVD': int(cvd_risk > 0.6),
                'Chronic_Kidney_Disease': int(ckd_risk > 0.5),
                'COPD': int(copd_risk > 0.4)
            })

            data.append(features)

    return pd.DataFrame(data)


# Main execution example
if __name__ == "__main__":
    print("Creating Chronic Disease Prediction Model...")

    # Create sample data
    print("Generating sample medical data...")
    df = create_sample_data(n_patients=500, n_visits_per_patient=15)

    # Initialize model
    predictor = ChronicDiseasePredictor(sequence_length=12, n_features=18, n_diseases=5)

    # Create model architecture
    print("Building model architecture...")
    model = predictor.create_model(lstm_units=128, gru_units=64, attention_heads=4)

    # Print model summary
    print("\nModel Architecture:")
    model.summary()

    # Prepare data
    print("Preparing sequential data...")
    X, y = predictor.prepare_data(df)

    print(f"Data shape: X={X.shape}")
    for disease in predictor.diseases:
        print(f"{disease}: {len(y[disease])} samples, {np.sum(y[disease])} positive cases")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert y to proper format for train/test split
    y_train_dict = {disease: y[disease][:len(X_train)] for disease in predictor.diseases}
    y_test_dict = {disease: y[disease][len(X_train):] for disease in predictor.diseases}

    print(f"Training data: {X_train.shape[0]} sequences")
    print(f"Test data: {X_test.shape[0]} sequences")

    # Train model (uncomment to actually train)
    # print("Training model...")
    # history = predictor.train_model(X_train, y_train_dict, epochs=50, batch_size=32)

    # Evaluate model (uncomment to evaluate)
    # print("Evaluating model...")
    # results = predictor.evaluate_model(X_test, y_test_dict)

    print("\nModel setup complete! Uncomment training code to start training.")