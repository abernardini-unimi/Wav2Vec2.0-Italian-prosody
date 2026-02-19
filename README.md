# Wav2Vec2 Italian Prosody (SER Regression)

This project implements a **Speech Emotion Recognition (SER)** system focused on **dimensional regression** (Arousal, Valence, Dominance) for the Italian language. The model is based on the **Wav2Vec2** architecture (specifically `facebook/wav2vec2-large-xlsr-53`) and has been adapted to predict continuous values between 0 and 1 instead of discrete categories.

The model is trained and evaluated using the **Ai4ser dataset**, an Italian emotional speech dataset annotated for both discrete emotions and dimensional values (https://huggingface.co/datasets/sirsalvo72/AI4SER).

## 🚀 Features

* **Base Model**: `facebook/wav2vec2-large-xlsr-53` (optimized for multilingual speech, including Italian).
* **Dataset**: Trained on **Ai4ser** (Artificial Intelligence for Speech Emotion Recognition), a dataset of Italian utterances by multiple speakers.
* **Task**: Multi-output regression (Arousal, Valence, Dominance).
* **Loss Function**: Concordance Correlation Coefficient (CCC), ideal for capturing the temporal dynamics and variance of emotions.
* **Output**: 3 float values normalized in the [0, 1] range via a final Sigmoid activation layer.

---

## 📂 Project Structure

* `run_wav2vec_clf.py`: Main script for training and evaluation.
* `src/models.py`: Contains the model architecture with the regression head and CCC Loss.
* `src/collator.py`: Handles data batching and dynamic padding.
* `prediction.py`: Get prediction to your audio file.

---

## 🛠️ Installation

Ensure you have a virtual environment active and install the dependencies:

```bash
pip install -r requirements.txt

```

---

## 🏋️ Training

To start training on the Italian dataset, run the following command in your terminal:

```jupyter
!python run_wav2vec_clf.py \
    --model_name_or_path facebook/wav2vec2-large-xlsr-53 \
    --train_file csv/train.csv \
    --validation_file csv/test.csv \
    --input_column path \
    --target_column arousal \
    --output_dir path/to/your/directory \
    --num_train_epochs 30 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-4 \
    --freeze_feature_extractor \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --load_best_model_at_end \
    --bf16 \
    --do_train \
    --do_eval \
    --delimiter comma

```

*Note: If you are using an NVIDIA GPU, ensure CUDA drivers are properly configured to accelerate the process.*

---

## 📈 Monitored Metrics

During training, the script automatically calculates:

* **CCC Mean**: Average of the correlation coefficients for the three dimensions.
* **CCC Arousal / Valence / Dominance**: Specific evaluation for each emotional axis.
* **MSE**: Standard Mean Squared Error.

---

## 📝 Technical Notes

* **Ai4ser Data**: The dataset provides mean values for Valence, Dominance, and Arousal on a scale from -3 to +3.
* **Resampling**: All audio files are automatically resampled to **16,000 Hz** during preprocessing.

---

## ⚖️ License

This project is based on the [Soxan](https://github.com/m3hrdadfi/soxan) repository and is distributed under the Apache 2.0 License.
