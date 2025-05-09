# Emotion-Aware Speech Recognition System

## Project Overview

This project aims to build an emotion-aware speech recognition system that can detect emotions from speech using a combination of speech-to-text conversion and machine learning-based emotion detection. The system performs several functions:

1. **Speech Recording**: Captures audio for analysis.
2. **Speech-to-Text (STT)**: Uses Google's Speech Recognition API to convert speech into text.
3. **Emotion Detection**: Analyzes the transcribed text for keywords representing emotions, and, if no emotion is detected from text, extracts audio features to predict emotions from the speech itself.
4. **Data Processing and Analysis**: Includes feature extraction from audio, such as duration, RMS energy, spectral centroid, and spectral rolloff.
5. **Evaluation Metrics**: Includes Word Error Rate (WER), language model perplexity, accuracy by accent, and latency metrics for system evaluation.

## Technologies Used

* **Python**: The main programming language used.
* **SpeechRecognition**: For transcribing speech into text.
* **Librosa**: For audio processing and feature extraction.
* **Soundfile**: For saving the recorded audio file.
* **Sounddevice**: For recording the audio in real-time.
* **Joblib**: For loading the pre-trained emotion classification model.
* **Scikit-learn**: For machine learning model and metrics.
* **Matplotlib** and **Seaborn**: For visualizations.
* **TQDM**: For progress bars during data processing.
* **Jiwer**: For computing Word Error Rate (WER).

## Project Structure

```
├── emotion_model.pkl        # Pre-trained emotion prediction model
├── real_record.wav          # Temporary audio file recorded by the user
├── analysis_summary.csv     # Data summary for Power BI export
├── waveform_and_spectrogram.png   # Visualizations of the first audio sample
├── main.py                  # Main Python script with the core logic
└── requirements.txt         # List of required Python packages
```

## Installation

1. Clone this repository to your local machine.

   ```bash
   git clone https://github.com/your-username/emotion-aware-speech-recognition.git
   ```

2. Install the required Python packages.

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that you have the following dependencies installed:

   * **FFmpeg** (for audio processing)
   * **Sounddevice** (for audio recording)

4. Download the [Common Voice dataset](https://commonvoice.mozilla.org/en/datasets) and place it in the `DATA_PATH` directory.

## Usage

1. **Record Audio**: The system records an audio clip of 5 seconds by default.

2. **Transcribe Speech**: The recorded audio is converted into text using Google's Speech Recognition API.

3. **Emotion Detection**: The transcribed text is analyzed for emotional keywords (e.g., "happy," "sad," "angry"). If no emotion is found in the text, the system uses audio features like MFCC to predict emotion from the audio.

4. **Feature Extraction & EDA**: The system processes audio clips from the Common Voice dataset, extracting features such as duration, RMS energy, spectral centroid, and spectral rolloff. Visualizations and data summary are exported for analysis.

5. **Evaluation**: Several metrics are calculated, including Word Error Rate (WER), simulated accuracy for accent groups, and latency.

   ```bash
   python main.py
   ```

## Metrics & Results

* **Word Error Rate (WER)**: Measures the accuracy of the speech-to-text system.
* **Accuracy by Accent**: Evaluates the accuracy of emotion detection for different accent groups (e.g., American, British, Indian).
* **Latency**: The time it takes to process each audio sample (feature extraction time).
* **User Feedback**: Simulated user feedback score based on system performance.
* **Improvement After Adaptation**: The system's performance after adapting to a specific user's voice or accent.

## Data Visualization

* The following plots are generated for exploratory data analysis (EDA):

  * **Audio Duration Distribution**
  * **RMS Energy Distribution**
  * **Sentence Length Distribution**
  * **Waveform and Spectrogram** of sample audio

## Evaluation Metrics

* **WER** (Word Error Rate): A measure of the ASR system's performance.
* **Perplexity**: Simulated from a language model (for future integration).
* **Accuracy by Accent**: Simulated for different accent groups (American, Indian, British, Australian).
* **Improvement After Adaptation**: Simulated performance improvement after adapting the system to different speakers.
* **Latency**: Simulated inference speed during feature extraction.

## Example Output

1. **Speech-to-Text Output**:

   ```
   📝 Transcribed Text: I am happy today!
   ```

2. **Emotion Detection**:

   ```
   🧠 Emotion Detected (from text): JOY
   ```

3. **Feature Extraction**:

   ```
   [INFO] Audio Duration: 3.5 seconds
   [INFO] RMS Energy: 0.23
   [INFO] Spectral Centroid: 1200 Hz
   [INFO] Spectral Rolloff: 0.95
   ```

4. **Metrics**:

   ```
   [METRIC] Average Word Error Rate (WER): 0.12
   [METRIC] Simulated Language Model Perplexity: 38.7
   [METRIC] Accuracy by Accent:
    - American: 89.2%
    - Indian: 75.5%
    - British: 81.3%
    - Australian: 79.0%
   ```

## Future Work

* **Real-time Emotion Detection**: Extend the system to detect emotions in real-time during conversations.
* **Accent Adaptation**: Implement speaker adaptation techniques (e.g., MLLR) to improve accuracy for different accents.
* **Multilingual Support**: Extend the system to work with multiple languages by training on multilingual datasets.
* **Integration with Virtual Assistants**: Integrate the emotion-aware ASR system into virtual assistants for improved user interaction.

## Acknowledgments

* [Mozilla Common Voice](https://commonvoice.mozilla.org) for providing the open-source speech dataset.
* [Librosa](https://librosa.org) for audio processing.
* [Scikit-learn](https://scikit-learn.org) for machine learning tools.

