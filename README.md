# Chatbot
# Conversational ChatBot

## Overview

This project implements a conversational chatbot that combines **classification** and **sequence-to-sequence (Seq2Seq)** deep learning models. The chatbot is designed to understand user input, classify intent, and generate coherent responses across multiple domains such as greetings, sports, politics, and science.

Originally started as a set of exploratory repositories in late 2023, this project went through multiple iterations during early 2024. In February 2024, all development was consolidated into this final repository to ensure stability, reproducibility, and maintainability.

---

## Features

* **Hybrid Architecture**

  * **Classification model**: Predicts the intent of user queries using a deep neural network.
  * **Seq2Seq model**: Generates context-aware responses with an encoder–decoder LSTM architecture.
* **Multi-domain conversation support**: Trained on YAML-based datasets (e.g., greetings, sports, politics, health, AI, etc.).
* **Graphical User Interface**: Built with Python Tkinter for an interactive user experience.
* **Model Persistence**: Pretrained models and parameters saved as `.h5` and `.pkl` files for reuse.

---

## Tech Stack

* **Programming Language**: Python 3
* **Deep Learning**: TensorFlow / Keras
* **Data Processing**: NumPy, Pandas, YAML
* **Visualization & Debugging**: Matplotlib
* **Interface**: Tkinter GUI

---

## Repository Structure

```
project2/
├── chatbot_gui.py                 # Tkinter-based chatbot interface
├── Conversational ChatBot - Part 1.ipynb  # Classification model development
├── Conversational_ChatBot_Part 2.ipynb    # Seq2Seq model development
├── classification_model.h5        # Trained intent classification model
├── seq2seq_model.h5               # Trained Seq2Seq response generator
├── save_params_classification.pkl # Saved preprocessing params for classification
├── save_params_seq2seq.pkl        # Saved preprocessing params for Seq2Seq
└── data/                          # YAML conversation datasets
```

---

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/conversational-chatbot.git
cd conversational-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Chatbot

```bash
python chatbot_gui.py
```

---

## Training

* **Intent Classification**: Implemented in *Part 1* notebook using tokenization, embeddings, and dense layers.
* **Response Generation**: Implemented in *Part 2* notebook with encoder–decoder LSTM.
* Models trained on domain-specific YAML data files located in `data/`.

---

## Example Interaction

**User**: *Hi, how are you?*
**Bot**: *Hello! I’m doing well, thanks for asking. How about you?*

**User**: *Tell me about AI.*
**Bot**: *Artificial Intelligence is a branch of computer science focused on creating systems that can learn and make decisions.*

---

## Project History

* **Late 2023**: Initial prototypes built in separate repos (classification-only models).
* **January 2024**: Expanded to Seq2Seq-based response generation.
* **February 2024**: Merged prior work into this stable repository, ensuring consistency in preprocessing, model storage, and dataset usage.

---

## Future Improvements

* Expand datasets with more diverse and conversational data.
* Add transformer-based architectures (e.g., BERT, GPT-style models).
* Enhance GUI with modern web-based interface.

---

## License

This project is released under the MIT License.
