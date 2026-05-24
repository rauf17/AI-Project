# Race Dataset AI ML Classifier - AI PROJECT🤖

This repository hosts `AI-Project`, a comprehensive initiative exploring fundamental concepts in Artificial Intelligence and Machine Learning through practical model development and data processing.

## Team Members
- RollNumber1: 23I-0591
- RollNumber2: 23I-0662
- Section: C

## Description 🧠

This project serves as a foundational exploration into the realm of AI, focusing on the implementation and comparison of distinct machine learning models (Model A and Model B). It encompasses the entire machine learning pipeline, from raw data preprocessing to model training and inference. Given the associated documentation (`TF-IDF_Student_Manual.pdf` and `AL2002_LabProject.pdf`), it is strongly suggested that this project primarily targets tasks involving text processing or natural language understanding, likely leveraging techniques like TF-IDF for feature extraction. The modular design allows for independent development and evaluation of different AI approaches.

## Installation 🚀

To get this project up and running on your local machine, please follow these steps:

1.  **Clone the Repository** 📥
    ```bash
    git clone https://github.com/your-username/AI-Project.git
    cd AI-Project
    ```

2.  **Create a Virtual Environment** 💻
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv .venv
    ```
    Activate the virtual environment:
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```

3.  **Install Required Dependencies** ✅
    Once your virtual environment is active, install all necessary packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage 💡

The project is structured to allow for clear separation of preprocessing, model training, and model execution.

1.  **Data Preprocessing** ✨
    Before training any model, raw data needs to be processed and transformed into a suitable format.
    To run the preprocessing script:
    ```bash
    python src/preprocessing.py
    # or if there's an alias:
    python src/prepreprocessing.py
    ```
    *Note: There are two preprocessing files, `prepreprocessing.py` and `preprocessing.py`. Please verify which one is the intended main script or if they serve different purposes.*

2.  **Model A Training** ⚙️
    After preprocessing, you can train Model A using its dedicated training script.
    ```bash
    python src/model_a_train.py
    ```

3.  **Model A Execution (Master)** ▶️
    Once Model A is trained, its master script can be used for evaluation, prediction, or further analysis.
    ```bash
    python src/model_a_master.py
    ```

4.  **Model B Execution (Master)** ▶️
    Similarly, Model B's master script facilitates its execution. This model might have its own training process not explicitly listed here, or it might be a pre-trained model.
    ```bash
    python src/model_b_master.py
    ```

## Contributing 🤝

We welcome contributions to this project! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  **Fork the Repository** 🍴
    Fork the `AI-Project` repository to your GitHub account.

2.  **Clone Your Fork** ⬇️
    Clone your forked repository to your local machine:
    ```bash
    git clone https://github.com/your-username/AI-Project.git
    cd AI-Project
    ```

3.  **Create a New Branch** 🌿
    Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    # or for a bug fix:
    git checkout -b bugfix/issue-description
    ```

4.  **Make Your Changes and Commit** ➕
    Implement your changes, ensure tests (if any) pass, and commit your changes with a clear, concise message:
    ```bash
    git commit -m "feat: Add new feature X"
    # or
    git commit -m "fix: Resolve bug in module Y"
    ```

5.  **Push to Your Fork** ⬆️
    Push your new branch to your forked repository on GitHub:
    ```bash
    git push origin feature/your-feature-name
    ```

6.  **Open a Pull Request** 📤
    Go to the original `AI-Project` repository on GitHub and open a pull request from your new branch. Please provide a detailed description of your changes.
