# Deep Abstract
**Deep Abstract is a powerful text summarization tool designed to generate concise summaries from large, lengthy articles. By leveraging advanced NLP techniques and deep neural networks, the project aims to help users quickly extract key information, concepts, and important details from vast amounts of textual data, reducing the time spent on understanding content.** 

Table of Contents
1. Features
2. Installation
3. Models and Techniques
4. Contact


## Features
- **Summarizes** large articles into concise, key-point summaries.
- **Uses** advanced NLP techniques and deep learning models like LSTM, GRUs, BERT, and GPT.
- **Supports** multiple sources, including Canadian News outlets like Toronto News.
- **Provides** an easy-to-use API via Flask for integration with different systems and environments.

## Installation

- **Prerequisites**
- Python 3.8+
- Flask
- Docker (optional, for containerized deployment)

### Steps
1. Clone the repository:
```bash
git clone https://github.com/raemilcf/deep-abstract.git
```
2. Navigate to the project directory:
```bash
cd deep-abstract
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the application:
```bash
python app.py
```
5. Usage
After starting the Flask server, you can use the API to generate summaries.

## Models and Techniques
- Deep Abstract uses a variety of NLP models and techniques:
BART (Bidirectional and Auto-Regressive Transformers) is a powerful model used for text generation tasks.
It combines the benefits of BERT (bidirectional context) and GPT (auto-regressive generation), making it highly effective for abstractive summarization.

Deep Abstract provides access to BART for users who require sophisticated summaries, especially for content that needs rephrasing or reorganization.

## Contact
For any inquiries or feedback, please reach out to Raemil Corniel at raemilcorniel@hotmail.com
