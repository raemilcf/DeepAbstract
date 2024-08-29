# Import necessary libraries
from transformers import TFBartForConditionalGeneration, BartTokenizer
import tensorflow as tf
import pickle

# Load the summarization model and tokenizer
model_name = "facebook/bart-large-cnn"
model = TFBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Save the model to an .h5 file
model.save_pretrained("bart_large_cnn")


# Save the tokenizer using pickle
import pickle

with open('bart_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

