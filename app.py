
    # Load model and tokenizer
import tensorflow as tf
from transformers import TFBartForConditionalGeneration
import pickle
import re
from utils import util



# Load the model from the saved .h5 file
model = TFBartForConditionalGeneration.from_pretrained("bart_large_cnn")

# Load the tokenizer from the pickle file
with open('bart_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)



# check if is a url or text 
def is_url(string):
    # Regex to check if the string is a URL
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return re.match(regex, string) is not None


# Get user input, could be an url or plain text 
def getSummary(userInput, minWords, maxWords):
    #determine if is text or news url
    if is_url(userInput):
        # Proced to clean extract url, obtain text, before proceding with the summarization 
        userInput = util.obtainTextFromURL(userInput)

    #after getting text from url or text directly apply summarization 
    return summarize(userInput, minWords, maxWords)

# using bart let's ge the text summaryzation 
def summarize(text,min_length =0, max_length=0):

    #calculate the amount of min words and max word the summary should have base on
    # the len of the text, always a 25% of the total len text

    total_words = len(text.split())
    target_summary_length = int(total_words * 0.25)

    if min_length ==0:
        # Set min_length and max_length for the summary
        min_length = max(1, target_summary_length - 5)  # Ensure min_length is at least 1

    if max_length ==0:
        max_length = target_summary_length + 5

    # Tokenize the text
    inputs = tokenizer.encode("summarize: " + text, 
                              return_tensors="tf", 
                              max_length=1024, 
                              truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs,
                                  
                                 max_length=max_length, 
                                 min_length=min_length, 
                                 length_penalty=2.0, 
                                 num_beams=4, 
                                 early_stopping=True, 
                                 no_repeat_ngram_size=3,
                                 decoder_start_token_id=tokenizer.bos_token_id,  # Ensure the decoder starts properly
                                 bad_words_ids=[[tokenizer.pad_token_id]],  # Avoid generating padding tokens
                                 eos_token_id=tokenizer.eos_token_id  # End-of-sequence token to stop at a complete sentence
    
                                 )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary




from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def getSumarization():
    userText = request.args.get('fullText')
    minWords = request.args.get('minWords')
    maxWords = request.args.get('maxWords')
    if minWords == '':
        minWords=40

    if maxWords =='':
        maxWords=100
    



    #get summary 
    result =  getSummary(userText, int(minWords), int(maxWords))

   
    return result


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4800)
    app.run()
