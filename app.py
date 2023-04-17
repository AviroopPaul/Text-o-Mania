import en_core_web_sm
import time
import spacy
import nltk
import torch
from flask import Flask, render_template, url_for, request

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer

from bs4 import BeautifulSoup

from urllib.request import urlopen, Request

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('vader_lexicon')

nlp = en_core_web_sm.load()

app = Flask(__name__)


def lex_summary(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


def luhn_summary(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    summarizer_luhn = LuhnSummarizer()
    summary_luhn = summarizer_luhn(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary_luhn]
    result = ' '.join(summary_list)
    return result


def lsa_summary(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    summarizer_lsa = LsaSummarizer()
    summary_lsa = summarizer_lsa(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary_lsa]
    result = ' '.join(summary_list)
    return result

# calculating reading time


def readingTime(mytext):
    totalwords = len([token.text for token in nlp(mytext)])
    estimatedtime = totalwords/200.0
    return estimatedtime

#FOR TEXT GENERATION
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generation(prompt, max_length):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, do_sample=True, top_p=0.95, top_k=60)
    # Print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # generated_text += '.'
    return generated_text


@app.route('/')
def index():
    return render_template('landing.html')


@app.route('/summariser')
def summariser():
    return render_template('summariser.html')

@app.route('/process', methods=['GET', 'POST'])
def process():
    start = time.time()
    if request.method == 'POST':
        inputText = request.form['inputText']
        modelChoice = request.form['modelChoice']
        finalReadingTime = readingTime(inputText)
        if modelChoice == 'default':
            finalSummary = lex_summary(inputText)
        elif modelChoice == 'lex_summarizer':
            finalSummary = lex_summary(inputText)
        elif modelChoice == 'luhn_summarizer':
            finalSummary = luhn_summary(inputText)
        elif modelChoice == 'lsa_summarizer':
            finalSummary = lsa_summary(inputText)

    summaryReadingTime = readingTime(finalSummary)
    end = time.time()
    finalTime = end-start
    return render_template('result.html', ctext=inputText, finalReadingTime=finalReadingTime, summaryReadingTime=summaryReadingTime, finalSummary=finalSummary, modelSelected=modelChoice)


@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')
    
@app.route('/sentimentresult', methods=['GET', 'POST'])
def sentimentprocess():
    if request.method=='POST':
        sentence=str(request.form['sentence'])
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(sentence)
        compound_score = scores['compound']
        
        if compound_score > 0:
            return render_template('sentimentResult.html', senti='Positive' , input=sentence )
        elif compound_score < 0:
            return render_template('sentimentResult.html', senti='Negative' , input=sentence )
        else:
            return render_template('sentimentResult.html', senti='Neutral' , input=sentence )

@app.route('/generate')
def generate():
    return render_template('generator.html')

@app.route('/generatorResult', methods=["GET", "POST"])
def generatorResult():
    if request.method=='POST' or request.method=='GET':
        prompt=str(request.form['prompt'])
        length=int(request.form['length'])
        generatedText=generation(prompt, length)
        
        return render_template('generatorResult.html', prompt=prompt, generatedText=generatedText)


if __name__ == '__main__':
    app.run(debug=True)
