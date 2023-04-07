import en_core_web_sm
from flask import Flask, render_template, url_for, request
import time
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import nltk
nltk.download('punkt')


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


@app.route('/')
def index():
    return render_template('landing.html')


@app.route('/summariser')
def summariser():
    return render_template('summariser.html')

@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')
    
@app.route('/sentimentresult', methods=['GET', 'POST'])
def sentimentprocess():
    pass

@app.route('/generate')
def generate():
    return render_template('generator.html')
    
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


if __name__ == '__main__':
    app.run(debug=True)
