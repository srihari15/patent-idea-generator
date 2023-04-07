from flask import Flask, render_template, request
import openai
import os
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI API key
openai_key = os.environ.get('OPENAI_API_KEY')
nlu_key = os.environ.get('NLU_API_KEY')

# Set up the authentication
authenticator = IAMAuthenticator(nlu_key)
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2018-03-16',
    authenticator=authenticator
)
natural_language_understanding.set_service_url(
    'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/237a815b-7be7-44db-a73d-1feaef411889')


'''# Define the function to get keywords
def get_keywords(text):
    response = natural_language_understanding.analyze(
        text=text,
        features=Features(keywords=KeywordsOptions(sentiment=False, emotion=False))
    ).get_result()
    keywords = [keyword['text'] for keyword in response['keywords']]
    return keywords'''


# Define function to extract keywords from input text using OpenAI's API
def get_keywords(text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Extract the keywords from the following text: {text}",
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    keywords = response.choices[0].text.split("\n")
    return [keyword.strip() for keyword in keywords if keyword.strip()]


# Define function to generate new patent idea and brief description using OpenAI GPT-3.5 API
def generate_idea(keywords):
    prompt = f"Generate a new patent idea and brief description using the following keywords: {', '.join(keywords)}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.5,
    )
    idea = response.choices[0].text.split("\n")
    idea = [line.strip() for line in idea if line.strip()]
    return " ".join(idea)


# Define the Flask route for the homepage
@app.route('/')
def index():
    return render_template('index.html')


# Define the Flask route for the idea generation page
@app.route('/generate', methods=['POST'])
def generate():
    # Get input text from form
    text = request.form['text']
    # Get keywords from input text
    keywords = get_keywords(text)
    # Generate new patent idea and brief description
    idea = generate_idea(keywords)
    # Pass generated idea to the template
    return render_template('generate.html', idea=idea)


if __name__ == '__main__':
    app.run(debug=True)
