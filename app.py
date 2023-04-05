from flask import Flask, render_template, request
import openai
import os

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")


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
