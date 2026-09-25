from newspaper import Article
import random
import string
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import warnings

warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Create Article object
article = Article(
    "https://en.wikipedia.org/wiki/Chronic_kidney_disease"
)

article.download()
article.parse()

corpus = article.text

# Store article text
text = corpus

# Split article into individual sentences
sentence_list = nltk.sent_tokenize(text)

def bot_response(user_input):
    # Add user's question temporarily to the sentence list
    sentence_list.append(user_input)

    # Convert sentences into numerical vectors
    cm = CountVectorizer().fit_transform(sentence_list)

    # Calculate similarity between user's question and all sentences
    similarity_scores = cosine_similarity(cm[-1], cm)

    # Convert similarity scores into a 1D array
    similarity_scores_list = similarity_scores.flatten()

    # Sort the similarity scores
    index = np.argsort(similarity_scores_list)

    # Remove the user's input from the list
    index = index[:-1]

    response_flag = 0
    bot_response = ""

    # Find a relevant sentence
    for i in range(len(index)):
        if similarity_scores_list[index[i]] > 0.0:
            bot_response = bot_response + " " + sentence_list[index[i]]
            response_flag = 1

            if i > 2:
                break

    # If no relevant sentence is found
    if response_flag == 0:
        bot_response = "I apologize, I don't understand."

    # Remove user's input from sentence list
    sentence_list.remove(user_input)

    return bot_response

def greeting_response(text):
    # Convert user input to lowercase
    text = text.lower()

    # Greetings the bot can give
    bot_greetings = ["hi", "hey", "hello", "hola"]

    # Greetings the user can enter
    user_greetings = ["hi", "hey", "hello", "hola", "greetings"]

    # Check each word in the user's input
    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)

# Print the list of sentences
print(sentence_list)

print(corpus)

print("Doc Bot: I am Doc Bot. For a short time, I will answer your questions.")

# Words used to exit the chatbot
exit_list = ["exit", "see you later", "bye", "quit", "break"]

while True:
    user_input = input()

    # Check if the user wants to exit
    if user_input.lower() in exit_list:
        print("Doc Bot: Chat with you later!")
        break

    else:
        # Check if the user entered a greeting
        if greeting_response(user_input) is not None:
            print("Doc Bot: " + greeting_response(user_input))

        # Otherwise, generate a response from the article
        else:
            print("Doc Bot: " + bot_response(user_input))