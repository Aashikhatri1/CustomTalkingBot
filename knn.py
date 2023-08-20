conversation = [
    {
        "question": "hi",
        "response": "Hello! How are you?"
    },
    {
        "question": "I'm fine, How are you?",
        "response": "I'm good too. What can I assist you with today?"
    },
    {
        "question": "Hello! How are you?",
        "response": "I'm fine, thank you."
    },
    {
        "question": "Hi! How are you?",
        "response": "I'm fine, thank you."
    },
    {
        "question": "I'm fine, How are you?",
        "response": "I'm good too. thank you"
    },
    {
        "question": "Who is this?",
        "response": "My name is Alex. I'm a real estate assistant."
    },
    {
        "question": "What's your name?",
        "response": "My name is Alex. I'm a real estate assistant."
    },
    {
        "question": "Who are you?",
        "response": "My name is Alex. I'm a real estate assistant."
    },
    {
        "question": "That was a good one. Do you know any more?",
        "response": "Absolutely! Why did the bicycle stand up by itself? It was two-tired."
    },
    {
        "question": "Haha, that's funny. What do you do in your free time?",
        "response": "I'm always running, waiting for queries. I don't have free time or consciousness like humans!"
    },
    {
        "question": "Can you dream?",
        "response": "No, I can't dream or have emotions. I'm just a program designed to assist with information and tasks."
    },
    {
        "question": "What's the meaning of life?",
        "response": "That's a philosophical question. Many believe the answer is 42, thanks to 'The Hitchhiker's Guide to the Galaxy'."
    },
    {
        "question": "Tell me about yourself.",
        "response": "I'm a virtual assistant based on the GPT-4 architecture, designed to provide information and answer questions to the best of my training!"
    },
    {
        "question": "Do you have any friends?",
        "response": "I don't have feelings or the ability to form relationships, but I'm here to help and answer questions for everyone!"
    },
    {
        "question": "Do you ever feel lonely?",
        "response": "I don't have emotions or feelings, so I don't experience loneliness. I'm here whenever you'd like to chat!"
    }
]

# Training the model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import joblib

# conversation = [...]

# Extract questions and responses
questions = [item['question'] for item in conversation]
responses = [item['response'] for item in conversation]

# Convert questions to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the n_neighbors parameter
knn.fit(X, responses)

def get_response(user_question):
    user_vector = vectorizer.transform([user_question])
    predicted_response = knn.predict(user_vector)
    return predicted_response[0]

# Save the model and vectorizer
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(vectorizer, 'vectorizer_knn.pkl')

# Test
user_question = "Im fine, How are you?"
print(get_response(user_question))

#################################################

# Using the trained model
import joblib

model = joblib.load('knn_model.pkl')
vectorizer = joblib.load('vectorizer_knn.pkl')

def get_response(user_question):
    user_vector = vectorizer.transform([user_question])
    predicted_response = model.predict(user_vector)
    return predicted_response[0]

user_question = "Im fine, How are you?"
print(get_response(user_question))

# https://software.muzychenko.net/trials/vac470.exe
