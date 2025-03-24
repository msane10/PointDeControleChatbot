import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import streamlit as st
import spacy

# Charger le modèle français de spaCy pour le traitement NLP
nlp = spacy.load("fr_core_news_sm")

# Chargement du fichier contenant les questions réponses
def charger_fichier(chemin_fichier):
    with open(chemin_fichier, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

# Chargement du texte
data = charger_fichier("data_science_qa.txt")

# Séparation en questions et réponses
qa_pairs = {}
sections = data.split("\n\n")
for section in sections:
    lines = section.strip().split("\n")
    for i in range(len(lines) - 1):
        if lines[i].startswith("Q:") and lines[i + 1].startswith("R:"):
            question = lines[i][3:].strip()
            reponse = lines[i + 1][3:].strip()
            qa_pairs[question] = reponse

# Prétraitement du texte avec spaCy pour le français
def preprocess(texte):
    doc = nlp(texte.lower())
    tokens = [token.lemma_ for token in doc if
              not token.is_stop and
              not token.is_punct and
              not token.is_space]
    return set(tokens)

# Fonction pour retrouver la réponse la plus pertinente
def trouver_meilleure_reponse(question):
    question_tokens = preprocess(question)
    meilleure_question = None
    meilleur_score = 0

    for q in qa_pairs.keys():
        q_tokens = preprocess(q)
        score = len(question_tokens & q_tokens) / (len(question_tokens | q_tokens) + 1e-5)  # Jaccard Similarity
        if score > meilleur_score:
            meilleur_score = score
            meilleure_question = q

    return meilleure_question, qa_pairs.get(meilleure_question,
                                        "Désolé, je n'ai pas trouvé de réponse à votre question.")

# Interface avec Streamlit
st.title("🤖 Chatbot - Science des Données")
st.write("Posez-moi une question sur la science des données, et je vous répondrai !")

# Stocker l'historique des conversations
if "historique" not in st.session_state:
    st.session_state.historique = []

# Récupérer la question de l'utilisateur
question = st.text_input("Votre question :")

if question:
    meilleure_question, reponse = trouver_meilleure_reponse(question)

    # Ajouter la question et la réponse à l'historique
    st.session_state.historique.append((question, reponse))

    # Afficher tout l'historique
    st.write("### 📜 Historique de la conversation")
    for q, r in st.session_state.historique:
        st.write(f"🟢 **Vous** : {q}")
        st.write(f"🔵 **Chatbot** : {r}")

    # Ajouter une séparation
    st.write("---")