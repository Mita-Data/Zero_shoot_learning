import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Charger le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def zero_shot_classification(text, labels, model, tokenizer):
    # Tokeniser et préparer les inputs
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    # Obtenir les outputs du modèle
    with torch.no_grad():
        outputs = model(**inputs)

    # Utiliser la moyenne des sorties du dernier état caché pour obtenir l'embedding du texte
    text_embedding = outputs.last_hidden_state.mean(1)

    label_embeddings = []
    for label in labels:
        with torch.no_grad():
            # Tokeniser les labels et calculer les embeddings
            label_inputs = tokenizer(label, padding=True, truncation=True, return_tensors="pt", max_length=512)
            label_outputs = model(**label_inputs)
            label_embeddings.append(label_outputs.last_hidden_state.mean(1))
    
    label_embeddings = torch.cat(label_embeddings)

    # Calculer la similarité cosinus entre l'embedding du texte et chaque embedding des étiquettes
    similarities = F.cosine_similarity(text_embedding, label_embeddings)

    # Trouver l'indice de la meilleure similarité
    best_label_idx = similarities.argmax().item()

    return labels[best_label_idx], similarities[best_label_idx].item()

def load_data(file_path):
    try:
        return pd.read_csv(file_path, sep='@', names=['sentence', 'sentiment'], engine='python', encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, sep='@', names=['sentence', 'sentiment'], engine='python', encoding='ISO-8859-1')

# Chemins des fichiers
files = ['Sentences_AllAgree.txt', 'Sentences_75Agree.txt', 'Sentences_66Agree.txt']

# Labels pour la classification
labels = ["positive", "negative", "neutral"]

# Charger et prédire pour toutes les phrases de chaque fichier
for file_path in files:
    data = load_data(file_path)
    correct_predictions = 0
    total_predictions = 0
    for index, row in data.iterrows():
        predicted_label, _ = zero_shot_classification(row['sentence'], labels, model, tokenizer)
        # Compter les prédictions correctes
        if predicted_label == row['sentiment']:
            correct_predictions += 1
        total_predictions += 1
    # Calculer et afficher la précision pour le fichier
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Accuracy for {file_path}: {accuracy:.4f}")