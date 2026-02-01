import os
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json
import config


def load_french_legal_data(dataset_name: str = config.DATASET_NAME):
    dataset = load_dataset(dataset_name, split="train")
    print(f"Loaded dataset with {len(dataset)} articles")
    return dataset


def filter_active_articles(dataset):
    """
    Filter to keep only articles currently in force
    """
    if "in_force" in dataset.features:
        filtered = dataset.filter(lambda x: x["in_force"] == True)
        print(f"Filtered to {len(filtered)} articles in force")
        return filtered
    return dataset


def preprocess_articles(dataset):
    """
    Preprocess articles: clean text and add metadata.
    
    Returns:
        List of dictionaries with 'id', 'text', and 'metadata'
    """
    processed_articles = []
    
    for article in tqdm(dataset, desc="Preprocessing articles"):
        article_text = article.get("article_contenu_text", article.get("article_contenu_markdown", ""))
        article_id = article.get("article_identifier", "")
        code_name = article.get("texte_titre", "")
        article_num = article.get("article_num", "")
        
        if not article_text or len(article_text.strip()) < 50:
            continue
            
        processed_articles.append({
            "id": article_id,
            "text": article_text.strip(),
            "metadata": {
                "code": code_name,
                "article": article_num,
                "source": "cold-french-law"
            }
        })
    
    print(f"Preprocessed {len(processed_articles)} articles")
    return processed_articles


def save_processed_data(articles: list, output_path: str):
    """
    Save processed articles to JSON file
    
    Args:
        articles: List of processed article dictionaries
        output_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(articles)} articles to {output_path}")


def load_processed_data(input_path: str) -> list:
    with open(input_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles from {input_path}")
    return articles


if __name__ == "__main__":
    dataset = load_french_legal_data()
    filtered = filter_active_articles(dataset)
    articles = preprocess_articles(filtered)
    save_processed_data(articles, "data/processed_articles.json")
