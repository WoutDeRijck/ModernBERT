# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import argparse

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

def train_on_mldr(model, lr, model_shortname, language='en'):
    # Load MLDR dataset for specific language
    train_dataset = load_dataset('Shitao/MLDR', language, split='train')
    eval_dataset = load_dataset('Shitao/MLDR', language, split='dev')
    
    # Convert dataset to the format needed for training
    def format_for_training(dataset):
        from datasets import Dataset
        # Extract text from passages and ensure they're strings
        return Dataset.from_dict({
            'query': dataset['query'],
            'positive': [str(passages[0]['text']) for passages in dataset['positive_passages']],
            'negative': [str(passages[0]['text']) for passages in dataset['negative_passages']]
        })
    
    train_data = format_for_training(train_dataset)
    eval_data = format_for_training(eval_dataset)

    # Create evaluator with more detailed metrics
    dev_evaluator = InformationRetrievalEvaluator(
        queries=dict(zip(range(len(eval_data['query'])), eval_data['query'])),
        corpus=dict(zip(range(len(eval_data['positive'])), eval_data['positive'])),
        relevant_docs={i: {i} for i in range(len(eval_data['query']))},
        corpus_chunk_size=512,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1],
        precision_recall_at_k=[10],
        map_at_k=[10],
        show_progress_bar=True,
        name=f"mldr-{language}-dev"
    )


    # Evaluate base model
    print("\n=== Evaluating Base Model ===")
    base_score = dev_evaluator(model)
    print(f"Base Model Score: {base_score}\n")

    # Define loss function for MLDR
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)

    run_name = f"{model_shortname}-MLDR-{language}-{lr}"
    args = SentenceTransformerTrainingArguments(
        output_dir=f"output/{model_shortname}/{run_name}",
        num_train_epochs=1,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        learning_rate=lr/10,  # Lower learning rate for second phase
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=20,
        run_name=run_name,
        eval_strategy="steps",
        eval_steps=20,
        report_to=["none"],  # Disable wandb logging
    )

    # Create trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()
    
    # Evaluate final model
    print("\n=== Evaluating Final Model ===")
    final_score = dev_evaluator(model)
    print(f"Final Model Score: {final_score}\n")
    
    model.save_pretrained(f"output/{model_shortname}/{run_name}/final")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--language", type=str, default="zh", 
                       choices=['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh'])
    args = parser.parse_args()
    
    model_shortname = args.model_name.split("/")[-1]
    model = SentenceTransformer(args.model_name, 
                              model_kwargs={"attn_implementation": "flash_attention_2"})
    
    print(f"=== Training on MLDR ({args.language}) ===")
    model = train_on_mldr(model, args.lr, model_shortname, args.language)
    
    # Save locally only
    model.save_pretrained(f"output/{model_shortname}/mldr_{args.language}_trained")

if __name__ == "__main__":
    main() 