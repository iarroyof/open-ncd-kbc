from typing import List, Dict, Any
import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
import gc

class TextGenerationMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.smooth = SmoothingFunction()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Download required NLTK data
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')

    def decode_tokens(self, token_ids: torch.Tensor) -> List[str]:
        """Convert token IDs to text"""
        decoded = []
        for seq in token_ids.cpu().numpy():
            # Remove padding tokens
            seq = seq[seq != 0]
            text = self.tokenizer.decode(seq)
            decoded.append(text)
        return decoded

    def compute_bleu(self, predictions: List[str], references: List[List[str]]) -> float:
        """Compute BLEU score"""
        pred_tokens = [pred.split() for pred in predictions]
        ref_tokens = [[ref.split()] for ref in references]
        return corpus_bleu(ref_tokens, pred_tokens, smoothing_function=self.smooth.method1)

    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores"""
        scores = {'rouge1': 0., 'rouge2': 0., 'rougeL': 0.}
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            for key in scores:
                scores[key] += score[key].fmeasure
        
        # Average scores
        for key in scores:
            scores[key] /= len(predictions)
        return scores

    def compute_meteor(self, predictions: List[str], references: List[str]) -> float:
        """Compute METEOR score"""
        scores = []
        for pred, ref in zip(predictions, references):
            score = meteor_score([ref.split()], pred.split())
            scores.append(score)
        return np.mean(scores)

    @torch.no_grad()
    def compute_metrics(self, predictions, references):
        # Decode predictions
        pred_texts = []
        for pred in predictions:
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            text = self.tokenizer.decode(pred, skip_special_tokens=True)
            pred_texts.append(text)
        
        # Decode references
        ref_texts = []
        for ref in references:
            if isinstance(ref, torch.Tensor):
                ref = ref.cpu().numpy()
            text = self.tokenizer.decode(ref, skip_special_tokens=True)
            ref_texts.append(text)
        
        # Compute metrics using the decoded texts
        bleu = compute_bleu(pred_texts, ref_texts)  # Replace with your BLEU implementation
        rouge = compute_rouge(pred_texts, ref_texts)  # Replace with your ROUGE implementation
        meteor = compute_meteor(pred_texts, ref_texts)  # Replace with your METEOR implementation
        
        # Return the computed metrics
        return {
            'bleu': bleu,
            'rouge1': rouge['rouge1'],
            'rouge2': rouge['rouge2'],
            'rougeL': rouge['rougeL'],
            'meteor': meteor
        }
