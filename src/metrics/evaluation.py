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
    def compute_metrics(self, predictions: torch.Tensor, references: torch.Tensor) -> Dict[str, float]:
        """Compute all metrics"""
        # Convert tokens to text
        pred_texts = self.decode_tokens(predictions)
        ref_texts = self.decode_tokens(references)
        
        # Compute metrics
        metrics = {}
        metrics['bleu'] = self.compute_bleu(pred_texts, ref_texts)
        
        rouge_scores = self.compute_rouge(pred_texts, ref_texts)
        metrics.update(rouge_scores)
        
        metrics['meteor'] = self.compute_meteor(pred_texts, ref_texts)
        
        # Clear cache
        del pred_texts, ref_texts
        gc.collect()
        
        return metrics
