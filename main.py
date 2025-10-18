#!/usr/bin/env python3
"""
C4 Calibration Data Analysis with Progressive Block Masking

This script downloads 128 C4 calibration samples and analyzes how GPT-2's
predictions change as transformer blocks are progressively enabled/disabled.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models import GPT2Model, GPT2Config
import tiktoken
import requests
import json
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set up tokenizer
tokenizer = tiktoken.get_encoding('gpt2')

def load_c4_calibration_data(n_samples: int = 128) -> List[str]:
    """Load C4 calibration data from local JSON file."""
    print(f"Loading {n_samples} C4 calibration samples from local file...")
    
    try:
        with open('c4_calibration_texts.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        calibration_texts = data['calibration_texts']
        
        # Limit to requested number of samples
        if len(calibration_texts) > n_samples:
            calibration_texts = calibration_texts[:n_samples]
        
        print(f"Loaded {len(calibration_texts)} calibration samples from C4 dataset")
        print(f"Sample lengths: min={min(len(text) for text in calibration_texts)}, "
              f"max={max(len(text) for text in calibration_texts)}, "
              f"avg={sum(len(text) for text in calibration_texts) // len(calibration_texts)}")
        
        return calibration_texts
        
    except FileNotFoundError:
        print("Error: c4_calibration_texts.json not found!")
        print("Please ensure the file exists in the current directory.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return []
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return []

def analyze_block_contribution(model, input_text: str, block_mask: List[bool]) -> Dict:
    """Run inference with given block mask and return analysis."""
    # Tokenize input
    tokens = tokenizer.encode(input_text)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        logits, _ = model(input_ids, block_mask=block_mask)
        last_token_logits = logits[0, -1, :]
    
    # Get top predictions
    probs = F.softmax(last_token_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 5, dim=-1)
    
    # Decode predictions
    predictions = []
    for i in range(5):
        token_id = top_indices[i].item()
        prob = top_probs[i].item()
        token_text = tokenizer.decode([token_id])
        predictions.append((token_text, prob, token_id))
    
    return {
        'logits': last_token_logits,
        'predictions': predictions
    }

def measure_prediction_distance(logits1: torch.Tensor, logits2: torch.Tensor) -> Dict:
    """Measure distance between two prediction distributions."""
    probs1 = F.softmax(logits1, dim=-1)
    probs2 = F.softmax(logits2, dim=-1)
    
    # KL divergence
    kl_div = F.kl_div(probs1.log(), probs2, reduction='sum')
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(logits1, logits2, dim=-1)
    
    # L2 distance
    l2_dist = torch.norm(logits1 - logits2, p=2)
    
    return {
        'kl_divergence': kl_div.item(),
        'cosine_similarity': cos_sim.item(),
        'l2_distance': l2_dist.item()
    }

def weighted_overlap(pred1_with_probs: List[Tuple], pred2_with_probs: List[Tuple]) -> float:
    """Weighted overlap considering probabilities."""
    # Create probability dictionaries
    prob1 = {token: prob for token, prob, _ in pred1_with_probs}
    prob2 = {token: prob for token, prob, _ in pred2_with_probs}
    
    # Calculate weighted intersection
    intersection_weight = 0
    for token in set(prob1.keys()).intersection(set(prob2.keys())):
        intersection_weight += min(prob1[token], prob2[token])
    
    # Calculate weighted union
    union_weight = 0
    all_tokens = set(prob1.keys()).union(set(prob2.keys()))
    for token in all_tokens:
        union_weight += max(prob1.get(token, 0), prob2.get(token, 0))
    
    return intersection_weight / union_weight if union_weight > 0 else 0

def analyze_c4_calibration_data(model, calibration_texts: List[str]) -> Dict:
    """Analyze C4 calibration data with progressive block masking."""
    print("Analyzing C4 calibration data...")
    
    # Test configurations
    configs = []
    for i in range(13):  # 0 to 12 blocks
        if i == 0:
            mask = [False] * 12
            configs.append((mask, f"0 blocks"))
        else:
            mask = [False] * 12
            for j in range(i):
                mask[j] = True
            configs.append((mask, f"{i} blocks"))
    
    # Store results for all samples
    all_results = {
        'l2_distances_from_baseline': [],
        'l2_distances_to_full': [],
        'weighted_overlaps_from_baseline': [],
        'weighted_overlaps_to_full': [],
        'block_numbers': []
    }
    
    # Process each calibration text
    for sample_idx, text in enumerate(calibration_texts):
        if sample_idx % 20 == 0:
            print(f"Processing sample {sample_idx + 1}/{len(calibration_texts)}")
        
        # Run all configurations for this text
        results = []
        for mask, description in configs:
            result = analyze_block_contribution(model, text, mask)
            results.append(result)
        
        # Calculate metrics
        baseline_logits = results[0]['logits']
        full_model_logits = results[-1]['logits']
        
        for i, result in enumerate(results):  # Include all results including baseline
            current_logits = result['logits']
            
            # Distance metrics
            dist_from_baseline = measure_prediction_distance(baseline_logits, current_logits)
            dist_to_full = measure_prediction_distance(full_model_logits, current_logits)
            
            # Overlap metrics
            overlap_from_baseline = weighted_overlap(results[0]['predictions'], result['predictions'])
            overlap_to_full = weighted_overlap(results[-1]['predictions'], result['predictions'])
            
            # Store results
            all_results['l2_distances_from_baseline'].append(dist_from_baseline['l2_distance'])
            all_results['l2_distances_to_full'].append(dist_to_full['l2_distance'])
            all_results['weighted_overlaps_from_baseline'].append(overlap_from_baseline)
            all_results['weighted_overlaps_to_full'].append(overlap_to_full)
            all_results['block_numbers'].append(i)
    
    return all_results

def create_box_plots(results: Dict):
    """Create box plots for the analysis results."""
    print("Creating box plots...")
    
    # Convert to numpy arrays for easier manipulation
    block_numbers = np.array(results['block_numbers'])
    l2_from_baseline = np.array(results['l2_distances_from_baseline'])
    l2_to_full = np.array(results['l2_distances_to_full'])
    overlap_from_baseline = np.array(results['weighted_overlaps_from_baseline'])
    overlap_to_full = np.array(results['weighted_overlaps_to_full'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: L2 Distance from Baseline
    data_l2_baseline = []
    labels_l2_baseline = []
    for i in range(13):  # 0 to 12 blocks
        mask = block_numbers == i
        data_l2_baseline.append(l2_from_baseline[mask])
        labels_l2_baseline.append(f'{i}')
    
    axes[0, 0].boxplot(data_l2_baseline, labels=labels_l2_baseline)
    axes[0, 0].set_title('Last Token Logits, Distance from Baseline')
    axes[0, 0].set_xlabel('Number of Blocks Enabled')
    axes[0, 0].set_ylabel('L2 Distance')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: L2 Distance to Full Model
    data_l2_full = []
    labels_l2_full = []
    for i in range(13):  # 0 to 12 blocks
        mask = block_numbers == i
        data_l2_full.append(l2_to_full[mask])
        labels_l2_full.append(f'{i}')
    
    axes[0, 1].boxplot(data_l2_full, labels=labels_l2_full)
    axes[0, 1].set_title('Last Token Logits, Distance to Full Model')
    axes[0, 1].set_xlabel('Number of Blocks Enabled')
    axes[0, 1].set_ylabel('L2 Distance')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Weighted Overlap from Baseline
    data_overlap_baseline = []
    labels_overlap_baseline = []
    for i in range(13):  # 0 to 12 blocks
        mask = block_numbers == i
        data_overlap_baseline.append(overlap_from_baseline[mask])
        labels_overlap_baseline.append(f'{i}')
    
    axes[1, 0].boxplot(data_overlap_baseline, labels=labels_overlap_baseline)
    axes[1, 0].set_title('Top 5 Predictions, Overlap from Baseline')
    axes[1, 0].set_xlabel('Number of Blocks Enabled')
    axes[1, 0].set_ylabel('Weighted Overlap')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Weighted Overlap to Full Model
    data_overlap_full = []
    labels_overlap_full = []
    for i in range(13):  # 0 to 12 blocks
        mask = block_numbers == i
        data_overlap_full.append(overlap_to_full[mask])
        labels_overlap_full.append(f'{i}')
    
    axes[1, 1].boxplot(data_overlap_full, labels=labels_overlap_full)
    axes[1, 1].set_title('Top 5 Predictions, Overlap to Full Model')
    axes[1, 1].set_xlabel('Number of Blocks Enabled')
    axes[1, 1].set_ylabel('Weighted Overlap')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('c4_calibration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary statistics
    summary_stats = {}
    for i in range(13):  # 0 to 12 blocks
        mask = block_numbers == i
        summary_stats[f'{i}_blocks'] = {
            'l2_from_baseline_mean': np.mean(l2_from_baseline[mask]),
            'l2_from_baseline_std': np.std(l2_from_baseline[mask]),
            'l2_to_full_mean': np.mean(l2_to_full[mask]),
            'l2_to_full_std': np.std(l2_to_full[mask]),
            'overlap_from_baseline_mean': np.mean(overlap_from_baseline[mask]),
            'overlap_from_baseline_std': np.std(overlap_from_baseline[mask]),
            'overlap_to_full_mean': np.mean(overlap_to_full[mask]),
            'overlap_to_full_std': np.std(overlap_to_full[mask])
        }
    
    return summary_stats

def main():
    """Main function to run C4 calibration analysis."""
    print("Loading GPT-2 124M model...")
    model = GPT2Model.from_pretrained('gpt2')
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}!")
    
    # Load C4 calibration data
    calibration_texts = load_c4_calibration_data(128)
    
    if not calibration_texts:
        print("Failed to load calibration data. Exiting.")
        return None, None
    
    # Analyze the data
    results = analyze_c4_calibration_data(model, calibration_texts)
    
    # Create visualizations
    summary_stats = create_box_plots(results)
    
    print("Analysis complete! Results saved as 'c4_calibration_analysis.png'")
    
    return results, summary_stats

if __name__ == "__main__":
    results, stats = main()