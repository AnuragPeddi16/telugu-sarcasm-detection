import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
import os

def load_translations(file_path):
    """Load translations from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_bleu_scores(reference, hypothesis):
    """Compute BLEU-1, BLEU-2, and BLEU-L scores"""
    # Tokenize sentences
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    
    # Smoothing function for better scores with short sentences
    smoothing = SmoothingFunction().method1
    
    # BLEU-1
    bleu1 = sentence_bleu(reference_tokens, hypothesis_tokens, 
                          weights=(1, 0, 0, 0), 
                          smoothing_function=smoothing)
    
    # BLEU-2
    bleu2 = sentence_bleu(reference_tokens, hypothesis_tokens, 
                          weights=(0.5, 0.5, 0, 0), 
                          smoothing_function=smoothing)
    
    # BLEU-L (using BLEU-4 as approximation)
    bleu_l = sentence_bleu(reference_tokens, hypothesis_tokens, 
                           weights=(0.25, 0.25, 0.25, 0.25), 
                           smoothing_function=smoothing)
    
    return bleu1, bleu2, bleu_l

def compute_rouge_scores(reference, hypothesis):
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = scorer.score(reference, hypothesis)
    
    # Extract F1 scores
    rouge1 = scores['rouge1'].fmeasure
    rouge2 = scores['rouge2'].fmeasure
    rouge_l = scores['rougeL'].fmeasure
    
    return rouge1, rouge2, rouge_l

def evaluate_against_baseline(baseline_data, hypothesis_data, baseline_name, hypothesis_name):
    """Evaluate hypothesis translations against baseline"""
    
    scores = {
        'bleu1': [], 'bleu2': [], 'bleu_l': [],
        'rouge1': [], 'rouge2': [], 'rouge_l': []
    }
    
    for i in range(min(len(baseline_data), len(hypothesis_data))):
        baseline_item = baseline_data[i]
        hypothesis_item = hypothesis_data[i]
        
        # Get utterances (translated text)
        baseline_utterance = baseline_item.get('utterance', '')
        hypothesis_utterance = hypothesis_item.get('utterance', '')
        
        # Skip if either is empty
        if not baseline_utterance or not hypothesis_utterance:
            continue
        
        # Compute scores
        bleu1, bleu2, bleu_l = compute_bleu_scores(baseline_utterance, hypothesis_utterance)
        rouge1, rouge2, rouge_l = compute_rouge_scores(baseline_utterance, hypothesis_utterance)
        
        scores['bleu1'].append(bleu1)
        scores['bleu2'].append(bleu2)
        scores['bleu_l'].append(bleu_l)
        scores['rouge1'].append(rouge1)
        scores['rouge2'].append(rouge2)
        scores['rouge_l'].append(rouge_l)
    
    # Calculate average scores
    avg_scores = {k: np.mean(v) if v else 0.0 for k, v in scores.items()}
    
    return avg_scores, len(scores['bleu1'])

# def evaluate_all_translations():
#     """Evaluate all translation models against both baselines"""
    
#     # Define all translation files
#     translation_files = [
#         "translations_ai4bharat.json",
#         "translations_anithasoma.json",
#         "translations_apigoogle.json",
#         "translations_archita.json",
#         "translations_aryaumesh.json",
#         "translations_googlemadlad400.json",
#         "translations_gvaishnavi_10_1024.json",
#         "translations_nllb_200.json",
#         "translations_rope_indictrans2.json",
#         "translations_seamlessm4tv2.json"
#     ]
    
#     # Filter files that exist
#     existing_files = [f for f in translation_files if os.path.exists(f)]
    
#     # Load baseline datasets
#     baseline_files = {
#         "API Google": "translations_apigoogle.json",
#         "ROPE IndIcTrans2": "translations_rope_indictrans2.json",
#         "Codex": "translations_codex.json"  # Assuming this is the Codex baseline file
#     }
    
#     baselines = {}
#     for baseline_name, baseline_file in baseline_files.items():
#         if baseline_file in existing_files:
#             baselines[baseline_name] = load_translations(baseline_file)
    
#     # Evaluate each model against both baselines
#     for baseline_name, baseline_data in baselines.items():
#         print("=" * 100)
#         print(f"EVALUATION AGAINST {baseline_name.upper()} AS BASELINE")
#         print("=" * 100)
#         print()
        
#         results = []
        
#         for trans_file in existing_files:
#             # Skip if evaluating baseline against itself or against the other baseline
#             if trans_file in baseline_files.values():
#                 continue
            
#             model_name = trans_file.replace("translations_", "").replace(".json", "")
#             hypothesis_data = load_translations(trans_file)
            
#             avg_scores, num_samples = evaluate_against_baseline(
#                 baseline_data, hypothesis_data, baseline_name, model_name
#             )
            
#             results.append({
#                 'model': model_name,
#                 'scores': avg_scores,
#                 'samples': num_samples
#             })
        
#         # Sort results by BLEU-1 score
#         results.sort(key=lambda x: x['scores']['bleu1'], reverse=True)
        
#         # Print results in a formatted table
#         print(f"{'Model':<25} {'BLEU-1':<10} {'BLEU-2':<10} {'BLEU-L':<10} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'Samples':<10}")
#         print("-" * 100)
        
#         for result in results:
#             model = result['model']
#             scores = result['scores']
#             samples = result['samples']
            
#             print(f"{model:<25} "
#                   f"{scores['bleu1']:<10.4f} "
#                   f"{scores['bleu2']:<10.4f} "
#                   f"{scores['bleu_l']:<10.4f} "
#                   f"{scores['rouge1']:<10.4f} "
#                   f"{scores['rouge2']:<10.4f} "
#                   f"{scores['rouge_l']:<10.4f} "
#                   f"{samples:<10}")
        
#         print()
#         print(f"Best performing model: {results[0]['model']} (BLEU-1: {results[0]['scores']['bleu1']:.4f})")
#         print()
    
#     # Cross-comparison between baselines
#     print("=" * 100)
#     print("BASELINE CROSS-COMPARISON")
#     print("=" * 100)
#     print()
    
#     if "API Google" in baselines and "ROPE IndIcTrans2" in baselines:
#         # API Google vs ROPE IndIcTrans2
#         scores_api_vs_rope, samples = evaluate_against_baseline(
#             baselines["ROPE IndIcTrans2"], 
#             baselines["API Google"], 
#             "ROPE IndIcTrans2", 
#             "API Google"
#         )
        
#         scores_rope_vs_api, _ = evaluate_against_baseline(
#             baselines["API Google"], 
#             baselines["ROPE IndIcTrans2"], 
#             "API Google", 
#             "ROPE IndIcTrans2"
#         )
        
#         print(f"API Google (vs ROPE IndIcTrans2 as reference):")
#         print(f"  BLEU-1: {scores_api_vs_rope['bleu1']:.4f}, BLEU-2: {scores_api_vs_rope['bleu2']:.4f}, BLEU-L: {scores_api_vs_rope['bleu_l']:.4f}")
#         print(f"  ROUGE-1: {scores_api_vs_rope['rouge1']:.4f}, ROUGE-2: {scores_api_vs_rope['rouge2']:.4f}, ROUGE-L: {scores_api_vs_rope['rouge_l']:.4f}")
#         print()
        
#         print(f"ROPE IndIcTrans2 (vs API Google as reference):")
#         print(f"  BLEU-1: {scores_rope_vs_api['bleu1']:.4f}, BLEU-2: {scores_rope_vs_api['bleu2']:.4f}, BLEU-L: {scores_rope_vs_api['bleu_l']:.4f}")
#         print(f"  ROUGE-1: {scores_rope_vs_api['rouge1']:.4f}, ROUGE-2: {scores_rope_vs_api['rouge2']:.4f}, ROUGE-L: {scores_rope_vs_api['rouge_l']:.4f}")
#         print()
#         print(f"Samples evaluated: {samples}")

def evaluate_all_translations():
    """Evaluate all translation models against all baselines"""
    
    # Define all translation files
    translation_files = [
        "translations_ai4bharat.json",
        "translations_anithasoma.json",
        "translations_apigoogle.json",
        "translations_archita.json",
        "translations_aryaumesh.json",
        "translations_codex.json",
        "translations_googlemadlad400.json",
        "translations_gvaishnavi_10_1024.json",
        "translations_nllb_200.json",
        "translations_rope_indictrans2.json",
        "translations_seamlessm4tv2.json"
    ]
    
    # Filter files that exist
    existing_files = [f for f in translation_files if os.path.exists(f)]
    
    # Load baseline datasets - now including all three
    baseline_files = {
        "Codex": "translations_codex.json",
        "API Google": "translations_apigoogle.json",
        "ROPE IndIcTrans2": "translations_rope_indictrans2.json"
    }
    
    baselines = {}
    for baseline_name, baseline_file in baseline_files.items():
        if baseline_file in existing_files:
            baselines[baseline_name] = load_translations(baseline_file)
            print(f"Loaded baseline: {baseline_name} ({len(baselines[baseline_name])} samples)")
    
    print()
    
    # Evaluate each model against all baselines
    for baseline_name, baseline_data in baselines.items():
        print("=" * 100)
        print(f"EVALUATION AGAINST {baseline_name.upper()} AS BASELINE")
        print("=" * 100)
        print()
        
        results = []
        
        for trans_file in existing_files:
            # Skip if evaluating baseline against itself
            if trans_file == baseline_files[baseline_name]:
                continue
            
            model_name = trans_file.replace("translations_", "").replace(".json", "")
            hypothesis_data = load_translations(trans_file)
            
            avg_scores, num_samples = evaluate_against_baseline(
                baseline_data, hypothesis_data, baseline_name, model_name
            )
            
            results.append({
                'model': model_name,
                'scores': avg_scores,
                'samples': num_samples
            })
        
        # Sort results by BLEU-1 score
        results.sort(key=lambda x: x['scores']['bleu1'], reverse=True)
        
        # Print results in a formatted table
        print(f"{'Model':<25} {'BLEU-1':<10} {'BLEU-2':<10} {'BLEU-L':<10} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'Samples':<10}")
        print("-" * 100)
        
        for result in results:
            model = result['model']
            scores = result['scores']
            samples = result['samples']
            
            print(f"{model:<25} "
                  f"{scores['bleu1']:<10.4f} "
                  f"{scores['bleu2']:<10.4f} "
                  f"{scores['bleu_l']:<10.4f} "
                  f"{scores['rouge1']:<10.4f} "
                  f"{scores['rouge2']:<10.4f} "
                  f"{scores['rouge_l']:<10.4f} "
                  f"{samples:<10}")
        
        print()
        print(f"Best performing model: {results[0]['model']} (BLEU-1: {results[0]['scores']['bleu1']:.4f})")
        print()
    
    # Cross-comparison between all baselines
    print("=" * 100)
    print("BASELINE CROSS-COMPARISON")
    print("=" * 100)
    print()
    
    baseline_names = list(baselines.keys())
    
    # Compare each baseline against all others
    for i, baseline1 in enumerate(baseline_names):
        for baseline2 in baseline_names[i+1:]:
            scores_1_vs_2, samples = evaluate_against_baseline(
                baselines[baseline2], 
                baselines[baseline1], 
                baseline2, 
                baseline1
            )
            
            scores_2_vs_1, _ = evaluate_against_baseline(
                baselines[baseline1], 
                baselines[baseline2], 
                baseline1, 
                baseline2
            )
            
            print(f"{baseline1} (vs {baseline2} as reference):")
            print(f"  BLEU-1: {scores_1_vs_2['bleu1']:.4f}, BLEU-2: {scores_1_vs_2['bleu2']:.4f}, BLEU-L: {scores_1_vs_2['bleu_l']:.4f}")
            print(f"  ROUGE-1: {scores_1_vs_2['rouge1']:.4f}, ROUGE-2: {scores_1_vs_2['rouge2']:.4f}, ROUGE-L: {scores_1_vs_2['rouge_l']:.4f}")
            print()
            
            print(f"{baseline2} (vs {baseline1} as reference):")
            print(f"  BLEU-1: {scores_2_vs_1['bleu1']:.4f}, BLEU-2: {scores_2_vs_1['bleu2']:.4f}, BLEU-L: {scores_2_vs_1['bleu_l']:.4f}")
            print(f"  ROUGE-1: {scores_2_vs_1['rouge1']:.4f}, ROUGE-2: {scores_2_vs_1['rouge2']:.4f}, ROUGE-L: {scores_2_vs_1['rouge_l']:.4f}")
            print()
            print(f"Samples evaluated: {samples}")
            print("-" * 50)
            print()

if __name__ == "__main__":
    evaluate_all_translations()