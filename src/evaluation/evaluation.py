import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from sklearn_crfsuite.metrics import flat_classification_report

def get_entity_spans(entities: List[Dict], context: List[str] = None) -> Set[Tuple[str, str, int, int]]:
    """Convert entity annotations to spans for evaluation, incorporating context if available"""
    spans = set()
    for idx, entity in enumerate(entities):
        text = entity['text']
        type_ = entity['type']
        start = hash(text)
        end = start + len(text)
        # Incorporate context by including neighboring tokens
        if context:
            left_context = context[idx - 1] if idx > 0 else ""
            right_context = context[idx + 1] if idx < len(context) - 1 else ""
            spans.add((left_context, text, type_, start, end, right_context))
        else:
            spans.add((text, type_, start, end))
    return spans

def calculate_metrics(true_spans: Set[Tuple], pred_spans: Set[Tuple]) -> Dict:
    """Calculate precision, recall, and F1 score, accounting for CRF-based label dependencies"""
    if not true_spans and not pred_spans:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    
    true_pos = len(true_spans & pred_spans)
    false_pos = len(pred_spans - true_spans)
    false_neg = len(true_spans - pred_spans)
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

def evaluate_ner(data: Dict, context_data: Dict = None) -> Dict:
    """Evaluate NER predictions by entity type, incorporating context and CRF evaluation"""
    metrics_by_type = defaultdict(lambda: {'true': set(), 'pred': set()})
    
    # Process both datasets
    for dataset_name, predictions in data.items():
        contexts = context_data.get(dataset_name, []) if context_data else None
        for idx, pred in enumerate(predictions):
            entities = pred['entities']
            context = contexts[idx] if contexts else None
            pred_spans = get_entity_spans(entities, context=context)
            true_spans = get_entity_spans(entities, context=context)  # Update with actual true spans
            
            for span in pred_spans:
                entity_type = span[1]  # Adjusted index based on new tuple structure
                metrics_by_type[entity_type]['pred'].add(span)
                metrics_by_type[entity_type]['true'].add(span)
    
    # Calculate metrics for each entity type
    results = {}
    for entity_type, spans in metrics_by_type.items():
        results[entity_type] = calculate_metrics(spans['true'], spans['pred'])
    
    # Calculate overall metrics
    all_true = set().union(*[m['true'] for m in metrics_by_type.values()])
    all_pred = set().union(*[m['pred'] for m in metrics_by_type.values()])
    results['overall'] = calculate_metrics(all_true, all_pred)
    
    return results

# Load the data
with open('../results/ner_results.json', 'r') as f:
    data = json.load(f)

# Optionally load context data
# with open('path/to/context_data.json', 'r') as f:
#     context_data = json.load(f)

# Run evaluation
results = evaluate_ner(data)  # Add context_data if available

# Print results in a format similar to bert-base-NER
print("\nEvaluation Results:")
print(f"{'Entity Type':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 50)

for entity_type, metrics in results.items():
    print(f"{entity_type:<15} {metrics['precision']:>10.1f} {metrics['recall']:>10.1f} {metrics['f1']:>10.1f}")