import datasets, evaluate
from transformers.trainer_utils import EvalPrediction

accuracy = evaluate.load("accuracy").compute
precision = evaluate.load("precision").compute
recall = evaluate.load("recall").compute
f1 = evaluate.load("f1").compute
squad_v2 = evaluate.load("squad_v2").compute


def compute_classification_metric(p: EvalPrediction):
    """
    'predictions': Value(dtype='int32', id=None)
    'references': Value(dtype='int32', id=None)
    """
    predictions = p.predictions.argmax(axis=1)
    references = p.label_ids
    metric = accuracy(predictions=predictions, references=references)
    metric.update(precision(predictions=predictions, references=references))
    metric.update(recall(predictions=predictions, references=references))
    metric.update(f1(predictions=predictions, references=references))
    return metric


def compute_squad_v2(p: EvalPrediction):
    """
    'predictions': {
        'id': Value(dtype='string', id=None),
        'prediction_text': Value(dtype='string', id=None),
        'no_answer_probability': Value(dtype='float32', id=None)
    }
    'references': {
        'id': Value(dtype='string', id=None),
        'answers': Sequence(
            feature={
                'text': Value(dtype='string', id=None), 
                'answer_start': Value(dtype='int32', id=None)
            },
            length=-1, id=None
        )
    }
    """
    predictions = p.predictions
    references = p.label_ids
    return squad_v2(predictions=predictions, references=references)