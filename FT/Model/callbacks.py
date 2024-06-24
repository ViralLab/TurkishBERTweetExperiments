import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


class Accuracy:
    def __call__(self, y_true, y_pred):
        return (y_pred.flatten() == y_true.flatten()).float().mean()


class ConfusionMatrix:
    def __call__(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)


class ClassificationReport:
    def __call__(self, y_true, y_pred, target_names):
        return classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            output_dict=True,
        )


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
