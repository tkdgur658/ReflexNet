from monai.metrics import DiceMetric as Dice_Function
from monai.metrics import compute_iou as IoU_Function
from monai.metrics import ConfusionMatrixMetric

def Intersection_over_Union(yhat, ytrue, threshold=0.5):
    yhat = yhat>threshold
    return IoU_Function(yhat, ytrue).nanmean().item()
 
def Dice_Coefficient(yhat, ytrue, threshold=0.5):
    yhat = yhat>threshold
    return Dice_Function()(yhat, ytrue).nanmean().item()

def Confusion_Matrix(yhat, ytrue, threshold=0.5):
    yhat = yhat>threshold
    confusion_matrix = ConfusionMatrixMetric(metric_name = ["recall", "precision", "f1 score"], reduction ='mean', compute_sample =True)
    confusion_matrix(yhat, ytrue)
    recall, precision, f1 = confusion_matrix.aggregate()
    return recall, precision, f1