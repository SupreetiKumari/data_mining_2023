from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


#The below was used on the dataset loader made for the GNN task

#Random Classifier Baseline
def random_predictions(train_dataset, val_dataset):
    train_y = [data.y[0].item() for data in train_dataset]
    val_y = [data.y[0].item() for data in val_dataset]

    dummy = DummyClassifier(strategy='uniform')
    dummy.fit(train_dataset, train_y)
    dummy_pred = dummy.predict(val_dataset)

    print('Accuracy: {:.4f}'.format(accuracy_score(val_y, dummy_pred)))
    print('Precision: {:.4f}'.format(precision_score(val_y, dummy_pred)))
    print('Recall: {:.4f}'.format(recall_score(val_y, dummy_pred)))
    print('F1: {:.4f}'.format(f1_score(val_y, dummy_pred)))
    print('ROC-AUC: {:.4f}'.format(roc_auc_score(val_y, dummy_pred)))

random_predictions(train_dataset, val_dataset)

#Logistic Regression Baseline
def logistic_regression(train_dataset, val_dataset):
    train_y = [data.y[0].item() for data in train_dataset]
    val_y = [data.y[0].item() for data in val_dataset]

    train_x = np.array([np.sum(data.x.cpu().detach().numpy(), axis=0) for data in train_dataset])
    val_x = np.array([np.sum(data.x.cpu().detach().numpy(), axis=0) for data in val_dataset])

    logistic = LogisticRegression()
    logistic.fit(train_x, train_y)
    logistic_pred = logistic.predict(val_x)

    print('Accuracy: {:.4f}'.format(accuracy_score(val_y, logistic_pred)))
    print('Precision: {:.4f}'.format(precision_score(val_y, logistic_pred)))
    print('Recall: {:.4f}'.format(recall_score(val_y, logistic_pred)))
    print('F1: {:.4f}'.format(f1_score(val_y, logistic_pred)))
    print('ROC-AUC: {:.4f}'.format(roc_auc_score(val_y, logistic_pred)))

logistic_regression(train_dataset, val_dataset)
