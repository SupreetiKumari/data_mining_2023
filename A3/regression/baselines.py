from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#The following was used on the dataloaders made for the GNN task

#Random Classifier Baseline
def random_predictions(train_dataset, val_dataset):
    train_y = [data.y[0].item() for data in train_dataset]
    val_y = [data.y[0].item() for data in val_dataset]

    dummy = DummyClassifier(strategy='uniform')
    dummy.fit(train_dataset, train_y)
    dummy_pred = dummy.predict(val_dataset)

    mse = mean_squared_error(val_y, dummy_pred)
    r2 = r2_score(val_y, dummy_pred)

    print('Mean Squared Error: {:.4f}'.format(mse))
    print('R^2 Score: {:.4f}'.format(r2))

random_predictions(train_dataset, val_dataset)

#Linear Regression Baseline
def linear_regression(train_dataset, val_dataset):
    train_y = np.array([data.y[0].item() for data in train_dataset])
    val_y = np.array([data.y[0].item() for data in val_dataset])

    train_x = np.array([np.sum(data.x.cpu().detach().numpy(), axis=0) for data in train_dataset])
    val_x = np.array([np.sum(data.x.cpu().detach().numpy(), axis=0) for data in val_dataset])

    linear = LinearRegression()
    linear.fit(train_x, train_y)
    linear_pred = linear.predict(val_x)

    mse = mean_squared_error(val_y, linear_pred)
    r2 = r2_score(val_y, linear_pred)

    print('Mean Squared Error: {:.4f}'.format(mse))
    print('R^2 Score: {:.4f}'.format(r2))

linear_regression(train_dataset, val_dataset)
