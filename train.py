import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

def batch_gd(model, optimizer, criterion, train_loader, validation_loader, epochs, device):
    softmax = nn.Softmax(dim=1)
    train_losses = np.zeros(epochs)
    validation_losses = np.zeros(epochs)
    validation_loss_min = np.inf

    save_dir = './saved_models/'

    for e in range(epochs):
        #print(e)
        t0 = datetime.now()
        train_loss = []
        y_pred = []
        y_true = []

        # Model training
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print("TRAINING DONE")
        print('*'*20)

        train_loss = np.mean(train_loss)

        validation_loss = []
        y_pred = []
        y_true = []
        # model validation
        model.eval()
        for inputs, targets in validation_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            
            predictions = torch.argmax(softmax(output),dim=1)
            y_pred.extend(predictions)
            y_true.extend(targets)
            
            loss = criterion(output, targets)
            validation_loss.append(loss.item())

        print("VALIDATION DONE")
        print('*'*20)

        validation_loss = np.mean(validation_loss)
        y_true = torch.tensor(y_true, device = 'cpu')
        y_pred = torch.tensor(y_pred, device = 'cpu')

        train_losses[e] = train_loss
        validation_losses[e] = validation_loss

        dt = datetime.now() - t0

        print(
            f"Epoch : {e+1}/{epochs} Train_loss:{train_loss:.3f} Validation_loss:{validation_loss:.3f} Duration:{dt}"
        )
        print('*'*50)

        # Save the model if validation loss has decreased
        if validation_loss <= validation_loss_min:
            print('Validation loss decreased ({:.5f} --> {:.5f}). Saving model ...'.format(
                  validation_loss_min,
                  validation_loss))
            model_name = 'model_' + str(validation_loss.item())+'_'+str(e)+'.pt' 
            file_name = 'model_' + str(validation_loss.item())+'_'+str(e)+'.csv' 
            torch.save(model.state_dict(), save_dir + '/' + model_name)
            
            report = classification_report(y_true, y_pred, output_dict=True)
            df = pd.DataFrame(report).transpose()
            df.to_csv(save_dir + '/' + file_name, sep='\t')
            validation_loss_min = validation_loss

    return model, train_losses, validation_losses