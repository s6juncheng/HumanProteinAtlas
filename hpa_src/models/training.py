import time
from sklearn.metrics import f1_score
from hpa_src.data.functional import preds2label, preds2onehot, array2str
from hpa_src.models.utils import AverageMeter
from keras.callbacks import History

def train(model, 
          train_loader, 
          criterion, 
          optimizer):
    ''' Train model for one epoch
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    f1 = AverageMeter()

    model.train()

    end = time.time()
    
    # Iterate over data.
    for inputs, labels in train_loader:
        # measure data loading time
        data_time.update(time.time() - end)
        
        # to gpu
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        outputs = model(inputs)
        # to class prediction to caulucate f1
        preds = preds2onehot(outputs)  
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()

        # statistics
        losses.update(loss.item(), inputs.size(0))
        f1.update(f1_score(labels, preds, average='macro'))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return {
        'train_loss': losses.avg,
        'train_f1': f1.avg
    }

def evaluate(model, val_loader, criterion):
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            # to gpu
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            # to class prediction to caulucate f1
            preds = preds2onehot(outputs)  
            loss = criterion(outputs, labels)
            losses.update(loss.item(), inputs.size(0))
            val_preds.append(preds)
            val_true.append(labels.cpu())
    val_preds = np.concatenate(val_preds)
    val_true = np.concatenate(val_true)
    f1 = f1_score(val_true, val_preds, average='macro')
    return {
        'val_loss': losses.avg,
        'val_f1': f1
    }


class ModelTrainer(object):
    def __init__(self,
                model,
                ):
        self.model = model
    
    def compile(self,
               optimizer,
               loss):
        self.optimizer = optimizer
        self.criterion = loss
    
    def fit(self, 
            train_loader, 
           val_loader,
           model_checker=None,
           epochs=100):
        self.history = History()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.history.on_train_begin()
        
        for epoch in range(epochs):
            train_logs = train(self.model,
                               train_loader,
                               self.criterion, 
                               self.optimizer)
            # evaluate
            val_logs = evaluate(self.model,
                                 val_loader,
                                 self.criterion)
            self.history.on_epoch_end(epoch, {**train_logs, **val_logs})
            if model_checker is not None:
                model_checker.on_epoch_end(epoch, {**train_logs, **val_logs})