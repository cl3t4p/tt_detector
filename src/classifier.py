import torch
from typing import Literal
import torch.nn as nn
import lightning
from torchmetrics import Accuracy
import torchmetrics


__valid_classes = {
        'surface' : 13,
        'spin' : 3
        }


class AudioClassifier(lightning.LightningModule):

    def __init__(self,task : Literal['surface','spin'],learning_rate: float = 1e-3):
        super().__init__()
        if(task not in __valid_classes):
            raise ValueError(
                    f"Unknow task '{task}'. Expected one of {sorted(__valid_classes)}"
                    )
        self.learning_rate = learning_rate
        num_classes = __valid_classes[task]


        self.save_hyperparameters({
            "learning_rate" :learning_rate,
            "num_classes" : num_classes,
            "task" : task
            })


        self.features = nn.Sequential(
                # Layer 1
                nn.Conv2d(1,2,kernel_size=5,stride=(2,2),padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(2),
                # Layer 2
                nn.Conv2d(2,4,kernel_size=3,stride=(2,1),padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(4),
                # Layer 3
                nn.Conv2d(4,8,kernel_size=3,stride=(2,1),padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(8),
                # Layer 4
                nn.Conv2d(8,16,kernel_size=3,stride=(2,1),padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                # Layer 5
                nn.Conv2d(16,32,kernel_size=3,stride=(2,1),padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                # LOayer 6
                nn.Conv2d(32,64,kernel_size=3,stride=(2,1),padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                # Pooling
                nn.AdaptiveAvgPool2d(1),
                )
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64,num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()


        # Metrics
        self.train_acc = Accuracy(task="multiclass",num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass",num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass",num_classes=num_classes,average='macro')
        self.val_precision = torchmetrics.Precision(task='multiclass',num_classes=num_classes,average='macro')
        self.val_recall = torchmetrics.Recall(task='multiclass',num_classes=num_classes,average='macro')

        # Self init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,x):
        featuers = self.features(x)
        result = self.classifier(featuers)
        return result


    def _shared_step(self,batch,stage):
        mel,surface,spin = batch
        logits = self(mel)


        if(self.hparams.task == "surface"):
            target = surface
        else:
            target = spin
        
        loss = self.loss_fn(logits,target)
        preds = logits.argmax(dim=1)
        return loss,preds,target

    def training_step(self,batch,batch_idx):
        loss,preds,target = self._shared_step(batch,'train')
        self.train_acc(preds,target)
        
        # Log
        self.log('train_loss',loss,prog_bar=True)
        self.log('train_acc',self.train_acc,prog_bar=True)

    def validation_step(self,batch,batch_idx):
        loss,preds,target = self._shared_step(batch,'val')
        self.val_acc(preds,target)
        self.val_f1(preds,target)
        self.val_precision(preds,target)
        self.val_recall(preds,target)

        # Log
        self.log('val_loss',loss,prog_bar=True)
        self.log('val_acc',self.val_acc,prog_bar=True)
        self.log('val_f1',self.val_f1)
        self.log('val_precision',self.val_precision)
        self.log('val_recall',self.val_recall)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

