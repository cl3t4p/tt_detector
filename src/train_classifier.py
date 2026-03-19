import argparse 
import lightning
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from classifier import AudioClassifier
from dataset import SoundDataModule



def main():
    parser = argparse.ArgumentParser(description="Train bounce classifier")

    parser.add_argument('--classify',choices=['surface','spin'],default='surface')
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--lr',type=int,default=1e-3)
    parser.add_argument('--data_dir',type=str,default='data')
    parser.add_argument('--seed',type=int,default=42)

    args = parser.parse_args()

    lightning.seed_everything(args.seed)


    model = AudioClassifier(task=args.classify, learning_rate=args.lr)
    dm = SoundDataModule(data_dir=args.data_dir, batch_size=args.batch_size)

    wandb_logger = WandbLogger(
        project="tt_detector",
        name=f"{args.classify}-lr{args.lr}",
        log_model="all",   # logs checkpoints as W&B artifacts
    )

    callbacks = [
            ModelCheckpoint(
                dirpath='models',
                filename=f'{args.classify}_best',
                monitor='val_acc',
                mode='max',
                save_top_k=1,
                ),
            EarlyStopping(monitor='val_loss',patience=20,mode='min')
            ]

    trainer = lightning.Trainer(
            max_epochs=args.epochs,
            callbacks=callbacks,
            deterministic=True,
            accelerator='auto'
            )

    # optional: log gradients / parameter histograms / graph
    wandb_logger.watch(model, log="all", log_freq=100)

    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)



if __name__ == '__main__':
    main()
