import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class KerasGraphCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self,logs=None):
        if hasattr(self,'fig') == False:
            fig,ax = plt.subplots(2,1,facecolor="white");
            self.fig = fig
            self.ax = ax
            self.hd = display("[...Loading Loss/Accuracy Graph...]", display_id=True);
            self.ax[0].set_xlim(0, 1) 
            self.ax[1].set_xlim(0, 1) 
            self.ax[0].plot([], [])
            self.ax[1].plot([], [])
            self.ax[0].set_title("Loss")
            self.ax[1].set_title("Accuracy")
            print('Beginning fit...')

    def on_epoch_end(self, epoch, logs=None):
        loss = self.model.history.history.get('loss')
        val_loss = self.model.history.history.get('val_loss')
        accuracy = self.model.history.history.get('accuracy')
        val_accuracy = self.model.history.history.get('val_accuracy')

        if val_accuracy is not None:
            
            x = list(range(0,len(val_accuracy)+1));

            self.ax[0].set_xlim(0, len(val_loss))
            self.ax[1].set_xlim(0, len(val_accuracy))
            
            self.ax[0].set_ylim(0, 1.1*max(loss+val_loss))
            self.ax[1].set_ylim(0, 1.1*max(accuracy+val_accuracy))
            
            loss = [loss[0]] + loss
            val_loss = [val_loss[0]] + val_loss
            accuracy = [accuracy[0]] + accuracy
            val_accuracy = [val_accuracy[0]] + val_accuracy

            self.ax[0].plot(x, loss, label="Training Accuracy", color="blue")
            self.ax[0].plot(x, val_loss, label="Validation Accuracy", color="orange")
            self.ax[1].plot(x, accuracy, label="Training Accuracy", color="green")
            self.ax[1].plot(x, val_accuracy, label="Validation Accuracy", color="red")
            
            if len(x) == 2:
                self.ax[0].legend()
                self.ax[1].legend()

            self.hd.update(self.fig);