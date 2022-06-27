import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class KerasGraphCallback(tf.keras.callbacks.Callback):
    def __init__(self,num_epochs):
        tf.keras.callbacks.Callback.__init__(self)
        self.num_epochs = num_epochs

    def on_train_begin(self,logs=None):
        if hasattr(self,'fig') == False:
            fig,ax = plt.subplots(2,1,facecolor="white",figsize=(9,7));
            self.fig = fig
            self.ax = ax
            self.hd = display("[...Loading Loss/Accuracy Graph...]", display_id=True);

    def on_epoch_end(self, epoch, logs=None):
        loss = self.model.history.history.get('loss')
        val_loss = self.model.history.history.get('val_loss')
        accuracy = self.model.history.history.get('accuracy')
        val_accuracy = self.model.history.history.get('val_accuracy')

        if val_accuracy is not None:
            
            x = list(range(2,len(val_accuracy)+2));
            
            self.ax[0].clear()
            self.ax[1].clear()

            self.ax[0].set_xlim(1, max(len(val_loss)+2+math.floor(len(val_loss)/10),6))
            self.ax[1].set_xlim(1, max(len(val_accuracy)+2+math.floor(len(val_accuracy)/10),6))
            
            y_lim_loss = 1.1*max(loss+val_loss)
            y_lim_accuracy = 1.1*max(accuracy+val_accuracy)
            
            self.ax[0].set_ylim(0, 1.1*max(loss+val_loss))
            self.ax[1].set_ylim(0, 1.1*max(accuracy+val_accuracy))
            
            if len(x) == 1:
                self.ax[0].scatter(x, loss, label="Training Loss", color="blue")
                self.ax[0].scatter(x, val_loss, label="Validation Loss", color="orange")
                self.ax[1].scatter(x, accuracy, label="Training Accuracy", color="green")
                self.ax[1].scatter(x, val_accuracy, label="Validation Accuracy", color="red")           
            else:
                self.ax[0].plot(x, loss, label="Training Loss", color="blue",marker='o')
                self.ax[0].plot(x, val_loss, label="Validation Loss", color="orange",marker='o')
                self.ax[1].plot(x, accuracy, label="Training Accuracy", color="green",marker='o')
                self.ax[1].plot(x, val_accuracy, label="Validation Accuracy", color="red",marker='o')
            
            self.fig.suptitle(f'Training Epoch ({len(x)+1}/{self.num_epochs})', fontsize=16)
            self.ax[0].set_title("Loss")
            self.ax[1].set_title("Accuracy")
            self.ax[0].legend(loc='lower left')
            self.ax[1].legend(loc='lower left')
            self.ax[0].grid()
            self.ax[1].grid()
            
            self.ax[0].text(x[-1]+0.05, loss[-1] - 0.06*y_lim_loss, '({:4f})'.format(loss[-1]),color="blue")
            self.ax[0].text(x[-1]+0.05, val_loss[-1] + 0.03*y_lim_loss, '({:4f})'.format(val_loss[-1]),color="orange")
            self.ax[1].text(x[-1]+0.05, accuracy[-1] - 0.06*y_lim_accuracy, '({:4f})'.format(accuracy[-1]),color="green")
            self.ax[1].text(x[-1]+0.05, val_accuracy[-1] + 0.03*y_lim_accuracy, '({:4f})'.format(val_accuracy[-1]),color="red")
            
            self.hd.update(self.fig);

    def on_train_end(self, epoch, logs=None):
        plt.close(self.fig)