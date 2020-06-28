from keras.callbacks import callbacks, ModelCheckpoint, CSVLogger, \
    EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from functools import partial
import math


# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None, save_best_only=True):
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True, verbose=1))
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        # step_decay函数，该函数以epoch号为参数（从0算起的整数），返回一个新学习率（浮点数）
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        # 当评价指标不在提升时，减少学习率
        # 该回调函数检测指标的情况，如果在patience个epoch中看不到模型性能提升，则减少学习率
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    return callbacks