import numpy as np

#import import_ipynb
import training_run as FWL_class


alpha = np.arange(0.0, 1.1, 0.1)
for a in alpha:
    print("#############################################################################\n {}\n#############################################################################\n".format(a))
    model_Fixed_weight =  FWL_class.Fixed_weight_loss_MFC(dataset="cifar10", batch_size = 128, n_epochs = 100, bool_alpha= True, alpha = a)
    model_Fixed_weight.procedure_train()
    del model_Fixed_weight


