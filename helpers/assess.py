import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.utils.class_weight import compute_sample_weight


def make_confusion_matrix(
  y_true,
  y_pred=None, # optionally pass in precalculated y predictions
  model=None, # optionally pass in a model (along with x) to autogenerate y_pred
  x=None, # optionally pass in a feature set (along with model) to autogenerate y_pred
  normalize="true", # parameter for the sklearn confusion matrix function specifying how to normalize the values
  sample_weight=None, # optionally provide precalculated sample weights
  autoweight_sample=True, # generate sample weights automatically before calculating matrix
  print_heatmap=True, # print the heatmap for confusion matrix
  label_names=None, # optionally provide an explicit list of label names for the heatmap
  label_encoder=None, # a label encoder for the to automatically get the label names
  figsize=(9,9), # size of the heatmap displayed
  title=None, # printed title of the heatmap
):

  # resolve polymorphisms / optional values
  y_pred = resolve_y_pred(
    y_pred=y_pred,
    model=model,
    x=x,
  )
  sample_weight = resolve_sample_weight(
    y_true,
    sample_weight,
    autoweight_sample
  )
  label_names = resolve_label_names(
    y_true,
    y_pred,
    model=model,
    label_names=label_names,
    label_encoder=label_encoder,
  )

  # calculate the confusion matrix
  cm = confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
    normalize=normalize,
    sample_weight=sample_weight,
  )

  # print a heatmap
  if print_heatmap:
    plt.figure(
      figsize=figsize,
      facecolor="white"
    )
    sns.heatmap(
      cm,
      annot=True,
      fmt=".3f",
      linewidths=0.5,
      square=True,
      cmap="Blues_r",
      xticklabels=label_names,
      yticklabels=label_names,
  )
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    if title:
      plt.title(title)

  # return the 2D confusion matrix array
  return cm

# resusable helper function to get a y_pred array from various polymorphisms
def resolve_y_pred(
  y_pred=None,
  model=None,
  x=None
):
  if y_pred:
    return y_pred
  elif model and x is not None:
    return model.predict(x)
  else:
    raise Exception('Both a model and x (feature set) must be passed if a y_pred is not provided.')

# resusable helper function to get a label_names array from various polymorphisms
def resolve_label_names(
  y_true,
  y_pred,
  model=None,
  label_names=None,
  label_encoder=None,
):

  # if no label names are explicitly provided
  if not label_names:
    # get a unique list of labels in the matrix
    unique_labels = None
    if model and model.classes_ is not None:
      unique_labels = list(model.classes_)
    else:
      np.sort(np.unique(np.concatenate([y_true,y_pred])))
    if label_encoder:
      # convert label to friendly names if encoder is provided
      label_names = label_encoder.inverse_transform(unique_labels)
    else:
      # otherwise use the raw (usually integer) label values
      label_names = [str(l) for l in unique_labels]
  return label_names

# resusable helper function to get a sample_weight array from various polymorphisms
def resolve_sample_weight(
  y_true,
  sample_weight=None,
  autoweight_sample=None,
):
  # compute the weights for each class if not explicitly provided
  if not sample_weight and autoweight_sample:
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_true)
  return sample_weight


def make_classification_report(
  # same params as sklearn.metrics.classification_report
  y_true,
  y_pred,
  labels=None,
  target_names=None,
  sample_weight=None,
  digits=2,
  output_dict=False,
  zero_division="warn",
):

  cr = classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=target_names,
    sample_weight=sample_weight,
    digits=digits,
    output_dict=True,
    zero_division=zero_division
  )

  # fix some bizarre parsing irregularity
  if type(cr['accuracy']) != 'dict':
    cr['accuracy'] =  {'f1-score': cr['accuracy'], 'support':cr['weighted avg']['support']}

  # generate mcc for each class
  for i in range(len(target_names)):
      cr[target_names[i]]['mcc'] = matthews_corrcoef(y_true == i,y_pred == i)

  # generate multiclass mcc
  cr['accuracy']['mcc'] = matthews_corrcoef(y_true,y_pred)

  # return dictionary as-is if requested
  if output_dict:
    return cr

  # parse a table to a string

  # put in empty key to simulate newline been actual labels and metalabels
  cr[''] = {}

  # format all floats as strings
  cr_arr = [{'label':label,**{metric_key:('{:.'+str(0 if metric_key == 'support' else digits)+'f}').format(metric_value)
             for metric_key,metric_value in l_object.items()}}
               for label,l_object in cr.items()]

  # split labels into real labels and meta labels for calculating sort order
  meta_labels = ['','accuracy','macro avg','weighted avg']
  real_labels = [k for k in cr.keys() if k not in meta_labels]
  real_labels.sort()

  # sort all table rows
  cr_arr.sort(key=lambda x: real_labels.index(x['label']) if x['label'] in real_labels else len(real_labels) + meta_labels.index(x['label']))

  cols = ['label','precision','recall','f1-score','support','mcc']

  # prepend header row and simulated newline
  cr_arr = [{c:c for c in cols if c != 'label'},{}] + cr_arr

  # calculate padding necessary to align all rows
  paddings = {col:max([len(r.get(col,'')) for r in cr_arr])+2 for col in cols}

  return "\n".join([
      ''.join([
          r.get(c,'').rjust(paddings[c],' ')
       for c in cols]) 
  for r in cr_arr])