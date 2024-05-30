"""
Classes to convert tokenized, integerized inputs to
representations, using either PyTorch neural networks
through Huggingface or through one-hot encodings of 
"""
from yaml import YAMLObject

import torch.nn as nn
import torch
from transformers import AutoModel, AutoConfig, BertModel
#from allennlp.modules.elmo import _ElmoBiLm

from utils import InitYAMLObject

class ELMoModel(nn.Module, InitYAMLObject):
    """
    Taking ELMo-formatted character sequences
    and providing representation layers from an
    ELMo model, as provided by allennlp.
    """
    yaml_tag = '!ELMoModel'
    def __init__(self, args, options_path, weights_path, layer_index, trainable):
      super(ELMoModel, self).__init__()
      self.args = args
      self.options_path = options_path
      self.weights_path = weights_path
      self.layer_index = layer_index
      self.elmo_bilm = _ElmoBiLm(options_path, weights_path, trainable)
      self.to(args['device'])

    def forward(self, batch):
      """
      Computes ELMo representations from a batch of character ids

      Arguments:
        batch: a torch.tensor of shape
        (batchlen, charseqlen, max_word_length)
      Returns:
        A batch of ELMo representations for the corpus-given
        tokens, shape (batchlen, corpus_seq_len, feature_count)
      """
      annotation, alignment = batch
      batch = self.elmo_bilm(annotation)['activations'][self.layer_index]
      batch = batch[:,1:-1,:] # Remove SOS/EOS tokens from begin/end
      return batch



class HuggingfaceModel(nn.Module, InitYAMLObject):
  """
  Taking token-ids and providing representation
  layers from a Huggingface-loaded model
  """
  yaml_tag = '!HuggingfaceModel'
  def __init__(self, args, model_string, trainable, index):
    """
    Arguments:
      args: The arguments dictionary
      model_string: The huggingface-specfied model identifier string,
                    e.g., google/bert-base
    """
    super(HuggingfaceModel, self).__init__()
    self.huggingface_config = AutoConfig.from_pretrained(model_string, output_hidden_states=True, output_norms = True)
    self.huggingface_model = AutoModel.from_pretrained(model_string, config=self.huggingface_config)
    for param in self.huggingface_model.parameters():
      param.requires_grad = trainable
    self.index = index
    self.to(args['device'])

  def forward(self, batch):
    """
    Args:
      batch: a tuple containing:
        - a tensor of shape (batch_size, subword_seq_len) containing
          token id indices
        - a tensor of shape (subword_len, seq_len) containing
          alignment between the subwords and the output sequence
    Returns:
      Representations from the given huggingface-formatted model, aligned
      to the corpus-given tokens
    """
    annotation, alignment = batch
    _, _, hiddens = self.huggingface_model(annotation)
    #hiddens = self.huggingface_model(annotation).hidden_states
    #print(hiddens[self.index].size())
    return torch.bmm(hiddens[self.index].transpose(1,2), alignment).transpose(1,2)
  

class BertATTNKobayashiModel(nn.Module, InitYAMLObject):
  """
  Taking token-ids and providing representation
  layers from a BERT model. Implemented separately
  to account for Kobayashi's transformers library
  """
  yaml_tag = '!BertATTNKobayashiModel'

  def pad_to_feature_count(self, tensor, feature_count):
    pad_width = (0, feature_count - tensor.size(2))
    padded_tensor = nn.functional.pad(tensor, pad_width, mode='constant', value=0)
    return padded_tensor

  def __init__(self, args, model_string, trainable, index):
    super(BertATTNKobayashiModel, self).__init__()
    self.bert_model = BertModel.from_pretrained(model_string)
    for param in self.bert_model.parameters():
      param.requires_grad = trainable
    self.index = index
    self.to(args['device'])
  
  def forward(self, batch):
    """
    Args:
      batch: a tuple containing:
        - a tensor of shape (batch_size, subword_seq_len) containing
          token id indices
        - a tensor of shape (subword_len, seq_len) containing
          alignment between the subwords and the output sequence
    Returns:
      Representations from the given huggingface-formatted model, aligned
      to the corpus-given tokens
    """
    #print("Enter Model....")
    annotation, alignment = batch
    #_, _, hiddens = self.huggingface_model(annotation)
    if annotation.shape[1] > 512:
      annotation = annotation[:, :512]
    _, _, norms = self.bert_model(annotation, output_norms = True)
    norms_padded = self.pad_to_feature_count(norms[self.index][1], 768)
    #norms_padded = norms[self.index][1]
    return torch.bmm(norms_padded.transpose(1,2), alignment).transpose(1,2)

class BertATTNRESKobayashiModel(nn.Module, InitYAMLObject):
  """
  Taking token-ids and providing representation
  layers from a BERT model. Implemented separately
  to account for Kobayashi's transformers library
  """
  yaml_tag = '!BertATTNRESKobayashiModel'

  def pad_to_feature_count(self, tensor, feature_count):
    pad_width = (0, feature_count - tensor.size(2))
    padded_tensor = nn.functional.pad(tensor, pad_width, mode='constant', value=0)
    return padded_tensor

  def __init__(self, args, model_string, trainable, index):
    super(BertATTNRESKobayashiModel, self).__init__()
    self.bert_model = BertModel.from_pretrained(model_string)
    for param in self.bert_model.parameters():
      param.requires_grad = trainable
    self.index = index
    self.to(args['device'])
  
  def forward(self, batch):
    """
    Args:
      batch: a tuple containing:
        - a tensor of shape (batch_size, subword_seq_len) containing
          token id indices
        - a tensor of shape (subword_len, seq_len) containing
          alignment between the subwords and the output sequence
    Returns:
      Representations from the given huggingface-formatted model, aligned
      to the corpus-given tokens
    """
    #print("Enter Model....")
    annotation, alignment = batch
    #_, _, hiddens = self.huggingface_model(annotation)
    if annotation.shape[1] > 512:
      annotation = annotation[:, :512]
    _, _, norms = self.bert_model(annotation, output_norms = True)
    norms_padded = self.pad_to_feature_count(norms[self.index][2], 768)
    return torch.bmm(norms_padded.transpose(1,2), alignment).transpose(1,2)

class BertATTNRESLNKobayashiModel(nn.Module, InitYAMLObject):
  """
  Taking token-ids and providing representation
  layers from a BERT model. Implemented separately
  to account for Kobayashi's transformers library
  """
  yaml_tag = '!BertATTNRESLNKobayashiModel'

  def pad_to_feature_count(self, tensor, feature_count):
    pad_width = (0, feature_count - tensor.size(2))
    padded_tensor = nn.functional.pad(tensor, pad_width, mode='constant', value=0)
    return padded_tensor

  def __init__(self, args, model_string, trainable, index):
    super(BertATTNRESLNKobayashiModel, self).__init__()
    self.bert_model = BertModel.from_pretrained(model_string)
    for param in self.bert_model.parameters():
      param.requires_grad = trainable
    self.index = index
    self.to(args['device'])
  
  def forward(self, batch):
    """
    Args:
      batch: a tuple containing:
        - a tensor of shape (batch_size, subword_seq_len) containing
          token id indices
        - a tensor of shape (subword_len, seq_len) containing
          alignment between the subwords and the output sequence
    Returns:
      Representations from the given huggingface-formatted model, aligned
      to the corpus-given tokens
    """
    #print("Enter Model....")
    annotation, alignment = batch
    #_, _, hiddens = self.huggingface_model(annotation)
    if annotation.shape[1] > 512:
      annotation = annotation[:, :512]
    _, _, norms = self.bert_model(annotation, output_norms = True)
    norms_padded = self.pad_to_feature_count(norms[self.index][3], 768)
    return torch.bmm(norms_padded.transpose(1,2), alignment).transpose(1,2)

class AnnotationModel(nn.Module, InitYAMLObject):
  yaml_tag = '!AnnotationModel'

  def __init__(self, args, max_cardinality, trainable):
    super(AnnotationModel, self).__init__()
    self.args = args
    self.embeddings = nn.Embedding(num_embeddings=max_cardinality, embedding_dim=max_cardinality, 
        _weight=nn.Parameter(torch.eye(max_cardinality)))
    self.embeddings.weight.requires_grad = trainable

    # Set the pad token to 0
    self.embeddings.weight.data[0,0] = torch.tensor(0)
    self.to(args['device'])

  def forward(self, batch):
    """
    Args:
      batch: a tuple containing:
          - a tensor of shape (batch_size, seq_len) containing
            annotation indices
          - a tensor of shape (seq_len, seq_len) containing the
            token alignment (which is just the identity, so ignored)
    Returns:
      Embedding of each annotation (optionally learnable)
    """
    annotation, alignment = batch
    return self.embeddings(annotation)

class ListModel(nn.Module, InitYAMLObject):
  """
  Container class for collecting multiple models,
  each having its own annotation, and concatenating
  their resulting representations for output to the
  probe.
  """
  yaml_tag = '!ListModel'

  def __init__(self, args, models):
    super(ListModel, self).__init__()
    self.args  = args
    self.models = nn.ModuleList(models)
    self.to(args['device'])

  def forward(self, batch):
    """
    Calls all underlying models and concatenates their outputs
    in the feature index.
    Arguments:
      batch: ((annotation_batch_1, ...), (alignment_batch_1, ...)
    Output:
      representations: (batch_size, seq_len, model_1_size + model_2_size + ... )
    """
    representations = []
    annotations, alignments = batch
    for index, input_model in enumerate(self.models):
      representations.append(self.models[index]((annotations[index], alignments[index])))
    representations = torch.cat(representations, dim=2)
    return representations



