class SingleTokenPOSTagger(nn.Module):
  """
  Parameters
  ------------
  vocab_size : int
    number of unique characters
  labels_num : int 
    number of tags
  embedding_size : int
    size of embedding vector

  backbone - ResNet layer
  global_pooling - transform matrix into vector by pooling
  out - applies a linear transformation to the incoming data: x*W.T + b
  """
  def __init__(self, vocab_size, labels_num, embedding_size=32, **kwargs):
    super().__init__()
    self.char_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
    self.backbone = StackedConv1d(embedding_size, **kwargs)
    self.global_pooling = nn.AdaptiveMaxPool1d(1)
    self.out = nn.Linear(embedding_size, labels_num)
    self.labels_num = labels_num
    
  def forward(self, tokens):
    """ tokens : tensor (batchSize x maxSentLen x maxTokenLen)  """

    # reduce 3d tensor to 2d one
    batch_size, max_sent_len, max_token_len = tokens.shape
    tokens_flat = tokens.view(batch_size*max_sent_len, max_token_len)

    # build embeddings
    char_embeddings = self.char_embeddings(tokens_flat)  # BatchSize*MaxSentenceLen x MaxTokenLen x EmbSize
    char_embeddings = char_embeddings.permute(0, 2, 1)  # BatchSize*MaxSentenceLen x EmbSize x MaxTokenLen

    # send embeddings to backbone to take into account a context of each character 
    features = self.backbone(char_embeddings)

    # thus we have vectors of features for every character but 
    # we want to tag a token, so let us aggregate characters into tokens
    # by pooling. 

    # global pooling take a matrix (NxM) and build vector (N), where i_th element
    # is max element from i_th column of the matrix.

    global_features = self.global_pooling(features).squeeze(-1) # BatchSize*MaxSentLen x EmbSize

    logits_flat = self.out(global_features) # BatchSize*MaxSentLen x LabelsNum
    
    # add sentence's dimension 
    logits = logits_flat.view(batch_size, max_sent_len, self.labels_num)
    logits = logits.permute(0,2,1) # BatchSize x LabelsNum x MaxSentLen
    return logits

class SentenceLevelPOSTagger(nn.Module):
  """
  Parameters
  ------------
  vocab_size : int
    number of unique characters
  labels_num : int 
    number of tags
  embedding_size : int
    size of embedding vector

  backbone - ResNet layer
  global_pooling - transform matrix into vector by pooling
  out - applies a linear transformation to the incoming data: x*W.T + b
  """
  def __init__(self, vocab_size, labels_num, embedding_size=32, single_backbone_kwargs={}, context_backbone_kwargs={}):
    super().__init__()
    self.embedding_size = embedding_size
    self.char_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
    self.single_token_backbone = StackedConv1d(embedding_size, **single_backbone_kwargs)
    self.context_backbone = StackedConv1d(embedding_size, **context_backbone_kwargs)
    self.global_pooling = nn.AdaptiveMaxPool1d(1)
    self.out = nn.Conv1d(embedding_size, labels_num, 1)
    self.labels_num = labels_num
    
  def forward(self, tokens):
    """ tokens : tensor (batchSize x maxSentLen x maxTokenLen)  """

    # reduce 3d tensor to 2d one
    batch_size, max_sent_len, max_token_len = tokens.shape
    tokens_flat = tokens.view(batch_size*max_sent_len, max_token_len)

    # build embeddings
    char_embeddings = self.char_embeddings(tokens_flat)  # BatchSize*MaxSentenceLen x MaxTokenLen x EmbSize
    char_embeddings = char_embeddings.permute(0, 2, 1)  # BatchSize*MaxSentenceLen x EmbSize x MaxTokenLen

    # send embeddings to backbone to take into account a context of each character 
    char_features = self.single_token_backbone(char_embeddings)

    # thus we have vectors of features for every character but 
    # we want to tag a token, so let us aggregate characters into tokens
    # by pooling. 

    # global pooling take a matrix (NxM) and build vector (N), where i_th element
    # is max element from i_th column of the matrix.

    token_features_flat = self.global_pooling(char_features).squeeze(-1) # BatchSize*MaxSentLen x EmbSize
    
    # features of tokens without it's context
    token_features = token_features_flat.view(batch_size, max_sent_len, self.embedding_size)
    token_features = token_features.permute(0,2,1) # batchSize x EmbSize x MaxSentLen
    
    # recalculate features with respect of context
    context_features = self.context_backbone(token_features)

    logits = self.out(context_features) # BatchSize*MaxSentLen x LabelsNum
    return logits

class POSTagger:
    def __init__(self, model, char2id, id2label, max_sent_len, max_token_len):
        self.model = model
        self.char2id = char2id
        self.id2label = id2label
        self.max_sent_len = max_sent_len
        self.max_token_len = max_token_len

    def __call__(self, sentences):
        print(sentences)
        tokenized_corpus = tokenize_corpus(sentences, min_token_size=0)
        print(tokenized_corpus)
        inputs = torch.zeros((len(sentences), self.max_sent_len, self.max_token_len + 2), dtype=torch.long)

        for sent_i, sentence in enumerate(tokenized_corpus):
            for token_i, token in enumerate(sentence):
                for char_i, char in enumerate(token):
                    print(self.char2id.get(char, 0), "--", char)
                    inputs[sent_i, token_i, char_i + 1] = self.char2id.get(char, 0)

        dataset = TensorDataset(inputs, torch.zeros(len(sentences)))
        predicted_probs = predict_with_model(self.model, dataset)  # SentenceN x TagsN x MaxSentLen
        predicted_classes = predicted_probs.argmax(1)

        result = []
        for sent_i, sent in enumerate(tokenized_corpus):
            result.append([self.id2label[cls] for cls in predicted_classes[sent_i, :len(sent)]])
        return result
