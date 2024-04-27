```python
import torch
from transformers.file_utils import is_tf_available, is_torch_available

import numpy as np
import random
import pandas as pd
import re

import mlflow
import torch
mlflow.end_run()
torch.cuda.empty_cache()

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed). Taken from https://www.thepythoncode.com/article/finetuning-bert-using-huggingface-transformers-python 
 
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf
 
        tf.random.set_seed(seed)
```


```python
set_seed(42)
from sklearn.model_selection import train_test_split
import pandas as pd
!pip install bs4 --user
from bs4 import BeautifulSoup

def clean_vector(v):
        v = np.where(v > 900000000, 0, v)
        return v


def read_data():
  dataset = pd.read_csv("MV_data_final_features_topics.csv")
  dataset['content'] =  dataset['content'].apply(lambda x: BeautifulSoup(x).get_text())
  print(list(dataset['content'])[0])




  topic_vectors_prep = [vector.strip("[").strip("]") for vector in list(dataset["topic_vector"])]
  topic_vectors = [np.fromstring(vector, dtype=float, sep=',').astype(np.float32) for vector in topic_vectors_prep]
  print(type(topic_vectors[0][0]))


  feature_vectors_prep = [np.fromstring(vector, dtype=float, sep=",").astype(np.float32) for vector in list(dataset["feature_vector"])]
  print(f"feature vec = {feature_vectors_prep[0]}")
    
  feature_vectors = [clean_vector(v) for v in feature_vectors_prep]


  new_labels = []
  for c in list(dataset['class']):
    if c == 1: # kritiikki
      new_labels.append(0)
    elif c == 2: # kopiointi
      new_labels.append(1)
    elif c == 4: # oma narratiivi
      new_labels.append(2)
  documents = list(dataset['content'])

  ids = list(dataset["id"])
  print(len(ids))

  return train_test_split(documents, new_labels, topic_vectors, feature_vectors, ids, random_state=42)
  
# call the function
(train_texts, valid_texts, train_labels, valid_labels, train_topics, valid_topics, train_features, valid_features, train_ids, valid_ids) = read_data()
class_labels = ["kritiikki","kopiointi","oma narratiivi"]

(train_classes, train_class_counts) = np.unique(train_labels,return_counts=True)
(valid_classes, valid_class_counts) = np.unique(valid_labels,return_counts=True)
(all_classes, all_class_counts) = np.unique(np.concatenate((train_labels,valid_labels)),return_counts=True)
```

    Collecting bs4
      Using cached bs4-0.0.2-py2.py3-none-any.whl (1.2 kB)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.9/site-packages (from bs4) (4.12.3)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.9/site-packages (from beautifulsoup4->bs4) (2.5)
    Installing collected packages: bs4
    Successfully installed bs4-0.0.2
    Verkkouuutiset uutisoi tänään, että Perussuomalaisten Teuvo Hakkarainen puhui tiistaina eduskunnassa ulkomaalaislain käsittelyssä.
    Hakkaraisen mielestä esityksen sisältämät asiat ovat positiivisia askelia, mutta eivät riittäviä.
    Teuvo Hakkarainen.
    Teuvo Hakkarainen sanoi:
    ”Valtaosa turvapaikkaturisteista on tullut ainakin kymmenen turvallisen maan läpi Suomeen, eikä heillä kotimaassakaan ole ollut konkreettista henkeen ja terveyteen kohdistuvaa uhkaa, vaikka kaikenlaisia tarinoita he ovatkin oppineet kertomaan.
    Enimmäkseen he ovat ilmaisen sosiaaliturvan perässä reissaavia nuoria miehiä, joita ei kiinnosta rakentaa omaa isänmaataan. Heitä kiinnostaa siivestäminen.”
    ”Koska alun alkaenkaan he eivät täytä kansainvälistä suojelua koskevia kriteereitä, en tiedä, miksi heillä ylipäänsä pitäisi olla valitusoikeudet.
    Valitusoikeus kuormittaa oikeuslaitostamme kohtuuttomasti. Se ensimmäinenkin hakemus pitäisi tehdä pikapäätöksenä rajalla.
    Jos asiallisia henkilöpapereita ei ole, hakemusta ei pitäisi käsitellä lainkaan vaan suorittaa saman tien pikakäännytys ja sellainen leima takapuoleen, ettei takaisin tule.”
    Hakkarainen kysyi, mihin oikeuslaitoksemme joutuu, jos tänne tulee vielä kymmenkertainen määrä tulijoita.
    Hakkarainen totesi:
    ”Minä arvioin, että todellisia hädänalaisia on meille tullut jokunen kymmenen, ehkä sata. Heitä meidän tulee auttaa, mikäli heidän tarinansa on aukottomasti todettavissa. Kaikkien muiden suhteen pikakäännytykset ovat ainoa oikeudenmukainen ratkaisu.”
    Hakkarainen viittasi puheessaan Australiaan, joka:
    ”ei salli turvapaikkaturisteja kuljettavien veneiden rantautua ollenkaan maaperälleen. Australian laivasto saattaa veneet lähimmälle saarelle, joka ei kuulu Australialle.”
    
    Hakkarainen sanoi lopuksi:
    ”Naurusaaret ja Papua-Uusi-Guinea ovat kohdemaat, jonne perustettuihin leireihin tulijat pistetään odottamaan. Siellä täytetään hakemukset ja siellä myös odotellaan päätöksiä niistä.
    EU:n tulisi ottaa mallia Australiasta. Koska EU:n päätöksiä ja jäsenmaiden yhtenäistä päätösten täytäntöönpanoa saamme ilmeisesti odottaa hamaan hautaan asti, tulisi meidän kyetä itsenäiseen päätöksentekoon. Suomi ei ole mikään maailman sosiaalitoimisto.”
    Lähde: verkkouutiset.fi (Juha-Pekka Tikka 5.4. 2016)
    T3/MV
    toimitus3mv@gmail.com
    
    <class 'numpy.float32'>
    feature vec = [1.8000000e+01 1.8000000e+01 1.0000000e+00 1.6411500e+05 1.4359950e+05
     4.1033067e-03 0.0000000e+00 5.3200000e+02 9.0266669e+02 2.2210000e+03
     2.2210000e+03 4.1322714e-01 1.2940000e+03 8.5600000e+02 0.0000000e+00
     0.0000000e+00 9.0000002e+12 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 9.0000002e+12 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 9.0000002e+12 0.0000000e+00
     0.0000000e+00 0.0000000e+00 3.5900000e+02 3.5900000e+02 3.5900000e+02
     5.0591362e-01 1.1260000e+03 1.0480000e+03 0.0000000e+00 0.0000000e+00
     9.0000002e+12 0.0000000e+00 0.0000000e+00 0.0000000e+00 5.9533331e+02
     1.3620000e+03 1.3300000e+02 4.3808833e-01 1.0470000e+03 4.5375000e+02
     0.0000000e+00 0.0000000e+00 9.0000002e+12 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 9.0000002e+12 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 9.0000002e+12
     5.4501569e-01 2.2580000e+03 2.2580000e+03 0.0000000e+00 0.0000000e+00
     9.0000002e+12 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 9.0000002e+12 0.0000000e+00 0.0000000e+00 0.0000000e+00]
    997



```python
set_seed(42)
def split_encodings(labels, topics, features, ids, encodings, max_length):
    new_labels = []
    new_input_ids = []
    new_token_type_ids = []
    new_attention_mask = []
    new_topic_vectors = []
    new_feature_vectors = []
    new_ids = []
    
    input_ids = encodings['input_ids']
    token_type_ids = encodings['token_type_ids']
    attention_mask = encodings['attention_mask']
    
    for index,label in enumerate(labels):
        cur_input_ids = input_ids[index]
        cur_token_type_ids = token_type_ids[index]
        cur_attention_mask = attention_mask[index]

        while len(cur_input_ids)>max_length:
            new_input_ids.append(cur_input_ids[0:max_length])
            new_token_type_ids.append(cur_token_type_ids[0:max_length])
            new_attention_mask.append(cur_attention_mask[0:max_length])
            new_labels.append(label)
            new_topic_vectors.append(topics[index])
            new_feature_vectors.append(features[index])
            new_ids.append(ids[index])
            
            cur_input_ids = cur_input_ids[max_length:]
            cur_token_type_ids = cur_token_type_ids[max_length:]
            cur_attention_mask = cur_attention_mask[max_length:]
            
        if len(cur_input_ids)>0:
            new_labels.append(label)
            new_topic_vectors.append(topics[index])
            new_feature_vectors.append(features[index])
            new_ids.append(ids[index])
            new_input_ids.append(np.lib.pad(cur_input_ids,(0,max_length-len(cur_input_ids)),constant_values=(0)))
            new_token_type_ids.append(np.lib.pad(cur_token_type_ids,(0,max_length-len(cur_input_ids)),constant_values=(0)))
            new_attention_mask.append(np.lib.pad(cur_attention_mask,(0,max_length-len(cur_input_ids)),constant_values=(0)))   
        
    return (new_labels, new_topic_vectors, new_feature_vectors, new_ids,{
        'input_ids': new_input_ids,
        'token_type_ids': new_token_type_ids,
        'attention_mask': new_attention_mask
    })


model_name = "TurkuNLP/bert-base-finnish-uncased-v1" 
max_length = 512

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained(model_name)

train_encodings = tokenizer(train_texts)

(train_snippets_labels, train_snippets_topics, train_snippets_features, train_snippets_ids, train_snippets_encodings) = split_encodings(train_labels, 
                                                                                                                                        train_topics, 
                                                                                                                                        train_features,
                                                                                                                                        train_ids,
                                                                                                                                        train_encodings, 
                                                                                                                                    
                                                                                                                                        max_length)

valid_encodings = tokenizer(valid_texts)

(valid_snippets_labels, valid_snippets_topics, valid_snippets_features, valid_snippets_ids, valid_snippets_encodings) = split_encodings(valid_labels,
                                                                                                                                       valid_topics,
                                                                                                                                       valid_features,
                                                                                                                                       valid_ids,
                                                                                                                                       valid_encodings,
                                                                                                                                       max_length)

(all_snippets_classes, all_snippets_class_counts) = np.unique(np.concatenate((train_snippets_labels,valid_snippets_labels)),return_counts=True)
(len(train_labels),len(train_snippets_labels),len(valid_labels),len(valid_snippets_labels),all_snippets_classes, all_snippets_class_counts)


```

    Token indices sequence length is longer than the specified maximum sequence length for this model (954 > 512). Running this sequence through the model will result in indexing errors





    (747, 1144, 250, 367, array([0, 1, 2]), array([ 137, 1040,  334]))




```python
# What is below is mostly a copy of BertForSequenceClassification, but with an added class_weights parameter, 
# which gets used to tune the loss function. While this could be done in other ways (compute_loss in Trainer)
# this copy also acts as a useful insight into what actually happens within the classifier

from transformers.models.bert import BertPreTrainedModel,BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.configuration_bert import BertConfig
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class BertConfigWithClassWeights(BertConfig):
    def __init__(
        self,
        class_weights = None,
        freeze_bert_weights = False,
        use_topics = False,
        use_features = False,
        use_embeddings = False,
        **kwargs
    ):
        self.class_weights = class_weights
        self.freeze_bert_weights = freeze_bert_weights
        self.use_topics = use_topics
        self.use_features = use_features
        self.use_embeddings = use_embeddings
        super().__init__(**kwargs)

class BertForWeightedSequenceClassification(BertPreTrainedModel):
    config_class = BertConfigWithClassWeights
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.class_weights = torch.tensor(self.config.class_weights) if self.config.class_weights else None
        if torch.cuda.is_available():
            self.class_weights = self.class_weights.to("cuda")
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        print("arvot")
        print(config.hidden_size, config.num_labels)
        
                # use topics and embeddings
        if self.config.use_topics:
            if self.config.use_embeddings:
                self.classifier = nn.Sequential(
                                                nn.Linear(config.hidden_size + 200, config.hidden_size + 200),
                                                nn.ReLU(),
                                                nn.Dropout(),
                                                nn.Linear(config.hidden_size + 200, config.num_labels))
                # use only topics
            elif not self.config.use_embeddings and not self.config.use_features:
                self.classifier = nn.Sequential(nn.ReLU(),
                                                nn.Dropout(),
                                                nn.Linear(200, config.num_labels))
                # use topics and features
            elif not self.config.use_embeddings and self.config.use_features:
                self.classifier = nn.Sequential(nn.ReLU(),
                                                nn.Dropout(),
                                                nn.Linear(280, config.num_labels))
                
                # use features and cls embeddings
        elif self.config.use_features:
            if self.config.use_embeddings:
                self.classifier = nn.Sequential(
                                                nn.Linear(config.hidden_size + 80, config.hidden_size + 80),
                                                nn.ReLU(),
                                                nn.Dropout(),
                                                nn.Linear(config.hidden_size + 80, config.num_labels))
                # use only features
            elif not self.config.use_embeddings:
                self.classifier = nn.Sequential(
                                                nn.Linear(80,80),
                                                nn.ReLU(),
                                                nn.Dropout(),
                                                nn.Linear(80, config.num_labels))


                # use only cls embeddings
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)


        self.init_weights()
        if config.freeze_bert_weights:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        topics=None,
        features=None,
        ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
                
            
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    

        pooled_output = outputs[1]
        
        if len(cls_dict) < len(train_snippets_ids):
            for index in range(len(input_ids)):
                cls_dict.append((int(ids[index]), pooled_output[index]))
        
        print(len(cls_dict))

  
        # TOPICS AND EMBEDDINGS
        if self.config.use_topics and self.config.use_embeddings and not self.config.use_features:
            list_of_tensors = [torch.cat((pooled_output[index], topics[index]), 0) for index in range(len(input_ids))]
            pooled_output = torch.stack(list_of_tensors)
            
        # TOPICS ONLY
        elif self.config.use_topics and not self.config.use_embeddings:
            if not self.config.use_features:
                list_of_tensors = [topics[index] for index in range(len(input_ids))]
                pooled_output = torch.stack(list_of_tensors)
            elif self.config.use_features:
                list_of_tensors = [torch.cat((topics[index], features[index]), 0) for index in range(len(input_ids))]
            pooled_output = torch.stack(list_of_tensors)
            
        elif self.config.use_topics and self.config.use_features and not self.config.use_embeddings:
            list_of_tensors = [torch.cat((topics[index], features[index]), 0) for index in range(len(input_ids))]
            pooled_output = torch.stack(list_of_tensors)
            
        # FEATURES ONLY
        
        elif self.config.use_features and not self.config.use_topics and not self.config.use_embeddings:
            list_of_tensors = [features[index] for index in range(len(input_ids))]
            pooled_output = torch.stack(list_of_tensors)
    
            
            
        elif self.config.use_features:
            list_of_tensors = [torch.cat((pooled_output[index], features[index]), 0) for index in range(len(input_ids))]
            pooled_output = torch.stack(list_of_tensors)

        # EMBEDDINGS ONLY
     
        pooled_output = self.dropout(pooled_output)


         
        logits = self.classifier(pooled_output)  
      
                      
    
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weights)
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


```


```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# this function defines the metrics reported by our trainer in each evaluation pass
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  acc = accuracy_score(labels, preds)
  cm = confusion_matrix(labels, preds)
  return {
      'accuracy': acc,
      'confusion_matrix': str(cm)
  }

# this container mostly serves to turn our data into torch tensors when needed
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, topics, features, ids):
        self.encodings = encodings
        self.labels = labels
        self.topics = topics
        self.features = features
        self.ids = ids
        
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["topics"] = torch.tensor(self.topics[idx])
        item["features"] = torch.tensor(self.features[idx])
        item["ids"] = self.ids[idx]
        return item

    def __len__(self):
        return len(self.labels)
    


```
