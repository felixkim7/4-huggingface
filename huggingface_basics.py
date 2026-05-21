#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers datasets huggingface_hub')


# # 1. 허깅페이스 개요

# ### BERT와 GPT-2 모델을 활용할 때 허깅페이스 트랜스포머 코드 비교

# In[ ]:


from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel

text = "What is Huggingface Transformers?"
# BERT 모델 활용
bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded_input = bert_tokenizer(text, return_tensors='pt')
bert_output = bert_model(**encoded_input)
# GPT-2 모델 활용
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2')
encoded_input = gpt_tokenizer(text, return_tensors='pt')
gpt_output = gpt_model(**encoded_input)


# In[ ]:


bert_output #출력값: 입력 텍스트의 각 토큰과 전체 시퀀스를 나타내는 고차원의 벡터(은닉 상태)
gpt_output #출력값: 로짓(logits) 벡터. 각 단어(토큰)가 다음에 나올 확률을 나타내는 값
#BERT: 분류 모델로, 텍스트화 불가, GPT: 다음 단어 예측하는 모델로, 텍스트 출력하려면 별도 작업 필요


# # 2. 트랜스포머 모델 활용하기

# ### 모델 불러오기

# ### 모델 아이디로 바디만 불러오기

# In[ ]:


from transformers import AutoModel
base_model_id = 'bert-base-uncased'
base_model =


# In[ ]:


#model metadata
base_model_config = base_model.config
config_dict = base_model_config.__dict__
config_dict


# ### 분류 헤드가 포함된 모델 불러오기

# In[ ]:


from transformers import AutoModelForSequenceClassification
classification_model_id = 'SamLowe/roberta-base-go_emotions'
classification_model =


# In[ ]:


#model metadata
classification_model_config = classification_model.config
config_dict = classification_model_config.__dict__
config_dict
#id2label확인


# ### 분류 헤드가 랜덤으로 초기화된 모델 불러오기

# In[ ]:


from transformers import AutoModelForSequenceClassification
random_model_id = 'klue/roberta-base'
random_model =


# In[ ]:


#model metadata
random_model_config = random_model.config
config_dict = random_model_config.__dict__
config_dict


# ## 토크나이저 활용하기

# ### 토크나이저 사용하기

# In[ ]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded = tokenizer("I love banana")
print(encoded)
# {
#   'input_ids':      [101, 1045, 2293, 15212, 102],
#   'token_type_ids': [0, 0, 0, 0, 0],
#   'attention_mask': [1, 1, 1, 1, 1]
# }


# In[ ]:


# 토큰화 결과 확인
print(tokenizer.convert_ids_to_tokens([101, 1045, 2293, 15212, 102]))
# ['[CLS]', 'i', 'love', 'banana', '[SEP]']

# 다시 텍스트로
print(tokenizer.decode([101, 1045, 2293, 15212, 102], skip_special_tokens=True))
# 'i love banana'


# # 3. 데이터셋 활용하기

# ## 데이터셋 다운로드

# ### 허깅페이스 허브에서 데이터셋 다운로드

# In[ ]:


get_ipython().system('pip install datasets')

from datasets import load_dataset
klue_mrc_dataset =
klue_mrc_dataset_only_train =


# In[ ]:


klue_mrc_dataset


# In[ ]:


klue_mrc_dataset_only_train


# In[ ]:


klue_mrc_dataset['train'][0]


# ### 로컬의 데이터 활용하기

# In[ ]:


# 로컬의 데이터 파일을 활용
dataset_json = load_dataset("json", data_files="/content/drive/MyDrive/git/3-huggingface/review_신라스테이 해운대.json")
#print(dataset_json)

#로컬 데이터로는 train 100%이 default.
dataset_json_train = load_dataset("json", data_files="/content/drive/MyDrive/git/3-huggingface/review_신라스테이 해운대.json", split='train')
#print(dataset_json_train)

#수동으로 split 하는 방법
dataset_json_test = dataset_json["train"].train_test_split(test_size=0.2, seed=42)["test"]
#print(dataset_json_test)


# 
# ### 데이터셋 제작

# In[ ]:


# 파이썬 딕셔너리 활용
from datasets import Dataset
my_dict = {"a": [1, 2, 3]}
dataset = Dataset.from_dict(my_dict)
print(dataset[0])

# 판다스 데이터프레임 활용
from datasets import Dataset
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3]})
dataset = Dataset.from_pandas(df)
print(dataset[0])


# ## 데이터셋 가공하기

# ### 실습에 사용하지 않는 불필요한 컬럼 제거

# In[ ]:


from datasets import load_dataset
klue_tc_train =
klue_tc_eval =
print(klue_tc_train)
print(klue_tc_eval)


# In[ ]:


klue_tc_train_removed =
klue_tc_eval_removed =
print(klue_tc_train_removed)
print(klue_tc_train_removed[0])


# ### 카테고리를 문자로 표기한 label_str 컬럼 추가

# In[ ]:


print(klue_tc_train_removed.features['title'])
print(klue_tc_train_removed.features['label'])


# In[ ]:


klue_tc_label = klue_tc_train_removed.features['label']

def make_str_label(batch):

  return batch

klue_tc_train_removed =
klue_tc_eval_removed =

klue_tc_train_removed[10]


# ### 학습/검증/테스트 데이터셋 분할

# In[ ]:


train_dataset = klue_tc_train_removed.train_test_split(test_size=10000, shuffle=True, seed=42)['test']
dataset = klue_tc_eval_removed.train_test_split(test_size=1000, shuffle=True, seed=42)
test_dataset = dataset['test']
valid_dataset = dataset['train'].train_test_split(test_size=1000, shuffle=True, seed=42)['test']


# # 4. 모델을 이용하여 추론하기

# ### 학습한 모델을 불러와 pipeline을 활용해 텍스트 분류하기

# In[ ]:


from transformers import pipeline

model_id = "hykiim/roberta-base-klue-ynat-classification"

model_pipeline = pipeline("text-classification", model=model_id)

model_pipeline()


# ### 커스텀 파이프라인 구현

# In[ ]:


import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CustomPipeline:
    def __init__(self, model_id):
        self.model = #목적에 맞는 모델 헤드 불러옴
        self.tokenizer = #모델과 동일한 토크나이저 불러옴
        self.model.eval() #모델 평가 모드

    def __call__(self, texts):
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)# 입력 텍스트를 토크나이저를 사용하여 토큰화
        # return_tensors="pt": PyTorch 텐서로 반환, padding=True: 패딩 추가, truncation=True: 잘라내기

        with torch.no_grad():#기울기 계산 비활성화(옵티마이저 계산 생략)-> 추론 과정에서는 옵티마이징 할 필요 없음.
            outputs = self.model(**tokenized)#**연산자는 딕셔너리를 키워드 인자로 unpacking하여 모델에 입력. 즉 토큰 아이디
            logits = outputs.logits# 모델 출력에서 logits 값을 추출. (분류되지 않은 예측값)

        #추론이 정확할 확률 구하기
        probabilities = softmax(logits, dim=-1)
        scores, labels = torch.max(probabilities, dim=-1)
        labels_str = [self.model.config.id2label[label_idx] for label_idx in labels.tolist()]

        return [{"label": label, "score": score.item()} for label, score in zip(labels_str, scores)]

custom_pipeline = CustomPipeline(model_id)
print(custom_pipeline(test_dataset['title'][:5]))
print(test_dataset['label_str'][:5])


# In[ ]:


import torch

gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2')
encoded_input = gpt_tokenizer(text, return_tensors='pt')
gpt_output = gpt_model(**encoded_input)


# GPT-2는 기본 pad 토큰이 없으므로, eos 토큰을 pad 토큰으로 설정 (일반적인 관행)
if gpt_tokenizer.pad_token is None:
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

# 2. 입력 텍스트 준비
text = "What is Huggingface Transformer?"
encoded_input = gpt_tokenizer(text, return_tensors='pt')
input_ids = encoded_input['input_ids']

# 3. 텍스트 생성 (model.generate 사용)
# max_length: 생성될 텍스트의 최대 길이 (입력 포함)
# num_return_sequences: 생성할 시퀀스 수
# pad_token_id: 패딩에 사용될 토큰 ID 설정
output_sequences = gpt_model.generate(
    input_ids=input_ids,
    max_length=50,  # 예시: 최대 50 토큰까지 생성
    num_return_sequences=1,
    pad_token_id=gpt_tokenizer.pad_token_id,
    # 더 다양한 결과를 원하면 다음 파라미터 추가 가능:
    do_sample=True,      # 샘플링 사용 여부
    top_k=50,            # 상위 K개 토큰 중에서만 샘플링
    top_p=0.95,          # 누적 확률 P 이상인 토큰 중에서만 샘플링 (nucleus sampling)
    temperature=0.7,     # 확률 분포를 조절 (낮을수록 결정적, 높을수록 무작위적)
)

# 4. 생성된 토큰 ID 시퀀스를 텍스트로 디코딩
# output_sequences[0]은 생성된 첫 번째 시퀀스를 의미
# skip_special_tokens=True 옵션은 <|endoftext|> 같은 특수 토큰을 결과에서 제외
generated_text = gpt_tokenizer.decode(output_sequences[0], skip_special_tokens=True)

# 5. 결과 출력
print("Input Text:", text)
print("Generated Text:", generated_text)

