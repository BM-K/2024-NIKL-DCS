# 2024-NIKL-DCS

국립국어원 AI 말평 일상 대화 요약 '모델뒤에사람있어요' 팀 <br><br>
![image](https://github.com/user-attachments/assets/e8b65f98-bd8a-4bcf-81d9-549e19cee408)

제안된 모델은 사전 훈련된 언어 모델 [EXAONE](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)을 기반으로 파인 튜닝되었으며, 특정 지식이나 기술을 습득하도록 설계되었습니다.
향후 연구에서는 이러한 지식이나 기술이 사람의 선호와 조화될 수 있도록 [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)과 같은 기법을 통해 학습할 계획입니다.<br><br>
현재 업로드된 소스 코드의 경우 인간 선호 데이터셋만 구축 된다면 위 기법을 적용할 수 있도록 작성되었으며, 추후 자동으로 적용되어 학습될 수 있도록 고도화 예정입니다.

# Quick Start
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "BM-K/EXAONE-3.0-7.8B-Daily-Conversation-Summary"
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

PROMPT = '''You are EXAONE model from LG AI Research, a helpful assistant. Please answer the user's questions kindly.'''
chat = "[Conversation]\n화자SD2000001: 저는 여행 다니는 것을 굉장히 좋아하는데요. 그래가지고 스페인이나 뭐 영국 유럽 아니면 국내에서도 뭐 강릉이나 전주 같은 데를 많이 다녔는데\n화자SD2000001: 혹시 여행 다니는 거 좋아하시나요?\n화자SD2000002: 저 여행 다니는 거 되게 좋아해서 대학교 내내 여행을 엄청 많이 다녔었는데요.\n화자SD2000002: 제가 고등학교 때는 여행에 대해 흥미가 없었는데 그게 좀 아버지가 짠대로 패키지처럼 여행을 다녀서 그런 것 같아요.\n화자SD2000002: 그래서 대학교 간 이후로는 해외여행을 되게 많이 갔었는데 그중에서 제일 기 좋았던 거는 스페인이랑 포르투갈이었거든요.\n화자SD2000002: 어~ 혹시 포르투갈이나 스페인 유럽 쪽 다녀오신 적 있으신가요?\n화자SD2000001: 어~ 네. 저도 우연히 스페인과 포르투갈을 다녀왔었었습니다.\n화자SD2000001: 어~ 저는 스페인 중에서도 마드리드에 근교에 있었던 톨레도라는 지역이 굉장히 좋았는데요. 그 톨레도에서 특히 기억에 남았던 거는 거기에 대성당이 있는데 그 성당이 엄청 화려하더라고요. 그래서 거기를 꾸며논 거를 보면은 금을 엄청 많이 사용해가지고 되게 빤짝빤짝하고 좀 성당은 보통 좀 소박하다라는 인식이 있었는데 아~ 이렇게 화려한 성당도 있구나라는 거를 새롭게 알게 됐었습니다.\n화자SD2000001: 어~ 또 톨레도에 지역 음식도 같이 먹었었는데 아~ 이름은 지금 잘 생각이 나지는 않지만 굉장히 달달했던 그런 디저트 종류였는데 그~ 디저트도 먹고 그다음에 천천히 걸어 다니면서 주변 풍경도 보고 근교 여행만의 약간 소박한 맛이 있었다고 생각을 합니다.\n화자SD2000001: 어~ 또 물론 마드리드도 굉장히 좋았는데 유럽 여행을 많이 가셨다고 해서 혹시 톨레도도 가본 적이 있나요?\n화자SD2000002: 아~ 제가 톨레도도 다녀왔는데 저는 이제 여행 일정을 길게 잡아서 톨레도는 하루를 봤는데 도 그렇게 너무 더웠기 때문에 많이 보진 못한 것 같아요.\n화자SD2000002: 그때는 버스 관광버스를 타고 계속 돌아다니면서 이제 내리는 데마다 관광을 할 수 있는 버스를 탔는데요. 그 버스를 타고 전체를 다 내려서 보려고 했지만 날씨가 너무 더워서 금방 금방 이제 xx 장소로 넘어갔던 것 같 같습니다.\n화자SD2000002: 거기는 이제 고대 도시라고 해서 사람들이 많이 추천한 거에 비해서는 저는 하루를 잡기에는 조금 부족한 여행지라는 생각이 들었고\n화자SD2000002: 오히려 광장에서 쇼핑을 했던 게 더 기억에 남습니다.\n\n[Question]\n위 해외여행 주제에 대한 대화를 요약해주세요."

message = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": chat}
]

source = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        return_tensors="pt"
)

outputs = model.generate(
        source.to(device),
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False
)

summary = tokenizer.decode(outputs[0][source.shape[-1]:], skip_special_tokens=True).replace('\n',' ').replace('  ', ' ')
```

# 학습 및 평가 방법

## 준비

```
# 필요 라이브러리를 설치합니다.
pip install -r requirements.txt
```

```
# 학습 및 평가 데이터셋을 위치시킵니다.
dcs_2024_data
├── 일상대화요약_train.json
├── 일상대화요약_dev.json
└── 일상대화요약_train.json
```

huggingface에서 [EXAONE](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct) access token을 [발급](https://huggingface.co/docs/hub/security-tokens)받습니다.

## 학습

```
CUDA_VISIBLE_DEVICES=0,1,2 python -m train model=exaone datasets=[DCS] loss=sft exp_name=exaone_sft batch_size=4 max_prompt_length=2048 max_length=2560 token='{access_token}'
```

학습이 완료되면 `model_ckpt/root/` 디렉토리안에 `exp_name + {학습 시작 시간}`에 해당하는 모델 체크포인트 파일이 저장됩니다.

## 추론

```
python inference/run_test.py --output result.json --model_id model_ckpt/root/exaone_sft_2024-08-26_16-28-10_430889/step-992/ --device cuda:0
```

`model_id` 변수에 모델 체크포인트 경로를 입력한 뒤 실행하면 소스코드 폴더에 `output` 파일이 생성되고, 해당 파일 제출 시 순위표(리더보드)에 성적이 반영됩니다.
