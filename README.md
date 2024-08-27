# 2024-NIKL-DCS
국립국어원 AI 말평 일상 대화 요약 '모델뒤에사람있어요' 팀 <br><br>
![image](https://github.com/user-attachments/assets/e8b65f98-bd8a-4bcf-81d9-549e19cee408)

제안된 모델은 [EXAONE](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)이라는 사전 훈련된 언어 모델을 기반으로 파인 튜닝되었으며, 특정 지식이나 기술을 습득하도록 설계되었습니다.
향후 연구에서는 이러한 지식이나 기술이 사람의 선호와 조화될 수 있도록 [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)과 같은 기법을 통해 학습할 계획입니다.<br><br>
현재 업로드된 소스 코드의 경우 인간 선호 데이터셋만 구축 된다면 위 기법을 적용할 수 있도록 작성되었으며, 추후 자동으로 적용되어 학습될 수 있도록 고도화 예정입니다.

# 실행 방법
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
## 학습
```
CUDA_VISIBLE_DEVICES=0,1,2 python -m train model=exaone datasets=[DCS] loss=sft exp_name=exaone_sft batch_size=4 max_prompt_length=2048 max_length=2560
```
학습이 완료되면 `model_ckpt/root/` 디렉토리안에 `exp_name + {학습 시작 시간}`에 해당하는 모델 체크포인트 파일이 저장됩니다.
## 추론
```
python inference/run_test.py --output result.json --model_id model_ckpt/root/exaone_sft_2024-08-26_16-28-10_430889/step-992/ --device cuda:0
```
`model_id` 변수에 모델 체크포인트 경로를 입력한 뒤 실행하면 소스코드 폴더에 `output` 파일이 생성되고, 해당 파일 제출 시 순위표(리더보드)에 성적이 반영됩니다.
