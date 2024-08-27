# 2024-NIKL-DCS
- 국립국어원 AI 말평 일상 대화 요약 '모델뒤에사람있어요' 팀 <br><br>
![image](https://github.com/user-attachments/assets/e8b65f98-bd8a-4bcf-81d9-549e19cee408)



# 실행 방법
## 학습
학습이 완료되면 `model_ckpt/root/` 디렉토리안에 `exp_name + {학습 시작 시간}`에 해당하는 모델 체크포인트 파일이 저장됩니다.
```
CUDA_VISIBLE_DEVICES=0,1,2 python -m train model=exaone datasets=[DCS] loss=sft exp_name=exaone_sft batch_size=4 max_prompt_length=2048 max_length=2560
```
## 추론
`model_id` 변수에 모델 체크포인트 경로를 입력한 뒤 실행하면 소스코드 폴더에 `output` 파일이 생성되고, 해당 파일 제출 시 순위표(리더보드)에 성적이 반영됩니다.
```
python run_test.py --output result.json --model_id model_ckpt/root/exaone_sft_2024-08-26_16-28-10_430889/step-992/ --device cuda:0
```
