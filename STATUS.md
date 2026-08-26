# 현황 요약 (2026-08-26 기준)

## 목적

MolE, tabfm, Uni-Mol 세 모델을 KISTI Neuron 슈퍼컴퓨터(KSC)에서 GPU로 재현 가능하게 만들고,
실제 작업 제출까지 확인하는 것. 테스트 완료 후 작업 환경 세팅 완료.

## 최종 결과

| 모델 | 최종 job ID | 노드/GPU | 결과 |
|---|---|---|---|
| MolE | 891701 | gpu26 (V100) | ✅ pretrain 2 epoch 정상 완주, validation loss 정상 감소 |
| tabfm | 891702 | gpu45 (**A100 80GB**) | ✅ classification/regression 예측 정상 + 단위 테스트 51개 전부 통과 |
| Uni-Mol | 892530 | gpu26 (V100) | ✅ 5-fold 학습·검증·예측 출력까지 에러 없이 완주 |

세 모델 다 `logs/` 폴더에 실제 KSC 실행 로그(`.o<jobid>` = stdout, `.e<jobid>` = stderr)를 넣어뒀다.
`logs/previous_failures/`에는 아래 7건의 문제를 실제로 겪었을 때의 원본 실패 로그도 그대로
넣어뒀다 (수정 전/후를 로그로 직접 대조 가능).

## 진행하면서 실제로 겪은 문제 (총 7건)

각 모델 폴더의 `ISSUES.md`에 원인·재현·수정 내용을 자세히 적어뒀고, 여기는 요약만.

1. **이 시스템은 순수 Slurm** — `qsub`는 `sbatch`를 그대로 호출하고 `#PBS` 지시어는 해석하지 않음(그냥 주석 처리됨). 세 잡 스크립트 모두 `#SBATCH` 문법으로 다시 작성.
2. **`--comment="field=...;appl=..."` 필수** — `showappl`로 확인한 값(`field=chem`, `appl=pytorch`/`jax`)을 세 스크립트 모두에 추가.
3. **`module load conda/...`는 `conda` 명령 자체를 안 줌** — 미리 만들어진 특정 환경의 `bin/`만 PATH에 얹어주는 것뿐이라, 실제 base Miniconda(`/apps/applications/Miniconda/25.11.1`)의 `conda.sh`를 직접 source하도록 수정.
4. **conda `activate.d` 훅이 `set -u`에 안전하지 않음** — MolE에서 `MKL_INTERFACE_LAYER: unbound variable`로 잡이 즉시 죽는 걸 실제로 확인. `conda activate` 호출을 `set +u`/`set -u`로 감싸서 세 스크립트 모두 방어.
5. **tabfm이 V100 16GB에서 OOM** — 체크포인트의 한 레이어가 정확히 1.5GB라 할당자를 바꿔도(`cuda_malloc_async`) 안 됨 → **A100 파티션으로 전환**해서 해결.
6. **Uni-Mol에서 `pip install torch`(버전 미지정)가 V100(CC 7.0) 커널이 없는 최신 cu130 빌드를 설치** — `torch.cuda.is_available()==True`인데도 실제 연산은 안 되는 함정. cu121 빌드로 고정 설치하도록 수정, 학습 진입 전 실제 연산 가드도 추가.
7. **`unimol_tools`의 `MolTrain.fit()`이 예측값을 반환하지 않음**(`None` 반환) — 예측값은 `clf.cv_pred`로 접근해야 함. 스크립트 수정.

## 사전학습 방향 확정

Uni-Mol은 처음부터의 사전학습(Uni-Core, 114.76GB 데이터 필요)이 아니라, **이미 학습된 모델을
가져와서 데이터에 맞게 활용(fine-tuning)**하는 방향이 맞는 것으로 확정. 그래서 Uni-Core 없이
HuggingFace에서 사전학습 가중치를 자동으로 받는 `unimol_tools` 경로로 진행함. 원본 Uni-Core 사전학습 커맨드는 잡 스크립트에서 제거함.