# PR: Replace failing Uni-Core pretraining smoke test with unimol_tools-based validation for HPC clusters

**Base**: `deepmodeling/Uni-Mol` ← **Head**: `han-seo-yun/Uni-Mol:ksc-env-setup`
**상태**: KSC GPU 잡 검증 완료 (job 892530에서 5-fold 학습·검증·예측 출력까지 에러 없이 전체 완주. **이제 실제로 열면 됨**)

## 배경 — 왜 원본 경로를 그대로 쓰지 않았는지

원본 `unimol/` 경로는 Uni-Core(fairseq 기반)로 처음부터 사전학습(pretraining)하는 구조이고, 이건:
- `--enable-cuda-ext`로 Uni-Core를 직접 빌드해야 함 (컴파일 실패 위험이 있는 무거운 단계)
- 사전학습 데이터셋이 필요 (약 114.76GB, 이 프로젝트에는 준비돼 있지 않음)
- 4-GPU 분산학습(`torch.distributed.launch --nproc_per_node=4`)까지 필요

즉, 원본 경로는 "실제 사전학습 데이터가 있고 처음부터 학습하는" 시나리오에만 맞고, 지금 프로젝트가 실제로 필요한 건 기존 사전학습 가중치를 가져와 다운스트림 태스크에 맞게 쓰는 것(fine-tuning)에 더 가깝다. deepmodeling이 별도로 배포하는 `unimol_tools`(PyPI 설치 가능, Uni-Core 의존성 없음, HuggingFace에서 사전학습 가중치 자동 다운로드)가 이 시나리오에 훨씬 적합해서, KSC 첫 GPU 파이프라인 검증은 이 경로로 먼저 진행했다.

**2026-08-25 확정**: 담당 교수님께 확인한 결과, 이 프로젝트가 필요한 방향은 처음부터의 사전학습(114.76GB 데이터 + Uni-Core)이 아니라 **"이미 학습된 Uni-Mol 모델을 가져와서 데이터에 맞게 활용"** — 즉 `unimol_tools` 기반 fine-tuning 경로로 확정. 원본 Uni-Core 사전학습 경로는 더 이상 필요하지 않아 잡 스크립트에서 제거했다.

## 변경 내용

- `environment.yml` 신규: GPU 없는 로컬 개발용 CPU-only conda 환경 (macOS arm64 검증). `unimol_tools`(권장 진입점, v0.1.0부터 Uni-Core 의존성 제거됨)와 원본 `unimol/`(Uni-Core 기반, CUDA 확장 커널은 기본적으로 꺼져 있어 CPU 스모크 테스트 가능) 두 서브프로젝트 모두 커버
- `install.sh` 신규: `./install.sh local`(CPU 개발용)/`./install.sh ksc`(GPU 클러스터용) 두 모드 설치 스크립트
- `ksc_job_unimol.pbs` 신규: Slurm(`#SBATCH`) 잡 스크립트. 페이로드는 `unimol_smoketest.py`(unimol_tools 5-fold 분류 스모크 테스트) — 교수님 확인 후 fine-tuning 방향으로 확정돼 원본 Uni-Core `unicore-train` 경로는 제거
  - conda는 `module load conda/...`가 아니라 base Miniconda의 `conda.sh`를 직접 source
  - `conda activate` 호출을 `set +u`/`set -u`로 방어적으로 감쌈 (MolE job 890315에서 실제 확인된 activate.d/`set -u` 충돌 문제, 아래 참고)
  - `HF_HOME`을 scratch로 지정
  - `--comment="field=chem;appl=pytorch"`, `--mail-user`/`--mail-type=END,FAIL`
- `unimol_smoketest.py` 신규: 20개 toy SMILES로 `MolTrain(task="classification", ...)` fit까지 end-to-end 확인하는 최소 스크립트

## upstream에 보고할 실제 문제

1. **conda `activate.d` 훅이 `set -u`(nounset)에 안전하지 않음** — MolE 잡(890315)에서 `MKL_INTERFACE_LAYER: unbound variable`로 실제 재현. Uni-Mol 잡에서도 동일 패턴의 환경 구성이면 같은 문제가 날 수 있어 선제적으로 수정.
2. `unimol_tools` README의 quick-start 예시는 `MolTrain.fit(data=<pandas DataFrame>)`가 되는 것처럼 보이지만, `datareader.py`의 `read_data()`는 `str`(csv/sdf 경로)·`dict`·`list`/`ndarray`만 받고 DataFrame은 `ValueError: Unknown data type`으로 거부함 — CSV로 저장 후 경로를 넘겨야 함 (`unimol_smoketest.py`에서 실제로 이렇게 우회).
3. **`MolTrain.fit()`이 예측값을 반환하지 않음** — `train.py`의 `fit()`은 `self.cv_pred = y_pred`로 내부에 저장만 하고 `return`(값 없이)으로 끝남. README/코드 어디에도 이게 명시돼 있지 않아 `pred = clf.fit(data=...)`처럼 반환값을 쓰면 `pred`가 항상 `None`이 됨 — 예측값은 `clf.cv_pred`로 접근해야 함 (2026-08-24 job 891712에서 `TypeError: 'NoneType' object is not subscriptable`로 실제 확인, `unimol_smoketest.py`에서 수정 반영).
4. **V100(compute capability 7.0)에서 `pip install torch`(버전 미지정)가 커널 없는 빌드를 설치** — 최신 cu130 빌드는 CC 7.0용 커널이 빠져 있어 `torch.cuda.is_available()==True`인데도 실제 연산에서 `CUDA error: no kernel image is available for execution on the device`로 죽음 (job 890322). cu121 빌드로 고정 설치해야 함.

## 검증 근거

- 로컬 macOS(CPU, Python 3.x)에서 `unimol_smoketest.py` 실행 — 5-fold 중 4-fold 실제 학습 완료 + AUC 계산 성공 (2026-08-22)
- macOS에서는 `unimol_tools.MolTrain`이 내부적으로 `multiprocessing.Pool`을 쓰는데, macOS의 spawn 방식 때문에 `if __name__ == "__main__":` 가드가 필요함을 확인 — Linux/KSC는 fork 방식이라 이 문제가 재현되지 않음(가드는 이미 스크립트에 있어 안전)

## 실제 KSC 실행 결과

(원본 로그: `logs/previous_failures/ksc_job_unimol.pbs.o890322`/`.e890322`,
`.o891712`/`.e891712` — 최종 성공 로그는 `logs/ksc_job_unimol.pbs.o892530`/`.e892530`)

**1차 (job 890322, 2026-08-23, 노드 gpu25, V100)**: GPU 인식·가중치 다운로드·데이터 준비까지 성공했으나 첫 배치 학습에서 크래시:
```
UserWarning: ... torch==2.13.0+cu130 does not include kernels for this GPU (CC 7.0)
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```
수정: `pip install torch --index-url https://download.pytorch.org/whl/cu121`로 V100 호환 빌드 고정, 학습 진입 전 실제 CUDA 연산 가드 추가.

**2차 (job 891712, 2026-08-24, 노드 gpu26, V100, torch cu121 재설치 후)**: torch 커널 가드 통과, 5-fold 전부 실제 학습 완료 + 지표 산출·저장까지 성공:
```
Uni-Mol metrics score: {'auc': 0.2499..., 'acc': 0.35, ...}
Uni-Mol & Metric result saved!
```
다만 스크립트 마지막 `print(pred[:5])`에서 `TypeError: 'NoneType' object is not subscriptable` — 위 upstream 문제 3번(`fit()`이 `None` 반환) 때문. `clf.cv_pred[:5]`로 수정.

**3차 (job 892530, 2026-08-25, 노드 gpu26, V100, `fit()` 반환값 수정 후)**: 처음부터 끝까지 에러 없이 완주:
```
torch CUDA 커널 정상 동작 확인
=== SMOKETEST OK ===
[[0.0293...]
 [0.0591...]
 [0.6241...]
 [0.0002...]
 [0.2498...]]
```

## 남은 작업

- [x] KSC GPU 잡 제출 → torch 커널 문제 발견·수정 (job 890322 → cu121 재설치)
- [x] 재제출 → 5-fold 학습·검증 완주 확인 (job 891712)
- [x] `unimol_smoketest.py`의 `fit()` 반환값 버그 수정
- [x] 수정본으로 3차 제출 → 처음부터 끝까지 에러 없이 완주 확인 (job 892530)
- [x] 교수님께 확인: fine-tuning(사전학습 가중치 활용) 경로가 맞음 → 원본 Uni-Core 사전학습 경로는 잡 스크립트에서 제거
- [x] 이 PR을 실제로 오픈 → https://github.com/deepmodeling/Uni-Mol/pull/374
