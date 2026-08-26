# PR: Add reproducible CUDA-12 environment, PBS job script, and CPU smoke-test config for HPC clusters

**Base**: `rolayoalarcon/MolE` ← **Head**: `han-seo-yun/MolE:ksc-env-setup`
**상태**: KSC GPU 잡 검증 완료 (job 890315 실패 → 원인 규명·수정 → job 891701에서 2 epoch 정상 완주. **이제 실제로 열어도 됨**)

## 요약

MolE를 CUDA 11.8 세대가 아닌 최신 GPU 클러스터(KSC: CUDA 12.1~13.0.2)에서 재현 가능하게 하고, 몇 분 안에 끝나는 CPU 스모크 테스트 경로를 추가합니다.

## 변경 내용

- `environment.yaml`: `name: moleß` 오타 수정, `pytorch-cuda=11.8` → `12.1` (KSC에 11.x 모듈 없음), `networkx` 명시적 추가 (`dataset/dataset_subgraph.py`가 직접 import하지만 전이 의존성 보장 안 됨)
- `environment_cpu.yaml` 신규: GPU 없는 로컬 개발용 (osx-arm64 포함, rdkit을 conda-forge에서 받도록 함 — 레거시 `rdkit` 채널은 osx-arm64 빌드 미배포)
- `gather_representation.py`: CPU 실행 시 크래시하던 부분 수정
- `config_smoketest.yaml` + `data/pubchem_data/pubchem_200_smoketest.txt` (200개 서브셋) 신규
- `ksc_job_mole.pbs` 신규: PBS 제출 템플릿
  - `num_workers`를 `select`의 `ncpus`에 맞춰 자동 조정 (원본 `config.yaml`의 `num_workers: 100`은 대부분 노드의 코어 수를 초과)
  - `gather_representation.py`의 실제 CLI 인자(`--config_yaml`, `config_representation.yaml`)로 수정 (원래 PBS 초안이 `--config`로 잘못 호출)
  - `conda activate` 호출을 `set +u`/`set -u`로 감싸서 activate.d 훅이 `set -u`에 안전하지 않을 때 잡이 죽는 문제 방지 (아래 "실제 KSC 실행 결과" 참고)

## upstream에 보고할 실제 버그

1. `config.yaml`의 `num_workers: 100`은 대부분의 클러스터 노드 코어 수를 초과해 DataLoader가 스래싱/메모리 폭증 위험
2. README의 `gcn_concat`/`gcn_noconcat` 변형은 `torch-scatter`/`torch-sparse`가 필요한데 `environment.yaml`에 없어서 즉시 실패
3. `ckpt/gin_concat_R1000_E8000_lambda0.0001/checkpoints/`가 빈 디렉토리라 README 예시를 그대로 실행하면 `FileNotFoundError` — Zenodo 다운로드가 선행돼야 한다는 게 README에서 눈에 잘 안 띔

## 검증 근거

- 로컬 CPU(ARM64, Python 3.8, torch 2.2.1) 환경에서 `pretrain.py --config config_smoketest.yaml` 1 epoch 완주 (validation loss 정상 출력) — 2회 재검증(2026-08-13, 2026-08-21) 모두 성공
- 생성된 체크포인트로 `gather_representation.py` 성공 (10개 분자 × 64차원 임베딩 출력)
- KSC 모듈 확인(glogin01, 2026-08-13): `module avail cuda` → 12.1/12.2/12.3/12.4/12.4.1/12.5/12.8/12.9/12.9.1/13.0.2 (11.x 없음)

## 실제 KSC 실행 결과 (job 890315, 2026-08-22, 노드 gpu25)

(원본 로그: `logs/previous_failures/ksc_job_mole.pbs.o890315` / `.e890315`)

1차 제출은 `conda activate` 직후 즉시 실패:

```
/scratch/r978a07/envs/mole/etc/conda/activate.d/libblas_mkl_activate.sh: line 1: MKL_INTERFACE_LAYER: unbound variable
```

**원인**: 이 환경의 MKL(mkl-service) activate 훅이 `set -u`(nounset) 조건에서 안전하게 작성돼 있지 않음. 잡 스크립트 전체에 걸어둔 `set -euo pipefail` 때문에 훅 내부에서 정의 안 된 변수를 참조하는 순간 셀 전체가 죽음. `pretrain.py`는 아예 실행되지 못했음.
**수정**: `conda activate "$ENV_PREFIX"` 호출을 `set +u` / `set -u`로 감싸서 activate.d 훅 실행 구간만 nounset을 해제. conda 자체의 알려진 이슈(activate 스크립트들이 `set -u` 비호환)이므로, 스크립트를 재사용하는 다른 사람도 동일 환경에서 동일하게 겪을 문제.

**재제출 결과 (job 891701, 2026-08-24, 노드 gpu26)**: 수정 반영 후 완주 성공.
```
CUDA available: True Tesla V100-PCIE-16GB
torch CUDA 커널 정상 동작 확인
...
0 89 614.7228515625 (validation)
1 89 506.13312377929685 (validation)   # epoch 2회, validation loss 정상 감소
```

## 남은 작업

- [x] KSC에서 `qsub ksc_job_mole.pbs` 1차 제출 → 실패 원인 규명·수정
- [x] 수정본으로 재제출 (job 891701) → `pretrain.py` 2 epoch 정상 완주 확인
- [x] 이 PR을 실제로 오픈 → https://github.com/rolayoalarcon/MolE/pull/1
