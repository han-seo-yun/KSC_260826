# PR: Add Slurm/KSC job script, conda-activation fix, and GPU-OOM fix for HPC clusters

**Base**: `google-research/tabfm` ← **Head**: `han-seo-yun/tabfm:ksc-env-setup`
**상태**: KSC GPU 잡 검증 완료 (job 890313, 890877 V100 OOM → A100 파티션 전환 → job 891702에서 classification/regression/단위 테스트 51개 전부 성공. **이제 실제로 열어도 됨**)

## 요약

tabfm을 KISTI Neuron(KSC, 순수 Slurm 기반 HPC) 클러스터에서 GPU로 재현 가능하게 하는 잡 스크립트를 추가합니다. 진행 중 실제로 두 가지 크래시(conda 활성화 훅 문제, GPU 메모리 부족)를 만났고 둘 다 원인을 규명해 수정했습니다.

## 변경 내용

- `environment.yml` 신규: GPU 없는 로컬 개발용 CPU-only conda 환경 (macOS/Linux). KSC 등 GPU 클러스터에서 실제 학습 전에 `jax`/`jaxlib`를 `jax[cuda12]`로 교체해야 한다는 안내를 파일 상단에 포함
- `ksc_job_tabfm.pbs` 신규: Slurm(`#SBATCH`) 잡 스크립트
  - conda는 `module load conda/...`가 아니라 base Miniconda의 `conda.sh`를 직접 source (아래 "실제 확인한 문제" 참고)
  - `PYTHONNOUSERSITE=1`로 `~/.local`의 시스템 python 패키지 간섭 차단
  - `HF_HOME`을 scratch로 지정해 홈 쿼터 절약 + 재다운로드 방지
  - python 버전(`>=3.11`)·GPU 인식 가드: 잘못된 환경으로 조용히 24시간을 날리는 사고 방지
  - `--comment="field=chem;appl=jax"` (KSC `qsub` 래퍼의 필수 옵션, `showappl`로 확인)
  - `--mail-user`/`--mail-type=END,FAIL`로 작업 종료/실패 이메일 알림

## upstream에 보고할 실제 문제

1. **conda `activate.d` 훅이 `set -u`(nounset)에 안전하지 않음** — tabfm 자체 코드는 아니지만, README가 권장하는 `conda create`/`activate` 워크플로우로 만든 환경에서 재현되는 문제라 잡 스크립트 작성자 입장에서 반드시 알아야 함. 아래 실행 결과 참고.
2. **classification 체크포인트가 16GB급 GPU에는 맞지 않음** — `icl_predictor.tf_icl.blocks.linear1.kernel` 레이어 하나가 `[24, 2048, 8192]` float32(=1.5GiB)이고, 이 레이어 하나를 올리는 것만으로 V100 16GB가 OOM. `XLA_PYTHON_CLIENT_PREALLOCATE=false` + `TF_GPU_ALLOCATOR=cuda_malloc_async`(에러 메시지가 직접 제안)를 적용해도 완전히 동일한 배열에서 완전히 동일하게 실패해, 단편화가 아니라 실제 메모리 부족임을 확인. README에 최소 GPU 메모리 요구사항이 명시돼 있지 않음 — 16GB급 GPU 사용자는 반드시 알아야 함.

## 실제 KSC 실행 결과

(원본 로그: `logs/previous_failures/ksc_job_tabfm.pbs.o890313`/`.e890313`,
`.o890877`/`.e890877` — 최종 성공 로그는 `logs/ksc_job_tabfm.pbs.o891702`/`.e891702`)

**1차 (job 890313, 2026-08-23, 노드 gpu25, V100)**: GPU 인식·가중치 다운로드까지는 성공했으나 체크포인트 복원 중 OOM으로 실패:

```
python 3.11.16 | jax 0.10.2 | devices: [CudaDevice(id=0)]
Tesla V100-PCIE-16GB, 16384 MiB
GPU 확인 완료
Fetching 134 files: 100%|██████████| 134/134
...
W bfc_allocator.cc:514] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.50GiB
...
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.50GiB.
```

1차 수정: `XLA_PYTHON_CLIENT_PREALLOCATE=false` + `TF_GPU_ALLOCATOR=cuda_malloc_async` 추가(에러 메시지가 직접 제안한 조치). 추가로 MolE job 890315에서 확인된 것과 같은 conda activate.d/`set -u` 충돌 가능성을 막기 위해 `conda activate` 호출을 `set +u`/`set -u`로 방어적으로 감쌈.

**2차 (job 890877, 2026-08-23, 노드 gpu25, V100, 위 수정 적용 후)**: 완전히 동일한 배열(`icl_predictor.tf_icl.blocks.linear1.kernel.value`, `[24,2048,8192]` float32 = 정확히 1.5GiB)에서 완전히 동일하게 OOM. 할당자를 바꿔도 안 되므로 단편화가 아니라 **모델이 V100 16GB에 실제로 안 맞는 것**으로 결론. `--partition`을 4-GPU A100 노드(`amd_a100_4`)로 변경.

**3차 (job 891702, 2026-08-24, 노드 gpu45, A100 80GB)**: 완주 성공.
```
NVIDIA A100 80GB PCIe, 81920 MiB
GPU 확인 완료
Classification predictions: [[0.639... 0.360...] [0.393... 0.606...]]
Regression predictions: [19.497... 14.783...]
...
Ran 51 tests in 80.270s
OK
```

## 검증 근거

- 로컬(macOS, Python 3.11.15, jax 0.10.2, CPU)에서 `classification_example.py` HuggingFace 134개 파일 다운로드 + 컴파일 + 예측까지 end-to-end 성공 (2026-08-21)
- KSC 로그인 노드(glogin01)에서 jax GPU 인식(`CudaDevice` 2개) 확인 (2026-08-22)
- KSC GPU 잡: V100에서 두 차례 동일한 OOM 재현(job 890313, 890877) 후 A100로 전환, classification/regression 예측 + 단위 테스트 51개 전부 통과 (job 891702, 2026-08-24)

## 남은 작업

- [x] KSC에서 `qsub ksc_job_tabfm.pbs` 1·2차 제출 → OOM이 단편화가 아닌 실제 메모리 부족임을 확인, 할당자 옵션으로는 해결 안 됨
- [x] A100 파티션(`amd_a100_4`)으로 3차 제출 → classification/regression/단위 테스트 51개 전부 정상 완주 확인
- [x] 이 PR을 실제로 오픈 → https://github.com/google-research/tabfm/pull/99
