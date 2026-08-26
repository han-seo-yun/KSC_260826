# KSC(KISTI Neuron) 환경 구축 전달 패키지 — 2026-08-26

MolE, tabfm, Uni-Mol 세 모델을 KISTI Neuron 슈퍼컴퓨터(KSC, Slurm 기반)에서 GPU로 돌릴 수 있도록
환경을 구축하고, 실제로 제출해서 끝까지 완주하는 것까지 확인한 결과물입니다.
`phdgil/fda-endocrine-disruption-ksc-bundles`에 올릴 예정이었으나 현재 접근 권한이 없어
우선 이 폴더로 정리해서 전달합니다.

## 폴더 구조

```
KSC_전달_2026-08-26/
├── STATUS.md              <- 먼저 읽을 문서: 전체 진행 상황·결과 요약
├── SETUP_GUIDE.md          <- 슈퍼컴퓨터에 작업 제출하는 법 (qsub 절차)
├── MolE/
│   ├── environment.yaml       (GPU용, KSC에서 이 파일로 conda 환경 생성)
│   ├── environment_cpu.yaml   (GPU 없는 로컬 개발/테스트용)
│   ├── config_smoketest.yaml  (빠른 동작 확인용 소규모 설정)
│   ├── ksc_job_mole.pbs       (KSC 제출용 잡 스크립트)
│   ├── ISSUES.md              (겪은 오류·원인·수정 내용 전체 기록)
│   └── logs/                 (실제 KSC 실행 로그: 최종 성공본 + previous_failures/에 이전 실패본)
├── tabfm/          (구조 동일)
├── Uni-Mol/        (구조 동일, install.sh 추가)
└── common/
    ├── ksc_smoketest.pbs      (큐/GPU/conda/네트워크 등 사전 점검용)
    └── submit_ksc_all.sh      (세 모델 잡을 순서대로 한 번에 제출하는 스크립트)
```

## 요약

- 세 모델 모두 **KSC GPU 잡 제출 → 실제 완주까지 확인 완료**
- 진행하면서 실제로 겪은 오류 7건을 각 모델 `이슈_및_수정사항.md`에 원인·재현·수정 내용으로 기록해둠
- 각 `ksc_job_*.pbs`는 그대로 갖다 써도 되도록 환경변수(`ENV_PREFIX`, `CONDA_BASE` 등)만 바꾸면 재사용 가능하게 작성됨
