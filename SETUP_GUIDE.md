# KSC(KISTI Neuron)에서 작업 제출하는 법

## 0. 사전 준비

- KSC 계정 발급 및 로그인 (`ssh <계정>@glogin01` 등)
- 파일 업로드: KSC의 `/scratch/$USER/ksc-models/` 아래에 이 저장소 구조를 그대로 올린다.
  즉 `MolE/`, `tabfm/`, `Uni-Mol/`, `common/` 네 폴더가 모두 `/scratch/$USER/ksc-models/` 바로
  아래(형제 폴더)에 오도록 — 예: `/scratch/$USER/ksc-models/MolE/`,
  `/scratch/$USER/ksc-models/common/ksc_smoketest.pbs`.
  이미 각 모델을 git fork로 관리 중이라면 로그인 노드에서 `git pull`로 받으면 된다.
  (터미널 접속이 안 되면 FileZilla 같은 SFTP 클라이언트로 드래그 앤 드롭도 가능)

## 1. 사전 점검 (선택, 최초 1회 추천)

큐/GPU/conda/네트워크가 정상인지 가볍게 확인하는 스크립트. `common/ksc_smoketest.pbs`가
`/scratch/$USER/ksc-models/common/`에 올라와 있어야 한다:

```bash
cd /scratch/$USER/ksc-models/common
qsub ksc_smoketest.pbs
qstat -u $USER
```

## 2. conda 환경 만들기 (모델별, 최초 1회)

conda 명령은 `module load conda/...`로는 안 나온다 — base Miniconda를 직접 source해야 한다.

```bash
source /apps/applications/Miniconda/25.11.1/etc/profile.d/conda.sh
```

**MolE**
```bash
cd /scratch/$USER/ksc-models/MolE
conda env create -f environment.yaml -p /scratch/$USER/envs/mole
```

**tabfm**
```bash
conda create -p /scratch/$USER/envs/tabfm python=3.11 -y
conda activate /scratch/$USER/envs/tabfm
cd /scratch/$USER/ksc-models/tabfm
pip install -U pip && pip install -e ".[jax,cuda]"
conda deactivate
```

**Uni-Mol** (V100은 torch 빌드를 cu121로 고정해야 함 — `Uni-Mol/ISSUES.md` 참고)
```bash
conda create -p /scratch/$USER/envs/unimol_tools python=3.10 -y
conda activate /scratch/$USER/envs/unimol_tools
pip install unimol_tools huggingface_hub
pip install torch --index-url https://download.pytorch.org/whl/cu121
conda deactivate
```

## 3. 작업 제출

```bash
cd /scratch/$USER/ksc-models/<모델이름>
qsub ksc_job_<모델이름>.pbs
```

세 개를 순서대로 한 번에 제출하려면 `common/submit_ksc_all.sh` 사용 (기본값은
`/scratch/$USER/ksc-models/{tabfm,MolE,Uni-Mol}` — 다른 경로에 뒀다면 스크립트 상단의
`BASE` 변수만 수정).

## 4. 상태 확인 / 결과 확인

```bash
qstat -u $USER          # Q=대기, R=실행중
```

완료되면 각 잡 스크립트에 지정된 `--mail-user`로 종료(END)/실패(FAIL) 이메일이 온다.
로그 파일은 `<스크립트이름>.o<jobid>`(표준출력), `<스크립트이름>.e<jobid>`(표준에러)로 생성된다
(제출 시 KSC의 `qsub` 래퍼가 이 이름으로 강제 지정 — 잡 스크립트 안의 `--output`/`--error`
지시어와 이름이 다르게 나올 수 있음).

## 자주 겪는 함정

- **`#PBS` 지시어는 무시된다** — 이 시스템은 순수 Slurm이라 `#SBATCH`만 인식한다.
- **`--comment="field=<분야>;appl=<프로그램>"`가 없으면 제출이 거부된다** — 유효한 값은
  `showappl` 명령으로 확인.
- **conda activate 직후 죽으면** — activate.d 훅이 `set -u`와 충돌하는 경우가 있다
  (`conda activate` 호출을 `set +u`/`set -u`로 감싸면 해결, 각 잡 스크립트에 이미 반영돼 있음).
- **git pull이 `libldap`/`OpenSSL` 에러로 실패하면** — conda/module 환경이 활성화된 상태라
  `LD_LIBRARY_PATH`가 오염된 것. `conda deactivate` 하고 다시 시도.
