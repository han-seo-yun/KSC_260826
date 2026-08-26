#!/bin/bash
# =============================================================================
# tabfm / MolE / Uni-Mol 세 잡을 KSC 로그인 노드에서 순서대로 제출하는 래퍼.
#
# 하나의 잡으로 합치지 않은 이유:
#   - 자원 요구가 다르다 (tabfm/MolE: 1 GPU, Uni-Mol 원본 사전학습 경로: 4 GPU).
#   - 하나가 실패해도 나머지는 독립적으로 받아야 한다.
#   -> 그래서 "각각 qsub, 한 스크립트로 순서대로 실행 + 상태를 한 번에 요약"
#      이 맞는 형태다. qsub 자체는 비동기라 세 개를 순서대로 던져도
#      실제 실행은 스케줄러가 자원에 따라 병렬로 돌려준다.
#
# 이 시스템(KISTI Neuron)은 순수 Slurm이다. `qsub`는 sbatch를 그대로 호출만
# 하므로 각 잡 스크립트 안의 지시어는 반드시 `#SBATCH`로 돼있어야 한다
# (showappl 확인, 2026-08-21: --comment="field=<field>;appl=<program>" 필수).
#
# 사전 조건: 아래 디렉토리 구조를 가정한다
#   (tabfm의 ksc_job_tabfm.pbs가 이미 /scratch/$USER/ksc-models/tabfm 를
#    기준으로 짜여 있어서, 나머지 둘도 같은 부모 디렉토리에 둔다고 가정):
#     /scratch/$USER/ksc-models/{MolE,tabfm,Uni-Mol}
#   각 디렉토리는 해당 repo를 `ksc-env-setup` 브랜치로 clone/pull 해둔 상태여야 하고,
#   conda 환경(mole / tabfm 전용 prefix / unimol_tools 전용 prefix)도 미리
#   만들어져 있어야 한다 (각 ksc_job_*.pbs 상단 주석 참고).
#
# 사용법:
#   ./submit_ksc_all.sh              # 셋 다 제출 (순서: tabfm -> mole -> unimol)
#   ./submit_ksc_all.sh tabfm mole   # 일부만 제출
#
# 권장 순서 이유: tabfm이 제약이 가장 적어 첫 성공을 가장 빨리 얻을 수 있고,
# Uni-Mol은 (unimol_tools 경로라도) 새 conda env라 가장 마지막에 시도한다.
# =============================================================================
set -uo pipefail

BASE="/scratch/$USER/ksc-models"

declare -A JOB_SCRIPT=(
  [tabfm]="$BASE/tabfm/ksc_job_tabfm.pbs"
  [mole]="$BASE/MolE/ksc_job_mole.pbs"
  [unimol]="$BASE/Uni-Mol/ksc_job_unimol.pbs"
)
declare -A JOB_DIR=(
  [tabfm]="$BASE/tabfm"
  [mole]="$BASE/MolE"
  [unimol]="$BASE/Uni-Mol"
)

if [ $# -eq 0 ]; then
  TARGETS=(tabfm mole unimol)
else
  TARGETS=("$@")
fi

LOGFILE="submit_ksc_all_$(date +%Y%m%d_%H%M%S).log"
echo "제출 로그: $LOGFILE"

declare -A SUBMITTED_JOBID=()

for name in "${TARGETS[@]}"; do
  script="${JOB_SCRIPT[$name]:-}"
  dir="${JOB_DIR[$name]:-}"

  if [ -z "$script" ]; then
    echo "!!! 알 수 없는 대상: $name (tabfm/mole/unimol 중 하나여야 함)" | tee -a "$LOGFILE"
    continue
  fi
  if [ ! -f "$script" ]; then
    echo "!!! $script 없음 -> $name 건너뜀 (해당 repo를 clone/pull 했는지 확인)" | tee -a "$LOGFILE"
    continue
  fi

  echo "--- $name 제출: qsub $(basename "$script") (작업 디렉토리: $dir) ---" | tee -a "$LOGFILE"
  jobid=$(cd "$dir" && qsub "$script" 2>&1)
  if [[ "$jobid" =~ ^[0-9] ]]; then
    echo "$name -> job id: $jobid" | tee -a "$LOGFILE"
    SUBMITTED_JOBID[$name]="$jobid"
  else
    echo "!!! $name qsub 실패: $jobid" | tee -a "$LOGFILE"
  fi
done

echo "" | tee -a "$LOGFILE"
echo "=== 제출 요약 ===" | tee -a "$LOGFILE"
for name in "${!SUBMITTED_JOBID[@]}"; do
  echo "  $name: ${SUBMITTED_JOBID[$name]}" | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "=== 현재 내 잡 상태 ($(date)) ===" | tee -a "$LOGFILE"
qstat -u "$USER" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "로그 확인: <repo>/*.log 또는 <repo>/*.pbs.o<jobid> / *.pbs.e<jobid> (qsub 래퍼가 강제하는 이름)" | tee -a "$LOGFILE"
