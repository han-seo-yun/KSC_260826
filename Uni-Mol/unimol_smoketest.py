# Uni-Core 없이 unimol_tools(HuggingFace 자동 다운로드)만으로 학습이 되는지
# 확인하는 최소 스모크 테스트. 로컬 macOS CPU 환경에서 5-fold 중 4-fold까지
# 실제 학습+AUC 계산 성공 확인함 (2026-08-21). KSC GPU에서는 fork 기반
# multiprocessing이라 로컬에서 겪은 macOS spawn 이슈 없이 5-fold 전부 될 것으로 예상.
import pandas as pd
from unimol_tools import MolTrain


def main():
    data = pd.DataFrame({
        "SMILES": [
            "CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O",
            "CCOCC", "CCCl", "CCBr", "c1ccncc1", "CC(C)O",
            "CCCCO", "CCCCN", "c1ccc(O)cc1", "CC(=O)N", "CCOC(=O)C",
            "CC#N", "CCCCC", "c1ccc(Cl)cc1", "CC(C)(C)O", "CCSCC",
        ],
        "TARGET": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    csv_path = "unimol_smoketest_toy_data.csv"
    data.to_csv(csv_path, index=False)

    clf = MolTrain(
        task="classification",
        data_type="molecule",
        epochs=1,
        batch_size=4,
        metrics="auc",
        save_path="unimol_smoketest_exp",
    )
    # MolTrain.fit()은 예측값을 반환하지 않고(train.py: `return` only) None을
    # 반환한다 (2026-08-24 job 891712 실제 확인: TypeError: 'NoneType' object
    # is not subscriptable). 예측값은 clf.cv_pred에 저장돼 있다.
    clf.fit(data=csv_path)
    print("=== SMOKETEST OK ===")
    print(clf.cv_pred[:5])


if __name__ == "__main__":
    main()
