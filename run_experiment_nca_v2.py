"""
run_experiment_nca_v2.py

NCA v2（JSON出力+Anti-sycophancy）vs Fixed Network vs NCA v1 の3way比較実験。
Fixed は既存結果を流用し、NCA v2 のみ新規実行する。
"""

import json
import sys
import time
from pathlib import Path

# sdnd-proof の src を参照
SDND_SRC = Path("D:/Users/pipe_render/sdnd-proof/src")
sys.path.insert(0, str(SDND_SRC))
sys.path.insert(0, str(Path(__file__).parent))

from task_generator import generate_tasks
from nca_network_v2 import run_nca_network

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FIXED_RESULTS_PATH   = RESULTS_DIR / "fixed_results.jsonl"
NCA_V1_RESULTS_PATH  = RESULTS_DIR / "nca_results.jsonl"
NCA_V2_RESULTS_PATH  = RESULTS_DIR / "nca_v2_results.jsonl"


def nca_verdict_to_bool(verdict: str) -> bool | None:
    if verdict == "CONSISTENT":
        return True
    elif verdict == "CONTRADICTION":
        return False
    return None


def load_fixed_results() -> list:
    with open(FIXED_RESULTS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_nca_v2_experiment(tasks: list) -> list:
    """実験群: NCA v2 で全タスクを実行。"""
    results = []
    correct_count = 0

    print("\n" + "=" * 60)
    print("実験群 (NCA v2 Network) 実行開始")
    print("=" * 60)

    with open(NCA_V2_RESULTS_PATH, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            task_input = f"World rule: {task.world_rule}\nStatement: {task.question}"

            t0 = time.time()
            try:
                nca_result = run_nca_network(task_input)
                verdict = nca_result["final_verdict"]
                prediction = nca_verdict_to_bool(verdict)
                raw_output = verdict
                num_steps = nca_result["num_steps"]
                steps = nca_result.get("steps", [])
            except Exception as e:
                prediction = None
                raw_output = f"ERROR: {e}"
                num_steps = 0
                steps = []

            elapsed = time.time() - t0
            is_correct = (prediction == task.label) if prediction is not None else False
            if is_correct:
                correct_count += 1
            cumulative_accuracy = correct_count / (i + 1)

            record = {
                "task_id": task.task_id,
                "question": task.question,
                "world_rule": task.world_rule,
                "label": task.label,
                "prediction": prediction,
                "is_correct": is_correct,
                "raw_output": raw_output,
                "num_steps": num_steps,
                "elapsed_sec": round(elapsed, 2),
                "cumulative_accuracy": round(cumulative_accuracy, 4),
                "steps": steps,
            }
            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            mark = "o" if is_correct else "x"
            print(f"  [{i+1:03d}/{len(tasks):03d}] {mark} | acc={cumulative_accuracy:.1%} | {elapsed:.1f}s | {task.question[:40]}")

            if (i + 1) % 10 == 0:
                recent = results[-10:]
                recent_acc = sum(1 for r in recent if r["is_correct"]) / len(recent)
                print(f"  --- 直近10件正解率: {recent_acc:.0%} ---")

    return results


def print_summary_v2(results_fixed, results_nca_v2):
    """3way比較サマリーを表示する。"""
    # v1結果を読み込む
    with open(NCA_V1_RESULTS_PATH, "r", encoding="utf-8") as f:
        results_nca_v1 = [json.loads(line) for line in f]

    total = len(results_fixed)
    acc_fixed = sum(1 for r in results_fixed if r["is_correct"]) / total
    acc_v1 = sum(1 for r in results_nca_v1 if r["is_correct"]) / total
    acc_v2 = sum(1 for r in results_nca_v2 if r["is_correct"]) / total

    print(f"\n{'=' * 60}")
    print("3way比較サマリー")
    print(f"{'=' * 60}")
    print(f"  Fixed Network:  {acc_fixed:.1%}")
    print(f"  NCA v1:         {acc_v1:.1%}  (vs Fixed: {acc_v1-acc_fixed:+.1%})")
    print(f"  NCA v2:         {acc_v2:.1%}  (vs Fixed: {acc_v2-acc_fixed:+.1%}, vs v1: {acc_v2-acc_v1:+.1%})")

    # CONSISTENTタスクのみの正解率
    consistent_fixed = [r for r in results_fixed if r["label"] == True]
    consistent_v1 = [r for r in results_nca_v1 if r["label"] == True]
    consistent_v2 = [r for r in results_nca_v2 if r["label"] == True]

    acc_c_fixed = sum(1 for r in consistent_fixed if r["is_correct"]) / len(consistent_fixed)
    acc_c_v1 = sum(1 for r in consistent_v1 if r["is_correct"]) / len(consistent_v1)
    acc_c_v2 = sum(1 for r in consistent_v2 if r["is_correct"]) / len(consistent_v2)

    print(f"\nCONSISTENTタスクのみの正解率（グループシンク改善の指標）:")
    print(f"  Fixed: {acc_c_fixed:.1%} | NCA v1: {acc_c_v1:.1%} | NCA v2: {acc_c_v2:.1%}")

    # CONTRADICTIONタスクのみ
    contra_fixed = [r for r in results_fixed if r["label"] == False]
    contra_v1 = [r for r in results_nca_v1 if r["label"] == False]
    contra_v2 = [r for r in results_nca_v2 if r["label"] == False]

    acc_d_fixed = sum(1 for r in contra_fixed if r["is_correct"]) / len(contra_fixed)
    acc_d_v1 = sum(1 for r in contra_v1 if r["is_correct"]) / len(contra_v1)
    acc_d_v2 = sum(1 for r in contra_v2 if r["is_correct"]) / len(contra_v2)

    print(f"\nCONTRADICTIONタスクのみの正解率:")
    print(f"  Fixed: {acc_d_fixed:.1%} | NCA v1: {acc_d_v1:.1%} | NCA v2: {acc_d_v2:.1%}")

    # 10タスクごと
    print(f"\n10タスクごとの正解率:")
    print(f"  {'Range':<12} {'Fixed':>8} {'NCA v1':>8} {'NCA v2':>8}")
    print(f"  {'-' * 40}")
    for start in range(0, total, 10):
        end = min(start + 10, total)
        n = end - start
        f_acc = sum(1 for r in results_fixed[start:end] if r["is_correct"]) / n
        v1_acc = sum(1 for r in results_nca_v1[start:end] if r["is_correct"]) / n
        v2_acc = sum(1 for r in results_nca_v2[start:end] if r["is_correct"]) / n
        print(f"  {start+1:>4}-{end:<4}    {f_acc:>6.0%}   {v1_acc:>6.0%}   {v2_acc:>6.0%}")

    # グループシンク検出（v2）
    all_contradiction = 0
    all_consistent = 0
    split = 0
    for r in results_nca_v2:
        steps = r.get("steps", [])
        if not steps:
            continue
        final_outputs = steps[-1].get("outputs", [])
        if not final_outputs:
            continue
        verdicts = []
        for out in final_outputs:
            decision = out.get("decision", "UNKNOWN") if isinstance(out, dict) else "UNKNOWN"
            if decision == "CONTRADICTION":
                verdicts.append("CONTRADICTION")
            elif decision == "CONSISTENT":
                verdicts.append("CONSISTENT")
            else:
                verdicts.append("UNKNOWN")
        if all(v == "CONTRADICTION" for v in verdicts):
            all_contradiction += 1
        elif all(v == "CONSISTENT" for v in verdicts):
            all_consistent += 1
        else:
            split += 1

    print(f"\nグループシンク検出（NCA v2）:")
    print(f"  全ノードCONTRADICTION: {all_contradiction} / {total}")
    print(f"  全ノードCONSISTENT:    {all_consistent} / {total}")
    print(f"  分裂:                   {split} / {total}")


def main():
    print("タスクを生成中...")
    tasks = generate_tasks()
    print(f"生成完了: {len(tasks)}タスク")

    # Fixed は既存結果を流用
    print("\n既存のFixed結果を読み込み中...")
    results_fixed = load_fixed_results()
    print(f"読み込み完了: {len(results_fixed)}件")

    # NCA v2 を実行
    results_nca_v2 = run_nca_v2_experiment(tasks)

    # 3way比較サマリー
    print_summary_v2(results_fixed, results_nca_v2)

    print(f"\n結果ファイル:")
    print(f"  {NCA_V2_RESULTS_PATH}")


if __name__ == "__main__":
    main()
