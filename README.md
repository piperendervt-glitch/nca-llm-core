# NCA LLM Experiment

NCA（Neural Cellular Automata）的更新ルールをLLMノードネットワークで検証する実験。

## 構成

- **実験群**: `nca_network.py` — 各ノードが隣接ノードの出力のみを見て回答を更新
- **対照群**: `sdnd-proof/fixed_network.py`（流用）

## 依存

- Ollama（qwen2.5:3b）がローカルで起動していること
- `pip install httpx scipy`

## 実行

```bash
python nca_network.py
```

## タスク

`world_consistency`: 与えられた文章群に論理的矛盾があるかを検出する。
