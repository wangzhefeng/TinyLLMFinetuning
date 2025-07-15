<details><summary>目录</summary><p>

- [数据](#数据)
    - [常用数据](#常用数据)
    - [数据格式](#数据格式)
</p></details><p></p>

# 数据

## 常用数据

* 现有的开源数据集，[Spider](https://huggingface.co/datasets/xlangai/spider)
* 使用 LLM 来创建合成数据集，[Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
* 使用人工来创建数据集，如 [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
* 使用上述方法的组合，如：
    - [Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca)
    - [Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/abs/2306.**02707**)

## 数据格式

* 对话格式

> conversational

```json
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

* 指令格式

> instruction

```json
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
```
