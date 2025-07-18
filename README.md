<details><summary>目录</summary><p>

- [Post-training](#post-training)
    - [Finetuning 模式](#finetuning-模式)
    - [SFT tools](#sft-tools)
- [理解 LLM 微调](#理解-llm-微调)
    - [微调的实际应用](#微调的实际应用)
    - [微调的优势](#微调的优势)
    - [微调的误解](#微调的误解)
    - [常见问题](#常见问题)
        - [为什么应该结合 RAG 与微调](#为什么应该结合-rag-与微调)
        - [LoRA 与 QLoRA：选择哪一个？](#lora-与-qlora选择哪一个)
        - [实验是关键](#实验是关键)
- [LLM 微调最佳实践](#llm-微调最佳实践)
    - [模型选择](#模型选择)
        - [指令模型](#指令模型)
        - [基础模型](#基础模型)
        - [如何选择](#如何选择)
    - [微调方法选择](#微调方法选择)
    - [数据选择](#数据选择)
    - [运行资源评估](#运行资源评估)
    - [关键参数](#关键参数)
    - [避免过拟合与欠拟合](#避免过拟合与欠拟合)
        - [过拟合](#过拟合)
        - [欠拟合](#欠拟合)
    - [训练和评估](#训练和评估)
        - [训练](#训练)
        - [评估](#评估)
    - [模型推理和保存](#模型推理和保存)
        - [模型推理](#模型推理)
        - [模型保存](#模型保存)
- [LLM 微调技术原理](#llm-微调技术原理)
- [LLM 微调实战](#llm-微调实战)
- [LoRA 和 QLoRA](#lora-和-qlora)
</p></details><p></p>

# Post-training

* Supervised finetuning(SFT)
    - Instruction finetuning
* Reinforcement Learning with Hunman Feedback(RLHF)
* Direct Preference Optimization(DPO)
* Online
    - 现有数据集上
* Offline
    - 奖励模型选择偏好响应实时应用于优化步骤（即在训练期间）
* Knowledge distillation
* Synthetic data

## Finetuning 模式

* SFT Only
* SFT + RLHF
* SFT + DPO

## SFT tools

* HuggingFace `trl`
* Unsloth `unsloth`
* Lit GPT `lightning`

# 理解 LLM 微调

微调 LLM 可以定制其行为，深化其领域专业知识，并优化其针对特定任务的性能。
通过使用专门数据对预训练模型（例如 Llama-3.1-8B）进行精炼，你可以：

* **更新知识**：引入模型原本未包含的新领域特定信息。
* **定制行为**：调整模型的语气、个性或响应风格，以满足特定需求或品牌声音。
* **针对任务进行优化**：提高模型在特定任务或查询中的准确性和相关性。

将微调想象成将通用模型转化为专业专家。有人争论是否应该使用检索增强生成（RAG）而不是微调，
但微调可以将知识和行为直接整合到模型中，这是 RAG 无法做到的。
在实践中，结合这两种方法能取得最佳效果——带来更高的准确性、更好的可用性和更少的幻觉。

## 微调的实际应用

微调可以应用于各种领域和需求。以下是它带来差异的几个实际例子：

* 金融领域的情感分析：训练一个 LLM 来判断一个新闻标题是否对公司产生正面或负面影响，并根据金融背景定制其理解。
* 客户支持聊天机器人：在过去的客户互动基础上进行微调，以提供更准确、个性化的回复，并符合公司的风格和术语。
* 法律文件辅助：在法律文本（合同、案例法、法规）上进行微调，用于合同分析、案例法研究或合规支持等任务，
  确保模型使用精确的法律语言。

## 微调的优势

**微调与检索增强生成：有何不同？**

微调几乎可以做检索增强生成能做的一切——但反过来就不行了。在训练过程中，微调将外部知识直接嵌入模型中。
这使得模型能够处理专业查询、总结文档，并在不依赖外部检索系统的情况下保持上下文。
但这并不意味着检索增强生成没有优势，因为它擅长从外部数据库获取最新信息。
事实上，通过微调也可以检索最新数据，但将检索增强生成与微调结合使用效率更高。

微调提供了基础模型或纯检索系统无法提供的若干显著优势：

1. **特定任务的精通**

微调将领域知识深度整合到模型中。这使得它在处理结构化、重复性或细微查询方面非常有效，
而单独使用检索增强生成的系统往往难以应对。换句话说，经过微调的模型成为其训练任务或内容的专家。

2. **摆脱检索依赖**

经过微调的模型在推理时无需依赖外部数据源。即使连接的检索系统失效或不完整，它仍然可靠，
因为所有所需信息都已经在模型的参数内部。这种自给自足意味着在生产中故障点更少。

3. **响应更快**

经过微调的模型在生成时无需调用外部知识库。跳过检索步骤意味着它们可以更快地生成答案。
这种速度使经过微调的模型成为对时间敏感的应用的理想选择，在这些应用中每一秒都至关重要。

4. **可以自定义行为和语气**

微调允许精确控制模型的沟通方式。这确保模型的响应与品牌的声音保持一致，遵守监管要求，或符合特定的语气偏好。
您将获得一个不仅知道该说什么，而且知道如何以期望的风格表达的模型。

5. **可靠的性能**

即使在同时使用微调和检索增强生成（RAG）的混合设置中，微调后的模型也能提供可靠的备用方案。
如果检索组件未能找到正确信息或返回错误数据，模型的内置知识仍然可以生成有用的回答。
这保证了您的系统更加一致和稳健的性能。

## 微调的误解

尽管微调有许多优势，但仍有一些误解存在。让我们来探讨关于微调的两个最常见的误解：

* **微调会给模型增加新的知识吗？**

是的，绝对可以。一个常见的误解是微调不会引入新的知识，但实际上它确实会。
如果你的微调数据集包含新的特定领域信息，模型在训练过程中会学习这些内容，并将其纳入其响应中。
实际上，微调可以而且确实能从零开始教会模型新的事实和模式。

* **RAG 总是比微调更好吗？**

许多人认为 RAG 将始终优于微调模型，但在微调得当的情况下，情况并非如此。
事实上，一个经过良好微调的模型在专业任务上往往能与基于 RAG 的系统匹敌，甚至超越它们。
声称“RAG 总是更好”的说法通常源于未正确配置的微调尝试：例如，使用了不正确的 LoRA 参数或训练不足。
这里有一份 Unsloth 关于 [LoRA 超参数调优](https://docs.unsloth.ai/get-started/fine-tuning-guide/lora-hyperparameters-guide) 的指南。

* **调很昂贵吗？**

尽管完整微调或预训练可能成本高昂，但这些并非必要（预训练尤其不是必要的）。在大多数情况下，LoRA 或 QLoRA 微调成本极低。

## 常见问题

### 为什么应该结合 RAG 与微调

与其在 RAG 和微调之间做选择，不如将两者结合起来以获得最佳效果。将检索系统与微调模型结合能发挥每种方法的优点。原因如下：

* **任务特定专长**：微调在专业任务或格式方面表现出色（使模型成为特定领域的专家），而 RAG 则使模型保持最新外部知识。
* **更好的适应性**：即使检索组件失效或返回不完整信息，微调模型仍能给出有用答案。
  同时，RAG 确保系统保持最新状态，无需为每价新数据重新训练模型。
* **效率**：微调为模型提供了强大的基础知识库，而 RAG 能够处理动态或快速变化的细节，无需从头进行彻底的重新训练。
  这种平衡产生了高效的流程，并降低了整体计算成本。

### LoRA 与 QLoRA：选择哪一个？

在实施微调方面，有两种流行的技术可以显著降低计算和内存需求：**LoRA** 和 **QLoRA**。
以下是每种技术的简要比较：

* **LoRA (Low-Rank Adaptation)**：仅微调一小部分额外的“适配器(adapter)”权重矩阵（以 16 位精度），
  而保持大部分原始模型不变。这显著减少了训练期间需要更新的参数数量。
* **QLoRA (Quantized LoRA)**：合 LoRA 与模型权重的 4 位(4-bit)量化，能够在极少的硬件上高效微调非常大的模型。
  通过尽可能使用 4 位(4-bit)精度，它显著降低了内存使用和计算开销。

推荐从 QLoRA 开始，因为它是最高效且最易获取的方法之一。得益于 Unsloth 的动态 4 位量化技术，
与标准 16 位 LoRA 微调相比，精度损失现在可以忽略不计。

### 实验是关键

没有单一的“最佳”微调方法——只有适用于不同场景的最佳实践。重要的是尝试不同的方法和配置，
以找到最适合您的数据集和使用场景的方案。一个很好的起点是 QLoRA（4 位），
它提供了一种非常经济、资源友好的方式来微调模型，而无需复杂的计算要求。

# LLM 微调最佳实践

> 微调没有唯一的“最佳”方法，只有最佳实践。实验是找到适合你需求的关键。

## 模型选择

> 指令模型(Instruct Model)还是基础模型(Base Model)？

在准备微调时，您将面临的一个早期决策是是否使用指令模型或基础模型。

我们建议从 Instruct 模型开始，因为它们允许使用对话式聊天模板（如 ChatML、ShareGPT 等）进行直接微调，
并且与 Base 模型（使用 Alpaca、Vicuna 等）相比，所需数据更少。

* 以 `"unsloth-bnb-4bit"` 结尾的模型名称表示它们是 Unsloth 动态 4 位量化模型。
  这些模型比标准的 BitsAndBytes 4 位模型消耗稍多一些 VRAM，但提供了显著更高的准确性。
* 如果模型名称以 `"bnb-4bit"` 结尾，且不包含 "unsloth"，则表示它是指标准的 BitsAndBytes 4 位量化。
* 没有后缀的模型处于其原始的 16 位或 8 位格式。虽然它们是官方模型创作者的原始模型，
  但我们有时会包含重要的修复——例如聊天模板或分词器修复。因此，当可用时，建议使用我们的版本。

### 指令模型

指令模型是预先使用内置指令进行训练的，因此无需微调即可直接使用。
这些模型包括 GGUF 和其他常见模型，它们针对直接使用进行了优化，能够立即有效地响应提示。
指令模型可与 ChatML 或 ShareGPT 等对话式聊天模板配合使用。

### 基础模型

另一方面，基础模型是未经指令微调的原始预训练版本。这些模型专门设计用于通过微调进行定制，
允许您根据独特需求进行调整。基础模型兼容 Alpaca 或 Vicuna 等指令式模板，
但通常不支持开箱即用的对话式聊天模板。

### 如何选择

这个决定通常取决于你的数据数量、质量和类型：

* 1,000+ 行数据：如果你有一个超过 1,000 行的大型数据集，通常最好对基础模型进行微调。
* 300–1,000 行高质量数据：对于中等规模、高质量的数据集，微调基础模型或指令型模型都是可行的选择。
* 少于 300 行：对于较小的数据集，指令模型通常是更好的选择。微调指令模型可以使其符合特定需求，
  同时保留其内置的指令功能。这确保了它可以在没有额外输入的情况下遵循一般指令，除非你打算显著改变其功能。

## 微调方法选择

如果你是初学者，最好从 Llama 3.1（8B）等小型指令模型开始，并在此基础上进行实验。
你还需要在 QLoRA 和 LoRA 训练之间做出选择：

* LoRA：在 16 位浮点数中微调小型可训练矩阵，而无需更新所有模型权重。
* QLoRA：结合 LoRA 与 4 位量化技术，以最少的资源处理非常大的模型。

还有其他可以切换的设置：

* `max_seq_length = 2048`：控制上下文长度。
    - 虽然 Llama-3 支持 8192，但我们建议测试时使用 2048。Unsloth 能够实现 4 倍更长的上下文微调。
* `dtype = None`：默认为 `None`；对于较新的 GPU，请使用 `torch.float16` 或 `torch.bfloat16`。
* `load_in_4bit = True`：启用 4 位(4-bit)量化，将内存使用减少 4 倍以进行微调。
  禁用它允许启用 LoRA 16 位微调。
* 要启用完整微调(FFT)，请设置 `full_finetuning = True`。对于 8 位(8-bit)微调，
  请设置 `load_in_8bit = True`。注意：一次只能将一种训练方法设置为 `True`。

我们推荐从 QLoRA 开始，因为它是最易于使用且最有效的模型训练方法之一。
我们动态的 4 位量化技术，现在已基本恢复了 QLoRA 相比 LoRA 的精度损失。

您也可以使用 Unsloth 进行文本到语音（TTS）、推理（GRPO）、视觉、强化学习（DPO、ORPO、KTO）、
持续预训练、文本补全以及其他训练方法。

## 数据选择

对于 LLMs，数据集是可用于训练我们模型的数据集合。为了能用于训练，文本数据需要是可分词的格式。

* 您需要创建一个数据集，通常包含两列——问题和答案。质量和数量将很大程度上反映您微调的最终结果，
  因此必须确保这一部分正确无误。
* 您可以使用 ChatGPT 或本地 LLMs [合成生成数据](https://docs.unsloth.ai/basics/datasets-guide#synthetic-data-generation)并构建数据集（如问答对）。
* 您也可以使用我们新的合成数据集笔记本，该笔记本可以自动解析文档（PDF、视频等），
  生成问答对，并使用本地模型（如 Llama 3.2）自动清理数据。在此处访问笔记本。
* 微调可以从现有的文档库中学习，并持续扩展其知识库，但仅靠单纯地倾倒数据效果不会那么好。
  为了获得最佳结果，需要精心策划一个结构良好的数据集，最好是问答对形式。
  这能提升学习效果、理解能力以及响应的准确性。
* 但是，情况并非总是如此，例如，如果你正在微调用于代码的 LLM，
  直接将所有代码数据全部倒入实际上可以使你的模型获得显著的性能提升，即使没有结构化格式。
  所以这真的取决于你的使用场景。

## 运行资源评估

## 关键参数

* [LoRA 超参数调优指南](https://docs.unsloth.ai/get-started/fine-tuning-guide/lora-hyperparameters-guide)

* Learning Rate: 定义模型权重每一步训练的调整量。
    - 较高的学习率：加快训练速度，减少过拟合，但需确保不要过高，否则会导致过拟合。
    - 较低的学习率：训练更稳定，但可能需要更多轮次。
    - 典型范围：`1e-4(0.0001)` 至 `5e-5(0.00005)`。
* Epochs: 模型完整看到训练数据集的次数。
    - 推荐：1-3 个 epoch（超过 3 个 epoch 通常不是最优的，除非你希望你的模型产生更少的幻觉，
      但也意味着答案的创造性和多样性会降低）
    - 更多周期：更好的学习效果，但过拟合风险更高。
    - 较少周期：可能导致模型欠拟合。

## 避免过拟合与欠拟合

### 过拟合

> 过于专业

模型记住了训练数据，无法泛化到未见过的输入。解决方案：

* 如果训练时间短，降低学习率(learning rate)。对于较长的训练，提高学习率(learning ratge)。
  因此，最好测试两者，看看哪个效果更好。
* 增加批处理大小(batch size)。
* 降低训练轮数(epochs)。
* 将你的数据集与通用数据集合并，例如 ShareGPT
* 增加 dropout 率以引入正则化。

### 欠拟合

> 过于泛化

尽管不常见，欠拟合是指低秩模型由于缺乏可学习参数而无法泛化，因此您的模型可能无法从训练数据中学习。解决方案：

* 如果训练时间短，请提高学习率(learning rate)。对于较长的训练运行，请降低学习率(learning rate)。
* 增加 `rank` 和 `alpha`。`alpha` 应至少等于 `rank` 数，而 `rank` 对于较小的模型/更复杂的数据集应更大；
  它通常在 `4` 到 `64` 之间。
* 使用更符合领域相关性的数据集。

## 训练和评估

### 训练

一旦所有设置完成，就可以开始训练了！如果遇到问题，记得你可以随时调整超参数、数据集等。

训练过程中你会看到一些数字的日志！这是训练损失，你的任务是调整参数使它尽可能接近 0.5！
如果你的微调没有达到 1、0.8 或 0.5，你可能需要调整一些数值。如果损失变为 0，那可能也不是一个好迹象！

我们通常建议保持默认设置，除非你需要更长时间的训练或更大的批量大小。

* `per_device_train_batch_size = 2`：增加以提升 GPU 利用率，但需注意因填充导致的训练速度变慢。
  相反，增加 `gradient_accumulation_steps` 以实现更平滑的训练。
* `gradient_accumulation_steps = 4`：模拟更大的批处理大小，而不会增加内存使用。
* `max_steps = 60`：加速训练。对于完整运行，替换为 `num_train_epochs = 1` （建议 1-3 个 epoch 以避免过拟合）。
* `learning_rate = 2e-4`：降低以实现更慢但更精确的微调。尝试 `1e-4`、`5e-5` 或 `2e-5` 等值。

### 评估

为了评估，你可以通过直接与模型聊天来进行手动评估，看看是否符合你的喜好。
你也可以启用 Unsloth 的评估功能，但要注意，这可能会根据数据集的大小而耗时。
为了加快评估速度，你可以：减少评估数据集的大小或设置 `evaluation_steps = 100`。

对于测试，你也可以取你训练数据的 20% 来用于测试。如果你已经用完了所有训练数据，那么你就需要手动评估它。
你也可以使用自动评估工具，如 EleutherAI 的 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。
要注意的是，自动化工具可能无法完全符合你的评估标准。

## 模型推理和保存

### 模型推理

现在让我们在完成训练过程后运行模型！你可以编辑黄色下划线部分！实际上，
因为我们创建了一个多轮聊天机器人，我们现在也可以像它过去看到一些对话那样调用模型，如下所示：

提醒 Unsloth 本身也提供了原生 2 倍的推理速度，所以永远不要忘记调用 `FastLanguageModel.for_inference(model)`。
如果你想让模型输出更长的响应，将 `max_new_tokens = 128` 设置为 256 或 1024 等更大的数字。
请注意，你也需要等待更长时间才能得到结果！

### 模型保存

为了在 Ollama、vLLM、Open WebUI 等期望的推理引擎中保存和使用您的模型，我们这里可以提供更多信息：

* https://docs.unsloth.ai/basics/running-and-saving-models

现在我们可以将微调后的模型保存为一个名为 LoRA 适配器的 100MB 小文件，如下所示。
如果您想上传您的模型，也可以将其推送到 Hugging Face Hub！
记得通过 https://huggingface.co/settings/tokens 获取 Hugging Face 令牌并添加您的令牌！

保存模型后，我们再次可以使用 Unsloth 来运行模型！使用 `FastLanguageModel` 再次调用它进行推理！

# LLM 微调技术原理

> 大模型参数高效微调技术原理综述

1. [背景、参数高效微调简介](https://zhuanlan.zhihu.com/p/635152813)
2. [BitFit、Prefix Tuning、Prompt Tuning](https://zhuanlan.zhihu.com/p/635686756)
3. [P-Tuning、P-Tuning v2](https://zhuanlan.zhihu.com/p/635848732)
4. [Adapter Tuning 及其变体](https://zhuanlan.zhihu.com/p/636038478)
5. [LoRA、AdaLoRA、QLoRA](https://zhuanlan.zhihu.com/p/636215898)
6. [MAM Adapter、UniPELT](https://zhuanlan.zhihu.com/p/636362246)
7. [Guide to fine-tuning LLMs using PEFT and LoRa techniques](https://www.mercity.ai/blog-post/fine-tuning-llms-using-peft-and-lora)
8. [In-depth guide to fine-tuning LLMs with LoRA and QLoRA](https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora)

# LLM 微调实战

> 模型参数高效微调技术实战

1. [PEFT 概述](https://zhuanlan.zhihu.com/p/651744834)

# LoRA 和 QLoRA

* [Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA)](https://lightning.ai/pages/community/tutorial/lora-llm/)
* [Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments](https://lightning.ai/pages/community/lora-insights/)
* [Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
* [Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation (DoRA) from Scratch](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch)
* [rasbt/LLM-finetuning-scripts](https://github.com/rasbt/LLM-finetuning-scripts/tree/main)
* [rasbt/dora-from-scratch](https://github.com/rasbt/dora-from-scratch)
* [DoRA](https://github.com/NVlabs/DoRA)
