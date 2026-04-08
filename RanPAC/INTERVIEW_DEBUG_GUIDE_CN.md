# RanPAC 面试调试指南（零基础友好）

本指南对应你在代码里看到的 `Debug-BP0` 到 `Debug-BP26` 注释。

建议调试命令（Windows / VSCode）:

```bash
python main.py -i 7 -d cifar224
```

---

## 1) 先记住整体流程（面试高频）

从执行路径看，核心链路是:

1. `main.py` 读取参数表（CSV）
2. `trainer.py` 创建 `Learner` + `DataManager`
3. `Learner.incremental_train(...)` 进入每个增量任务
4. `Learner._train(...)`:
   - 首任务可选 PETL 微调（Phase-1）
   - 每任务都做 `replace_fc(...)`（Phase-2，RanPAC 核心）
5. `eval_task()` 评估并记录准确率

论文对应关系（NeurIPS 2023 RanPAC）:

- 预训练 backbone 提特征：`inc_net.py` 的 `convnet`
- 随机投影 RP：`RanPAC.py` 中 `setup_RP()` + `replace_fc()`
- 闭式解分类头（岭回归）：`replace_fc()` 里的 `torch.linalg.solve(...)`

---

## 2) 最推荐断点顺序（按执行流）

1. `main.py` 的 `Debug-BP1`
   - 看 `args` 是否拿到了正确实验配置
2. `trainer.py` 的 `Debug-BP5`
   - 看 `Learner` 实例化成了什么网络
3. `utils/data_manager.py` 的 `Debug-BP24`
   - 看 `_increments`（任务如何切分）
4. `RanPAC.py` 的 `Debug-BP14`
   - 看 `_cur_task / _known_classes / _classes_seen_so_far`
5. `RanPAC.py` 的 `Debug-BP15`
   - 看进入了哪条训练分支（joint / PETL / NCM）
6. `RanPAC.py` 的 `Debug-BP17`
   - 看随机投影参数 `W_rand, Q, G` 初始化
7. `RanPAC.py` 的 `Debug-BP10` + `Debug-BP11`
   - 看 RanPAC 关键数学更新（最重要）
8. `inc_net.py` 的 `Debug-BP19`
   - 看分类头前向逻辑和张量形状
9. `trainer.py` 的 `Debug-BP8`
   - 看最终预测和准确率如何计算

---

## 3) 面试一定会问的“变量理解”

### A. 任务状态变量（必须会）

- `_cur_task`: 当前是第几个增量任务（从 0 开始）
- `_known_classes`: 之前任务已学过多少类
- `_classes_seen_so_far`: 到当前任务为止总共见过多少类
- `class_increments`: 每个任务的类别区间列表，如 `[0, 9], [10, 19]`

### B. RanPAC 数学变量（必须会）

- `Features_f`：backbone 抽到的原始特征，形状 `[N, D]`
- `W_rand`：随机投影矩阵，形状 `[D, M]`
- `Features_h`：投影后特征 `ReLU(Features_f @ W_rand)`，形状 `[N, M]`
- `Y`：one-hot 标签矩阵，形状 `[N, C]`
- `Q`：累积统计量，`Q += H^T Y`，形状 `[M, C]`
- `G`：累积统计量，`G += H^T H`，形状 `[M, M]`
- `Wo`：闭式解得到的分类器权重，`Wo = (G + λI)^(-1)Q` 的等价稳定求解

一句话记忆：

> RanPAC 不反复端到端训练大模型，而是固定/轻调 backbone，用增量统计量 `G,Q` 快速解出分类头。

---

## 4) Python 语法小抄（你现在就够用）

- `obj.attr`：访问对象属性
- `ClassName(...)`：实例化类对象
- `def f(...):`：定义函数
- `if / elif / else`：分支
- `for x in ...`：循环
- `with torch.no_grad():`：关闭梯度，做推理更省显存
- `self`：类实例自身

调试时你只要抓住两件事：

1. 谁在创建对象（实例化）
2. 谁在调用谁（函数调用路径）

---

## 5) 面试答题模板（可直接背）

如果被问“RanPAC 在代码里如何实现？”

你可以答：

1. 入口 `main.py` 从 CSV 读实验参数，进入 `trainer.train`。
2. `Learner` 根据 `model_name` 和 `convnet_type` 实例化预训练 backbone 与增量分类头。
3. 每个任务在 `incremental_train` 更新类别范围并构造 dataloader。
4. 首任务可选 PETL 微调；随后在 `replace_fc` 中提取特征并更新统计量 `G,Q`。
5. 使用岭回归闭式解更新分类头权重，再通过 `eval_task` 得到分组准确率与总准确率。

---

## 6) 你需要注意的当前仓库小改动

- `trainer.py` 当前 `for task in [0]`，只跑第一个任务（便于调试）。
- `RanPAC.py` 当前 `_init_train` 里 `tqdm(range(1))`，只跑 1 个 epoch（便于快速过流程）。

面试时请主动说明：

> 为了快速定位流程我把任务和 epoch 都缩短到最小，完整实验应恢复为全任务和完整训练轮数。
