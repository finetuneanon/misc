# evalset + [lm eval harness](https://github.com/EleutherAI/lm-evaluation-harness/)

better is: lower loss, higher acc and acc_norm

* [LAMBADA](https://arxiv.org/abs/1606.06031)
* [WinoGrande](https://arxiv.org/abs/1907.10641)
* [PIQA](https://arxiv.org/abs/1911.11641)
* [HellaSwag](https://arxiv.org/abs/1905.07830)

## EleutherAI/gpt-neo-2.7B, fp16
```
mystery literature evalset loss: 2.63671875
```

|   Task   |    Metric     |Value |
|----------|---------------|-----:|
|lambada   |ppl            |5.6244|
|          |ppl_stderr     |0.1389|
|          |acc            |0.6235|
|          |acc_stderr     |0.0068|
|winogrande|acc            |0.5991|
|          |acc_stderr     |0.0138|
|piqa      |acc            |0.7209|
|          |acc_stderr     |0.0105|
|          |acc_norm       |0.7296|
|          |acc_norm_stderr|0.0104|
|hellaswag |acc            |0.4186|
|          |acc_stderr     |0.0049|
|          |acc_norm       |0.5511|
|          |acc_norm_stderr|0.0050|

## EleutherAI/gpt-neo-2.7B, fp32

|   Task   |    Metric     | Value |
|----------|---------------|------:|
|lambada   |ppl            |5.6257|
|          |ppl_stderr     |0.1389|
|          |acc            |0.6224|
|          |acc_stderr     |0.0068|
|winogrande|acc            |0.5959|
|          |acc_stderr     |0.0138|
|piqa      |acc            |0.7220|
|          |acc_stderr     |0.0105|
|          |acc_norm       |0.7291|
|          |acc_norm_stderr|0.0104|
|hellaswag |acc            |0.4198|
|          |acc_stderr     |0.0049|
|          |acc_norm       |0.5521|
|          |acc_norm_stderr|0.0050|

## gpt2-xl, fp16
```
mystery literature evalset loss: 2.8828125
```

|   Task   |    Metric     | Value |
|----------|---------------|------:|
|lambada   |ppl            |10.6355|
|          |ppl_stderr     | 0.3305|
|          |acc            | 0.5115|
|          |acc_stderr     | 0.0070|
|winogrande|acc            | 0.5770|
|          |acc_stderr     | 0.0139|
|piqa      |acc            | 0.7089|
|          |acc_stderr     | 0.0106|
|          |acc_norm       | 0.7040|
|          |acc_norm_stderr| 0.0107|
|hellaswag |acc            | 0.3930|
|          |acc_stderr     | 0.0049|
|          |acc_norm       | 0.5039|
|          |acc_norm_stderr| 0.0050|

## gpt2-xl, fp32

|   Task   |    Metric     | Value |
|----------|---------------|------:|
|lambada   |ppl            |10.6341|
|          |ppl_stderr     | 0.3305|
|          |acc            | 0.5121|
|          |acc_stderr     | 0.0070|
|winogrande|acc            | 0.5793|
|          |acc_stderr     | 0.0139|
|piqa      |acc            | 0.7084|
|          |acc_stderr     | 0.0106|
|          |acc_norm       | 0.7051|
|          |acc_norm_stderr| 0.0106|
|hellaswag |acc            | 0.3931|
|          |acc_stderr     | 0.0049|
|          |acc_norm       | 0.5042|
|          |acc_norm_stderr| 0.0050|
