# evalset + [lm eval harness](https://github.com/EleutherAI/lm-evaluation-harness/), all fp16

better is: lower loss, higher acc and acc_norm

## EleutherAI/gpt-neo-2.7B
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

## gpt2-xl
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
