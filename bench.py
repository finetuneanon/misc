import torch
import time

print("seq_len\tmax_len\truntime\tmemory")
for seq_len in range(128,2049,128):
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  ids = torch.zeros(1, seq_len).long().cuda()
  runtime = 0.
  max_length = min(2049, seq_len + 40)
  for i in range(10):
    s = time.perf_counter()
    outputs = model.generate(ids, use_cache=True, do_sample=True, min_length=max_length, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    runtime += time.perf_counter() - s
    del outputs
  print(f"{seq_len}\t{max_length}\t{runtime/10.}s\t{torch.cuda.max_memory_allocated() / 1024. / 1024. / 1024.:.4f}GB")
  del ids
