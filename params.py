def num_parameters(hidden_size, num_layers, seq_len, vocab_size, human=False):
  frac_1  = 13 / (12 * hidden_size)
  numer = vocab_size + seq_len
  denom = 12 * num_layers*hidden_size
  frac_2 = numer / denom
  x = 1 + frac_1 + frac_2
  x = 12 * num_layers * hidden_size ** 2 * x
  return x
