# Pytorch Lightning AzureML

## Projects

### MNIST

Currently no script to run this one, but will provide one in the future. It shows a simpler example to get used to the workflow on AzureML.

### Evaluating HuggingFace Transformers Models on GLUE Benchmark

Below is a table I hope to fill out to document the performance of `transformers` models across the GLUE benchmark (and hopefully other benchmarks in the future). Currently only showing CoLA. Below, I report the median of 5 runs (w/ different random seeds) on the dev set for each model/task combination.

| Task  | Metric                       | `bert-base-uncased` | `distilbert-base-uncased` | `albert-base-v2` |
|-------|------------------------------|-------|-----| ---- |
| CoLA  | Matthew's corr               | 54.53 | 50.42 | 57.64 |
| SST-2 | Accuracy                     | - | - | - |
| MRPC  | F1/Accuracy                  | - | - | - |
| STS-B | Person/Spearman corr.        | - | - | - |
| QQP   | Accuracy/F1                  | - | - | - |
| MNLI  | Matched acc./Mismatched acc. | - | - | - |
| QNLI  | Accuracy                     | - | - | - |
| RTE   | Accuracy                     | - | - | - |
| WNLI  | Accuracy                     | - | - | - |
