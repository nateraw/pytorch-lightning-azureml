# Pytorch Lightning AzureML

## Projects

### MNIST

Currently no script to run this one, but will provide one in the future. It shows a simpler example to get used to the workflow on AzureML.

### Evaluating HuggingFace Transformers Models on GLUE Benchmark

Below is a table I hope to fill out to document the performance of `transformers` models across the GLUE benchmark (and hopefully other benchmarks in the future). Currently only showing CoLA. Models were run on 5 random seeds and I report the median of these 5 runs.

| Task  | Metric                       | `bert-base-uncased` | `distilbert-base-uncased` |
|-------|------------------------------|-------|-----|
| CoLA  | Matthew's corr               | 54.53 | 50.42 |
| SST-2 | Accuracy                     | - | - |
| MRPC  | F1/Accuracy                  | - | - |
| STS-B | Person/Spearman corr.        | - | - |
| QQP   | Accuracy/F1                  | - | - |
| MNLI  | Matched acc./Mismatched acc. | - | - |
| QNLI  | Accuracy                     | - | - |
| RTE   | Accuracy                     | - | - |
| WNLI  | Accuracy                     | - | - |
