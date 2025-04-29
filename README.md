# MSO3255 Assessment 2: Mini Project

## Improving and Evaluating a Distilled Question Answering Model

This repository contains the code and documentation for a mini-project focused on building a compact, efficient question answering (QA) system through **knowledge distillation**.

Starting from a distilled baseline student model (trained on SQuAD v1.0), the project implements several meaningful improvements:

- Intermediate layer distillation (hidden-state alignment)
- Hyperparameter tuning (temperature and α optimization)
- Model pruning and post-pruning fine-tuning
- Evaluation using Exact Match (EM) and F1 scores
- Reproducibility through version control and fixed random seeds

The goal is to improve the balance between **performance** and **model efficiency**, making the student model more suitable for low-resource environments.

## Setup Instructions

**Requirements:**
- Python 3.10
- PyTorch 2.0
- Hugging Face `transformers` and `datasets` libraries
- Google Colab or a local GPU environment (recommended)

Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install transformers datasets
```
## Project Structure

- `baseline/`: Running and evaluating the initial distilled model.
- `improvements/`: Implementation of intermediate distillation, hyperparameter search, and pruning.
- `evaluation/`: Post-training evaluation (EM, F1 scoring, loss curves).
- `teacher_logits.pt`: Saved teacher outputs for efficient student training (stored in Google Drive).
- `figures/`: Training loss plots and performance comparison charts.
- `report/`: Final project report (PDF format).

## Key Results

| Model Variant                  | EM (%) | F1 (%) | Parameters (M) | Inference Time (ms) |
|---------------------------------|--------|--------|----------------|---------------------|
| Baseline Student (5k)           | 2.8    | 6.9    | 67             | 28                  |
| + Intermediate Distillation     | 1.3    | 5.5    | 67             | 28                  |
| + Tuning (α=0.5, T=4.0)          | 2.0    | 6.59   | 26             | 22                  |
| + Pruning and Fine-tuning       | 1.3    | 5.3    | 26             | 22                  |

*Note: Results are based on training with 5,000 examples (subset of SQuAD v1.0) and 2 epochs due to computational limits.*

## Example Outputs

- Loss convergence plots after pruning and fine-tuning
- Bar charts comparing EM and F1 across model variants
- Sample QA predictions on unseen context-question pairs

## Ethical Considerations

While distillation and compression make models more deployable, they may also amplify biases inherited from the teacher model.  
Compression techniques like pruning can further reduce interpretability. Future extensions should include fairness evaluations and bias mitigation strategies.

## References

- [SQuAD v1.0 Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
