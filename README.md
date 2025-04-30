# MSO3255 Assessment 2: Mini Project

## Improving and Evaluating a Distilled Question Answering Model

This repository contains the code and documentation for a mini-project focused on building a compact and efficient question answering (QA) system using **knowledge distillation** on the SQuAD v1.0 dataset.

Starting from a distilled baseline student model, the project implements several enhancements:

- Intermediate layer distillation (hidden-state alignment with teacher)
- Hyperparameter tuning (α and temperature for distillation loss)
- Model pruning and post-pruning fine-tuning
- Final evaluation using the **official SQuAD EM/F1 metrics**
- Live QA demonstrations and visualizations of model performance

The final student model is optimized for low-resource environments by significantly reducing model size and inference time while preserving reasonable QA performance.

---

## Setup Instructions

**Requirements:**
- Python 3.10
- PyTorch 2.0+
- Hugging Face `transformers`, `datasets`, and `evaluate` libraries
- Google Colab with GPU (Colab Pro recommended)

Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install transformers datasets evaluate
```
## Project Structure

- `baseline/`: Initial distilled model implementation.  
- `improvements/`: Intermediate layer distillation, hyperparameter tuning, and pruning logic.  
- `evaluation/`: Loss tracking, SQuAD evaluation, prediction visualization.  
- `teacher_logits.pt`: Precomputed teacher outputs (stored in Google Drive).  
- `report/`: Final project report (PDF).  
- `figures/`: Training loss curves and performance comparison charts.  

---

## Key Results (Final Run: Full Dataset + Official Metrics)

| Model Variant                        | EM (%) | F1 (%) | Parameters (M) | Inference Time (ms) |
|-------------------------------------|--------|--------|----------------|---------------------|
| Baseline Student (5k subset)        | 2.80   | 6.90   | 67             | 28                  |
| + Intermediate Distillation         | 1.30   | 5.49   | 67             | 28                  |
| + Tuning (α = 0.5, T = 4.0)         | 2.00   | 6.59   | 26             | 22                  |
| + Pruning and Fine-tuning (full set) | 1.90  | 6.11   | 26             | 22                  |

The final model was trained on the full SQuAD v1.0 dataset for two full epochs plus one fine-tuning epoch after pruning. Evaluation used the official SQuAD metrics via the Hugging Face `evaluate` library.

---

## Example QA Predictions

| Question                                 | Ground Truth    | Student Answer |
|------------------------------------------|------------------|----------------|
| Which city is Galatasaray based in?      | Istanbul         | istanbul       |
| Who wrote the novel 1984?                | George Orwell    | george         |

These examples show that the final pruned model can still return accurate and context-aware predictions despite significant model compression.

---

## Visualizations

- Training loss convergence during post-pruning fine-tuning  
- Bar chart comparing Exact Match and F1 scores across model variants  
- QA predictions demonstrated on custom question–context pairs within the notebook  

---

## Ethical Considerations

Knowledge distillation transfers both performance and bias from the teacher model. Since the student model compresses the teacher’s behavior into a smaller structure, nuances—especially those affecting minority examples—may be lost or distorted. Additionally, pruning may reduce interpretability. Future work should incorporate bias auditing and fairness-aware training objectives.

---

## References

- [SQuAD v1.0 Dataset](https://rajpurkar.github.io/SQuAD-explorer/)  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)  
- [Evaluate Library](https://huggingface.co/docs/evaluate/index)  
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)  
- Hinton et al. (2015), *Distilling the Knowledge in a Neural Network*. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)  
- Han et al. (2015), *Learning Both Weights and Connections for Efficient Neural Networks*. [arXiv:1506.02626](https://arxiv.org/abs/1506.02626)  
