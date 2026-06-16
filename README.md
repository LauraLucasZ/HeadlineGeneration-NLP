# HeadlineGeneration-NLP

This project explores automatic news headline generation using transformer-based
natural language generation models. It starts with exploratory data analysis and
preprocessing of news articles, then fine-tunes and evaluates several Hugging Face
models for generating concise headlines from article text.

The work is organized as a set of Google Colab notebooks covering the full
pipeline: dataset inspection, train-ready preprocessing, model training, inference,
and ROUGE-based evaluation.

## Project Overview

The main task is text-to-text headline generation:

- Input: a full or shortened news article
- Output: a short headline that summarizes the article

The project experiments with multiple model families:

- T5 Small
- T5 Base
- FLAN-T5 Small
- FLAN-T5 Base
- BART Base
- GPT-2

## Repository Structure

| File | Purpose |
| --- | --- |
| `headlineGenerationNLP_AllTheNews_DS.ipynb` | Initial exploratory data analysis of the news dataset. |
| `headlineGenerationNLP_trainReadyPreprocessing.ipynb` | Cleans, validates, splits, and exports model-ready training data. |
| `headlineGenerationNLP_dataset_analysis_charts.ipynb` | Produces dataset statistics, before/after preprocessing analysis, and charts. |
| `headlineGenerationNLP_t5_small.ipynb` | Fine-tunes and evaluates a T5-Small headline generation model. |
| `t5_base_model.ipynb` | Fine-tunes and evaluates a T5-Base model. |
| `headlineGenerationNLP_flan_t5_small.ipynb` | Fine-tunes and evaluates a FLAN-T5-Small model. |
| `headlineGenerationNLP_flan_t5_base.ipynb` | Fine-tunes and evaluates a FLAN-T5-Base model. |
| `FLAN_T5_SMALL.ipynb` | Additional FLAN-T5-Small training/evaluation notebook. |
| `headlineGenerationNLP_BART_base.ipynb` | Fine-tunes BART Base for headline generation. |
| `gpt2_model.ipynb` | Fine-tunes and evaluates GPT-2 for headline generation. |

## Workflow

1. Explore the raw news dataset using
   `headlineGenerationNLP_AllTheNews_DS.ipynb`.
2. Preprocess the dataset using
   `headlineGenerationNLP_trainReadyPreprocessing.ipynb`.
3. Review preprocessing results and token-length analysis using
   `headlineGenerationNLP_dataset_analysis_charts.ipynb`.
4. Train one or more models using the model-specific notebooks.
5. Evaluate generated headlines with ROUGE metrics.
6. Save trained models, tokenizers, predictions, and evaluation CSV files.

## Dataset

The notebooks expect news article data with article text and headline/title
fields. The preprocessing notebook exports train, validation, and test splits for
each model format.

Common generated files include:

- `news_headline_train_wide.csv`
- `news_headline_validation_wide.csv`
- `news_headline_test_wide.csv`
- `t5_small_train.csv`, `t5_small_validation.csv`, `t5_small_test.csv`
- `t5_base_train.csv`, `t5_base_validation.csv`, `t5_base_test.csv`
- `flan_t5_small_train.csv`, `flan_t5_small_validation.csv`, `flan_t5_small_test.csv`
- `flan_t5_base_train.csv`, `flan_t5_base_validation.csv`, `flan_t5_base_test.csv`
- `bart_base_train.csv`, `bart_base_validation.csv`, `bart_base_test.csv`
- `gpt2_train.csv`, `gpt2_validation.csv`, `gpt2_test.csv`

Most notebooks are configured for Google Colab and read/write data from Google
Drive paths such as:

```text
/content/drive/MyDrive/headlineGenerationProjectNLP/
```

Update the path variables inside each notebook if your data is stored somewhere
else.

## Setup

The notebooks were designed to run in Google Colab, preferably with a GPU runtime.

Install the main dependencies with:

```bash
pip install transformers datasets accelerate evaluate rouge_score sentencepiece pandas numpy scikit-learn matplotlib seaborn
```

Main libraries used:

- `transformers`
- `datasets`
- `evaluate`
- `rouge_score`
- `sentencepiece`
- `torch`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Running the Notebooks

1. Open the notebook in Google Colab or Jupyter.
2. If using Colab, enable GPU:
   `Runtime -> Change runtime type -> GPU`.
3. Mount Google Drive when prompted.
4. Confirm that the dataset paths in the notebook match your Drive folder.
5. Run the preprocessing notebook before running the model notebooks.
6. Run a model notebook from top to bottom to train, evaluate, and save results.

## Evaluation

The project uses ROUGE metrics to compare generated headlines against reference
headlines:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- ROUGE-Lsum

Recorded evaluation results from the notebooks include:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 | 0.4231 | 0.3581 | 0.3811 | 0.3811 |
| T5-Small | 0.3705 | 0.1741 | 0.3409 | 0.3410 |
| T5-Base | 0.3501 | 0.2260 | 0.3284 | 0.3372 |
| FLAN-T5-Small | 0.3182 | 0.1321 | 0.2951 | 0.2951 |
| FLAN-T5-Base | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Note: the FLAN-T5-Base score shown above appears in the notebook output as zero
for all ROUGE metrics, which may indicate an incomplete run, formatting mismatch,
or evaluation issue. Re-run that notebook before using the result for final model
comparison.

## Outputs

Depending on the notebook, generated artifacts may include:

- Fine-tuned model checkpoints
- Saved tokenizers
- Final test predictions
- Generated headline examples
- Evaluation CSV files
- Dataset analysis dashboards and charts

Example output files referenced by the notebooks:

- `gpt2_evaluation_results.csv`
- `flan_t5_evaluation_results.csv`
- `final_test_predictions.csv`
- `generated_headlines_examples.csv`
- `before_after_analysis_dashboard.csv`

## Example Inference Flow

After a model and tokenizer are saved, the notebooks reload them with Hugging Face
APIs and generate a headline from a new article:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

article = "Your news article text goes here."
input_text = "summarize: " + article

inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
outputs = model.generate(**inputs, max_length=64, num_beams=4)
headline = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(headline)
```

GPT-2 uses a causal language modeling setup, so its inference code differs from
the sequence-to-sequence models.

## Notes

- Large datasets, trained checkpoints, and generated outputs are not stored in
  this repository.
- The notebooks contain absolute Google Drive paths that should be changed for a
  different environment.
- Run the preprocessing notebook first so that model-specific CSV files exist.
- For reproducible comparison, use the same train/validation/test splits across
  all model experiments.
