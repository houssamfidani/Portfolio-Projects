# Legal document classification and model benchmarking

Classifying French court decisions into four legal domains: civil, criminal, commercial, and social/employment law. The project compares TF-IDF classifiers, a fully fine-tuned JuriBERT Transformer, and zero-shot classification through Jev.

**[Open the notebook](legal_text_classification.ipynb)** · [Recorded metrics](results/recorded-metrics.json)

## Recorded results

All three final approaches use the same **5,571 held-out decisions**. Macro-F1 gives equal weight to each class; criminal-law decisions account for only about 1.4% of the dataset.

| Approach | Test macro-F1 | Test accuracy |
|---|---:|---:|
| TF-IDF + LinearSVC | **0.970** | **0.983** |
| Fully fine-tuned JuriBERT-base | 0.958 | 0.980 |
| Jev, zero-shot | 0.800 | 0.875 |

LinearSVC performed best with these settings. The notebook includes per-class scores and confusion matrices to examine where the methods differ.

## Implementation

- Cleaned HTML and normalized text, removed invalid records and label conflicts, and deduplicated exact normalized texts. Retained **27,855 of 27,998 decisions**, split into 22,284 development and 5,571 test documents.
- Compared Logistic Regression and LinearSVC with **five-fold stratified cross-validation**, fitting TF-IDF inside each training fold. Selected LinearSVC at 0.968 mean macro-F1 versus 0.950 for Logistic Regression, then refit it on the full development split.
- Fully fine-tuned [JuriBERT-base](https://huggingface.co/dascim/juribert-base) for **five epochs** on a Colab T4. Used AdamW, FP16, gradient accumulation, and a 512-token head-and-tail representation for long decisions.
- Evaluated `jev-1.14.0` with fixed French category descriptions and no labeled examples. Each request used the full decision unless it exceeded the API limit; two required a head-and-tail fallback.

## Run

From this directory, in a fresh Python 3.12 environment:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab legal_text_classification.ipynb
```

Supply the original `CIV.csv`, `CRIM.csv`, `COM.csv`, and `SOC.csv` files in `data/`. Each is a UTF-8, headerless CSV with two columns, `text,label`; every label must match its filename. The original row order matters because source-based identifiers track exclusions, including two manually reviewed incomplete records.

Run sections 1–3 for preprocessing and the classical baselines. Section 4 downloads JuriBERT and trains it; the notebook requires a CUDA or Apple MPS GPU, with FP16 enabled on CUDA. Section 5 is optional and uses TypeSafe's API. It installs `typesafe-sdk==0.7.5` and reads `TYPESAFE_API_KEY` from the environment or a hidden prompt. Running it sends each test document to the service and requires model access.

The source data and trained weights are not included. Saved results and plots can be reviewed without executing the notebook. Core dependency versions were not captured, so `requirements.txt` lists dependencies rather than a tested environment lock. Results are from the supplied runs, not a fresh execution of this portfolio copy.

## Evaluation limits

Exact normalized duplicates are excluded across splits; near duplicates, related case families, and temporal generalization were not evaluated. JuriBERT used a fixed five-epoch schedule without a separate validation split or hyperparameter search. Logistic Regression was evaluated only through cross-validation.

The models receive different input lengths: TF-IDF uses full texts, JuriBERT uses up to 512 tokens, and Jev usually receives the full text. This is a comparison of these configurations, not a controlled architecture-only experiment. No latency or cost benchmark, multi-seed significance analysis, or production deployment is claimed.
