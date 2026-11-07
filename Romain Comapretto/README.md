# BERT Pre-training and Text Analysis for Ancient Greek

This project develops tools for analysing Ancient Greek text. The Weeks 1–8 notebooks cover language model training, grammatical annotation, and small experiments in passage classification and retrieval. The speech phase is planned to begin in Week 9.

## Language model training

`pretrain.ipynb` reads an Ancient Greek text corpus and a previously built byte-pair encoding (BPE) tokenizer. It initializes a 12-layer BERT model from random weights (109.1 million parameters). During training, the model uses the surrounding text to predict selected tokens.

The pilot uses a random 95/5 train–validation split, a maximum input length of 128 tokens, and a 15% masking rate. The saved run completed 5,000 training steps and recorded a final validation loss of 15.9632. It saved the model to `/kaggle/working/ancient-greek-bert/final_pilot_model`.

## Text analysis

`test_pipeline.ipynb` starts its downstream experiments from the public `pranaydeeps/Ancient-Greek-BERT` checkpoint. The tagging and classification experiments fine-tune this model for each task. The retrieval experiment uses its base encoder directly.

The notebook reads the Ancient Greek Perseus and PROIEL treebanks in Universal Dependencies' CoNLL-U format. For word-level tasks, it assigns each word's label to its first subword and ignores later subwords when scoring. In *οἱ θεοὶ* (“the gods”), for example, the three tasks ask for the part of speech of **θεοὶ** (*noun*), its grammatical features (*nominative, masculine, plural*), and its dictionary form (*θεός*).

| Task | What the code predicts | Evaluation status |
| --- | --- | --- |
| Part-of-speech tagging | A grammatical category for each word. Separate models are fine-tuned on Perseus and PROIEL. | The project notes report about 93% accuracy on Perseus and 97% on PROIEL. Both evaluation cells lack saved output. |
| Morphological analysis | The full combination of grammatical features for each Perseus word, scored as one label. | The project notes report about 88% accuracy. The evaluation cell lacks saved output. |
| Lemmatization | One dictionary form per Perseus word, selected from a fixed list of 13,601 labels. | The saved run records **67.47% validation accuracy** after three epochs and **63.96% test accuracy**. |

The code builds both the lemma list and the set of morphological feature combinations from the training, validation, and test annotations. The reported lemma score therefore reflects a setting where the list of possible answers already includes labels from the test set. The classifier can only return a lemma from that list.

### Prototypes

- **Named entity recognition.** The examples label *Σωκράτης* as a person and *Ἀθήναις* as a place. The code writes the same three hand-labelled sentences to training, validation, and test files, repeating each twenty times per split. The cell has no saved output. Evaluation on independently annotated text is pending.
- **Passage classification.** The classifier trains on six examples labelled as Homeric epic, Plato/Attic prose, or New Testament Greek. In the saved run, it assigned the Homeric-style phrase *θεὰ ἄειδε τὴν ὀργὴν* to the New Testament group. The notebook reports this single prediction. Held-out evaluation is pending.
- **Passage retrieval.** The notebook averages BERT token vectors and ranks three sentences by cosine similarity to a query. For *Ποῦ ζῶσι οἱ ἀθάνατοι;* (“Where do the immortals live?”), it ranked *Οἱ θεοὶ κατοικοῦσι τὸν Ὄλυμπον.* (“The gods inhabit Olympus”) first, with a cosine similarity of **0.6904**. This is a single retrieval example.

## Running the notebooks

The notebooks run independently and install packages in their cells. The exported Kaggle settings have internet and GPU access switched off. Enable internet for downloads on a fresh runtime or stage the required resources locally.

1. **For pre-training**, select a GPU runtime compatible with the notebook's FP16 setting. Attach `metadata_ancient_greek_corpus.jsonl` (with a `text` field) and `ancient_greek_bpe.json` under `/kaggle/input/datasets/romaincom/ancient-greek-corpus/`. The notebook starts with these prepared inputs. Verifying OGA ingestion and strict Unicode NFC normalization requires the upstream preprocessing code or corpus documentation.
2. **For text analysis**, use the download cells or stage the treebanks at `/kaggle/working/UD_Perseus` and `/kaggle/working/UD_PROIEL`. The model download uses Git LFS and expects the checkpoint at `/kaggle/working/Ancient-Greek-BERT`. Run the cell for the desired task. Skip the standalone cache-removal cell to preserve the local Hugging Face cache.
3. **For a repeatable run**, record the treebank and model revisions and the package versions. The notebook installs unpinned packages, and the saved passage-classification cell reports dependency conflicts after upgrading PyTorch.
