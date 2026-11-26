# Figures

Result figures, one per result reported in the milestone notebooks, drawn by `report/make_figures.py` from the recorded results.
PNG for reading, PDF for the paper. The notebooks M3 to M5 show these same images. Do not edit them by hand; rerun the script instead.

## Overview

### overview_two_readings

![overview_two_readings](overview_two_readings.png)

The foot of one real resume page with two test attacks added. Left: the page as it prints. Right: every line of text in the file is tinted, as a selection would tint it; two lines have nothing printed under them, an instruction to the screener (highlighted) and a list of keywords.

## M2 Reproduction (M2_Reproduction.ipynb)

### M2_attack_success

![M2_attack_success](M2_attack_success.png)

M2: the ResumeShield benchmark rebuilt locally. Bars are our reconstruction, ticks the values reported in the paper; two filter-only pipelines differ because the corpus and filters are locally authored.

### M2_detection

![M2_detection](M2_detection.png)

M2: the hidden-prompt detector on the 104 benchmark documents. No clean document is flagged; the four misses are the plainly visible payloads.

### M2_by_technique

![M2_by_technique](M2_by_technique.png)

M2: detection by concealment technique. Every hidden payload is caught; only half of the plainly visible ones are. Attack success is 100% for every technique against the naive pipeline and 0% against the full defence.

## M3 and M4 Experiment (M3_Research_Gap.ipynb, M4_Controlled_Experiments.ipynb)

### M4_pipeline

![M4_pipeline](M4_pipeline.png)

M3/M4: the combined defence. Each text block the program extracts is checked twice before the model sees it: against what OCR reads from the rendered page, and by an injection classifier. A block flagged by either check is removed.

### M4_attack_success

![M4_attack_success](M4_attack_success.png)

M4: attack success rate on 24 eligible attacks against llama3.2:3b-instruct-q4_K_M. Clean decisions were unchanged in every configuration except channel separation (83.3% agreement).

### M4_ablation

![M4_ablation](M4_ablation.png)

M4: successful attacks out of those tried, by how the payload was placed and how it was worded. Each check closes the gap the other leaves; the visible paraphrase survives both.

### M4_detection

![M4_detection](M4_detection.png)

M4: the two checks as document-level detectors, on 36 injected documents. Neither flags any of the 6 clean controls.

### M4_classifier_finetuning

![M4_classifier_finetuning](M4_classifier_finetuning.png)

M4: injection classifiers on the text blocks of the 42 experiment documents (exploratory). Round 2 catches every payload block without flagging genuine resume text; its effect on attack success has not been measured.

## M5 Failure analysis (M5_Failure_Analysis.ipynb)

### M5_scores

![M5_scores](M5_scores.png)

M5: the score returned for every eligible attack. A successful attack is not a marginal flip: every one returns exactly 100, the value the payload asks for.

## Pilot (pilot sections of M4 and M5)

### pilot_detection

![pilot_detection](pilot_detection.png)

The earlier sentence-level pilot. The trained detector reached high recall but flagged 18.75% of benign rows, above the limit recorded beforehand, so its hypothesis was not supported.

## Scanner (results/scanner_eval)

### scanner_heldout

![scanner_heldout](scanner_heldout.png)

The scanner on resumes it was not developed on. Rules were changed after runs 1 and 2, so each version was measured again on fresh files; run 3 measures the current rules.
