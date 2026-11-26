# Product

Resume injection scanner: give it a resume PDF and it says whether the file carries an attack on an AI screener, which kind, and where it sits on the page.

## Platform

Web, served locally by `scanner/scanner_app.py` at http://127.0.0.1:8502. Two pages: Home (`/`) and the scanner (`/scan`). There is also a command line, `scanner/resume_scanner.py`.

## Stack

Python standard-library HTTP server and the scanner in `scanner/resume_scanner.py`. Pages in `web/`: React, Vite, Tailwind CSS 4, shadcn components on Radix.

## Users

- People who screen resumes, or run a pipeline that does, and want to check a file or a batch before a model reads it.
- Researchers and reviewers of this project who want to see what the scanner does and how well it was measured.

They use it at a desk, on a laptop, in ordinary office light, a few files at a time.

## Product purpose

Show the difference between what a person sees on a resume and what a program reads from the file, name each hidden text or instruction by how it was done, and leave the decision to a person.

## Operating context

Scanning is offline and files are not stored. Two extras are off until chosen: a test attack added to a file before scanning, and a comparison that asks an AI screener to score the resume with and without the flagged text (this one can send text to a model provider).

## Capabilities and constraints

- Three verdicts: attack found, needs review, no attack found.
- Thirteen attack techniques are recognised and can be added as test attacks.
- PDF only, up to 15 pages and 20 MB. Instructions are recognised in English only.
- No paid service is required.

## Brand commitments

- Selection blue means "in the file". Highlighter yellow means "reads as an instruction". These meanings hold on every page.
- Verdicts always carry a word and a shape as well as a colour.

## Evidence on hand

- `results/real_pdf/pdf_llama32_3b_01/`: attack success on a real model fell from 21 of 24 to 3 of 24 with both removals.
- `results/scanner_eval/confirm_summary.json`: the current scanner on 200 attacked and 140 clean resumes it had not seen.
- The Home page reads both through `scanner/build_home_specimens.py`; no figure on it is typed by hand.

## Product principles

1. Show the scanner's own output, never a mock-up of it.
2. State limits and misses beside the results.
3. A finding is a reason for a person to look, not proof of intent.

## Accessibility and inclusion

Text contrast at least 4.5:1. Every control works from the keyboard and shows focus. Motion is one opening sweep on Home and is skipped when reduced motion is requested. Charts carry their numbers as text.
