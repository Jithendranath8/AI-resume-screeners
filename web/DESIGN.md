# Design

The visual system of the web pages. Tokens live in `src/index.css`.

## The idea: select all

The oldest way to catch white text in a document is to select the whole page: the selection paints where
text is, even where nothing is printed. The pages borrow that gesture. "As a program reads it" selects
every line of text in the file; a selected line with nothing under it is hidden text, and the scanner
prints what it found there.

## Colour

| Token | Value | Job |
|---|---|---|
| `paper` | `#ffffff` | Page ground, rendered pages, the report |
| `desk` | `#eef0f3` | The surface a document lies on: stages, the closing band |
| `ink` | `#111417` | Text, primary buttons, chart bars |
| `graphite` | `#59616a` | Secondary text |
| `line` / `line-strong` | `#dfe3e8` / `#bcc3cc` | Hairlines, input borders |
| `select` / `select-tint` / `select-ink` | `#0b5fd9` / `#d6e6ff` / `#0a3a86` | "In the file": the selection, hidden text, keyboard focus |
| `marker` | `#ffe45c` | Highlighter: "reads as an instruction" |
| `attack` / `review` / `clean` | `#c2261f` / `#915700` / `#17754a` | Verdict words, glyphs and chart segments only |

Blue and yellow keep their meanings everywhere and are never used as decoration. Buttons are ink, not
blue, so that blue always means "text in the file". Verdicts carry a word and a shape as well as a colour.

## Type

- **Source Serif 4** (variable, optical size) for headings: weight 500, tight tracking. Resumes are typeset documents; so are these headings.
- **IBM Plex Sans** for body and controls: 17px body, 19px introductions, 15px controls.
- **IBM Plex Mono** only for text the scanner extracted from a file.

All three are bundled from npm (`@fontsource`), so the pages work offline. Sentence case throughout; no
capitalised labels, no label above a heading.

## Layout

Left-aligned on a 1240px column, white ground, sections separated by a hairline and generous space
rather than coloured bands. A rendered page always sits on a `desk` stage with the view switch above it.
Corners: 2px on marks, 6px on controls, 10px on stages. One shadow, under paper.

## Signature

- **Home opening.** A real sample resume is shown as printed, then selected line by line, once. Four selected lines have nothing under them.
- **Page view switch.** "As a person sees it" and "As a program reads it" on every rendered page.
- **Numbered markers.** An ink number in the page margin ties each mark to its finding in the report.

## Rules

- Show the scanner's own output. Regenerate it with `../scanner/build_home_specimens.py`; never edit `src/data/evidence.json` by hand.
- Motion answers an action, apart from the one opening sweep, and is skipped when reduced motion is requested.
- One light theme.
