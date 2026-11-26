# Web pages for the resume injection scanner

Home (`/`) and the scanner (`/scan`), served by `../scanner/scanner_app.py` from `dist/`.

    npm install
    npm run build        # after any change under src/, then reload the page
    npm run dev          # live reload on port 5173; needs scanner_app.py running on 8502

- `src/pages/Home.tsx`, `src/pages/Scan.tsx`: the two pages.
- `src/components/site/`: the header, the rendered page with its marks, the view switch and the drop zone.
- `src/components/ui/`: shadcn controls on Radix, themed in `src/index.css`.
- `src/data/evidence.json`, `public/specimens/`: written by `../scanner/build_home_specimens.py` from the real scanner and `../results/`. Do not edit by hand.
- `DESIGN.md`, `PRODUCT.md`: the design system and the product brief.

Fonts (Source Serif 4, IBM Plex Sans, IBM Plex Mono) come from `@fontsource` packages and are bundled into `dist/`, so the pages work offline.
A result can be opened by link: `/scan?sample=r03_software_engineering_analyst_32985311.pdf&attack=white_text`.
