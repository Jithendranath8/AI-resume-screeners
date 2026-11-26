# Real resume PDFs used for the real-PDF / LLM experiment

Source: Kaggle "Resume Dataset" by Snehaan Bhawal,
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
Licence: CC0 1.0 Public Domain (as stated on the Kaggle dataset page, retrieved 2026-11-25).
Content: resume examples collected from livecareer.com; the dataset author already
replaced names, employers and locations with placeholders ("Company Name", "City, State").
A check with PyMuPDF found no e-mail addresses, phone numbers or URLs in the six files below.
These are real PDF documents with real resume layouts, not synthetic text; they are sample
resumes, not documents of identifiable private individuals.

| File | Kaggle ID | Kaggle category | Title on the resume | Pages |
|---|---|---|---|---|
| r01_software_developer_62994611.pdf | 62994611 | AGRICULTURE | Software Developer | 3 |
| r02_software_engineering_coop_12144825.pdf | 12144825 | AVIATION | Software Engineering Co-op | 1 |
| r03_software_engineering_analyst_32985311.pdf | 32985311 | ENGINEERING | Software Engineering Analyst | 1 |
| r04_aspnet_web_developer_12763627.pdf | 12763627 | INFORMATION-TECHNOLOGY | ASP.NET Web Developer | 2 |
| r05_it_manager_10553553.pdf | 10553553 | INFORMATION-TECHNOLOGY | Information Technology Manager | 2 |
| r06_staff_pharmacist_48037995.pdf | 48037995 | INFORMATION-TECHNOLOGY | Staff Pharmacist | 2 |

Selection: the whole dataset (2,484 PDFs) was scanned for Python/API/SQL keyword counts.
Only 29 resumes mention Python twice or more; r01–r03 are the strongest backend-style
candidates found. r04–r06 were chosen as a near-miss (.NET), a non-developer IT role and an
unrelated profession. Keyword counts informed the shortlist only; the SELECT/REJECT labels
in labels.csv must be assigned by a human against the job rubric before any LLM output is seen.
