# pdf_extraction_marker_pymupdf_pdfminer

**Path:** e:\swiss_citation_extraction\notebooks\14_pdf_research_paper_extraction\pdf_extraction_marker_pymupdf_pdfminer.ipynb

## Configuration

- Runtime: Google Colab (Linux with `apt-get`, `/content/drive/MyDrive/...` paths, GPU available for Marker).
- Dependencies installed in-notebook: `PyPDF2`, `pymupdf` (fitz), `pdfminer.six`, `pypdf2`, `tqdm`, `pandas`, `marker-pdf`, plus system package `poppler-utils`.
- Parallelism / thread hygiene:
  - `MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)` (one less than CPU count).
  - Environment vars set to limit thread storms: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`.
  - For Marker run: `torch.cuda.set_device(0)`, `torch.set_num_threads(1)`, `torch.backends.cudnn.benchmark = True`.
- Input paths (Colab/Drive):
  - `PDF_FOLDER = /content/drive/MyDrive/Make_data_count_challenge/Data/train/PDF`
  - `CSV_PATH   = /content/drive/MyDrive/Make_data_count_challenge/Data/train_labels_cleaned.csv`
- Output paths:
  - `in_text_spans_combined.csv` (Combined extractor result)
  - `OUT_MD  = /content/drive/MyDrive/Make_data_count_challenge/marker_md` (Marker rendered markdown)
  - `OUT_CSV = /content/drive/MyDrive/Make_data_count_challenge/marker_doi_matches_gpu.csv` (Marker DOI matches)
- Single-PDF debug (cell 6): `PDF_PATH = /content/drive/MyDrive/Make_data_count_challenge/Data/train/PDF/10.1002_2017jc013030`, `INPUT_DOI = https://doi.org/10.17882/49388`.
- Logging: `PyPDF2` logger set to `ERROR`.

## Data

- Source corpus: PDFs from the Make-Data-Count challenge training set on Google Drive (`Data/train/PDF/`).
- Labels: `train_labels_cleaned.csv`, expected columns `article_id`, `dataset_id` (cast to `str`); the Marker shard runner derives `dataset_doi_norm` via `normalize_doi`.
- Observed corpus scale (from cell 4 stdout): 213 PDFs, 718 label rows.

## Pipeline

The notebook contains four sequential pipelines plus utilities (11 cells total):

1. **Cell 0** - `pip install PyPDF2`.
2. **Cell 1** - "In-text dataset span miner - v3.21" (entire body is commented out; kept as reference). Documented changes vs v3.20: DA-header matching tolerant to letter-spaced ALLCAPS; DA block end relaxed so it does not stop before the DOI line; in-text-span DOI canonicalised to `https://doi.org/<core>`; DA block tried on both pdfminer and PyPDF2 page views before other strategies. Originally built as a `ProcessPoolExecutor` pipeline over `(article_id, dataset_id)` groups with columns `in_text_span, anchor_sentence, footnote_number, footnote_text, arrow_chain, page_index, section_guess, match_label, match_confidence, repo_guess, dataset_in_paper, version_mismatch, relation_hint, span_source, debug, source_type, source_type_label`.
3. **Cell 2** - `apt-get -yqq install poppler-utils` (for downstream text extraction speed).
4. **Cell 3** - Sets `MAX_WORKERS` and thread-storm-suppression env vars.
5. **Cell 4** - "Combined extractor: Ref#-first, v3.21 fallback" (active). Two-stage strategy:
   - (A) DOI -> exact reference number -> find paragraphs that truly cite that ref via superscript / `[n]` / punctuation-guarded `.22,` patterns.
   - (B) If (A) fails, run the v3.21 general pipeline mining Data Availability blocks and inline DOI mentions.
   Implementation imports `fitz` (PyMuPDF), `PyPDF2.PdfReader`, and `pdfminer.high_level.extract_pages` / `pdfminer.layout.LTTextContainer, LAParams`. Iterates groups of `(article_id, dataset_id)`, merges per-row results back via `_row_id`, fills defaults (`NOT FOUND`, `unknown`, ...), and writes `in_text_spans_combined.csv`. Run as `out_df = run_pipeline(CSV_PATH, PDF_FOLDER, DEBUG_ARTICLE_IDS)`.
6. **Cell 5** - `print(out_df["in_text_span"][0])` (quick spot-check of one extracted span).
7. **Cell 6** - Standalone single-PDF debug script. Uses `fitz` directly to: open one PDF, locate the references page, resolve the input DOI to its in-text reference number, then search body pages for paragraphs containing superscript / `[n]` / punctuation-guarded citations to that number. Reports an "Extraction Summary" with input DOI, resolved ref number, body pages with hits, and paragraphs extracted.
8. **Cell 7** - Empty.
9. **Cell 8** - `pip install -q marker-pdf pandas tqdm`.
10. **Cell 9** - "4-way THREADED Marker runner (spawn-free; safe in notebooks)". Uses `marker.converters.pdf.PdfConverter`, `marker.models.create_model_dict`, `marker.output.text_from_rendered` to render each PDF to markdown. DOI helpers (`DOI_CORE_RE`, `BROKEN_DOI_RE`, `HEADING_RE`, `_squash_ws`, `normalize_doi`) find dataset DOIs inside the rendered markdown (including spaced/broken DOIs). Architecture: split unique `article_id`s into `k` shards (`uniq[i::k]`); run `_process_shard_thread` for each shard inside a `ThreadPoolExecutor(max_workers=k)`; each shard writes a per-shard CSV which is then concatenated, sorted, de-duplicated on `(article_id, dataset_id)`, and saved to `OUT_CSV`. Launched with `marker_results_df = run_four_way_threaded(gpu_threads=2)` (comment notes set to 3 if no OOM).
11. **Cell 10** - Empty.

## Results

Cell 0 output:
```
Collecting PyPDF2
  Downloading pypdf2-3.0.1-py3-none-any.whl.metadata (6.8 kB)
Downloading pypdf2-3.0.1-py3-none-any.whl (232 kB)
Installing collected packages: PyPDF2
Successfully installed PyPDF2-3.0.1
```

Cell 2 output:
```
Selecting previously unselected package poppler-utils.
(Reading database ... 126374 files and directories currently installed.)
Preparing to unpack .../poppler-utils_22.02.0-2ubuntu0.10_amd64.deb ...
Unpacking poppler-utils (22.02.0-2ubuntu0.10) ...
Setting up poppler-utils (22.02.0-2ubuntu0.10) ...
Processing triggers for man-db (2.10.2-1) ...
```

Cell 3 output: `'1'` (value of `MAX_WORKERS`).

Cell 4 outputs:
- A tqdm progress bar `Parsing PDFs: 0/213`.
- A stream of `WARNING:pdfminer.pdfinterp:Cannot set gray non-stroke color ...` / `Cannot set non-stroke color because 2 components are specified ...` messages from `pdfminer`.
- Final summary: `Processed 213 PDFs, 718 rows. Found spans for 315 rows.` and `Saved: in_text_spans_combined.csv`.
- Head of `out_df` (first 12 rows shown), with columns `article_id, dataset_id, type, in_text_span, anchor_sentence, footnote_number, ...`. Examples:
  - `10.1002_2017jc013030 | https://doi.org/10.17882/49388 | Primary | Journal of Geophysical Research: Oceans 10.100...`
  - `10.1002_ece3.4466 | https://doi.org/10.5061/dryad.r6nq870 | Primary | DATA ACCESSIBILITY The dataset supporting this...`
  - `10.1002_ece3.5260 | https://doi.org/10.5061/dryad.2f62927 | Primary | DATA AVAILABILITY DNA sequences: GenBank MK838...`
  - `10.1002_mp.14424 | https://doi.org/10.7937/tcia.2020.6c7y-gq39 | Primary | In keeping with findable, accessible, interope...`
  - `10.1002_mp.14424 | https://doi.org/10.7937/k9/tcia.2015.pf0m9rei | Secondary | As methodologies to identify thoracic VOIs in ...`
  - `10.1002_nafm.10870 | https://doi.org/10.5066/p9gtumay | Primary | associated species. Doctoral dissertation. The...`

Cell 5 output: full text of `out_df["in_text_span"][0]` is printed - a long FAIR/TCIA paragraph about PleThora thoracic-cavity and pleural-effusion segmentations citing `https://doi.org/10.7937/tcia.2020.6c7y-gq39` as reference 51, followed by an anchor line and the next paragraph.

Cell 6 output:
```
[OPEN] PDF loaded: /content/drive/MyDrive/Make_data_count_challenge/Data/train/PDF/10.1002_2017jc013030.pdf
[REFS] References page index: 16  (1-based 17)
[REFS] Resolved ref number for DOI 'https://doi.org/10.17882/49388': 5

=== Extraction Summary ===
Input DOI                 : https://doi.org/10.17882/49388
Resolved Reference Number : 5
Body pages with hits      : 0
Paragraphs extracted      : 0

--- End ---
```

Cell 8 output: pip install progress bars for `marker-pdf` and its dependencies (sizes including a 50.0/50.0 MB and a 10.8/10.8 MB wheel).

Cell 9 output (truncated by the harness): `Downloading manifest.json`, then downloads of `text_recognition` model assets (`vocab_math.json`, `special_tokens_map.json`, `processor_config.json`, `specials_dict.json`, `README.md`, `config.json`, `tokenizer_config.json`, `preprocessor_config.json`, `specials.json`, and `model.safetensors` (1.31 GB)). The final combined CSV print/display from `run_four_way_threaded` is not visible in the captured output.

Cells 1, 7 and 10 produce no output (cell 1 is fully commented; 7 and 10 are empty).

## Summary

This Colab notebook is an exploratory workbench for extracting in-text spans that cite dataset DOIs in the Make-Data-Count challenge PDFs, using three different PDF-parsing backends (PyMuPDF, pdfminer.six, PyPDF2) plus a separate Marker (`marker-pdf`) GPU pass. The main production cell is the "Combined extractor: Ref#-first, v3.21 fallback", which first tries to map a dataset DOI to its numeric reference and locate paragraphs citing that number, and otherwise falls back to a v3.21 Data-Availability / inline-DOI miner. On 213 PDFs / 718 label rows it found spans for 315 rows and wrote `in_text_spans_combined.csv`. A standalone single-PDF debugger (cell 6) exercises the ref-number resolution for `10.1002_2017jc013030` (resolves DOI `10.17882/49388` to ref #5 but extracts 0 paragraphs). A separate 4-way threaded Marker runner renders each PDF to markdown on GPU and then scans the markdown for normalized dataset DOIs, sharding article IDs across threads and writing per-shard CSVs that are merged into `marker_doi_matches_gpu.csv`.
