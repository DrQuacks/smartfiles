# SmartFiles Fullstack Workflow Commands

This document summarizes the common end-to-end commands for working on
both backend and frontend.

## One-Time Setup

Backend (from repo root):

```bash
cd /Users/Kellar/Develop/Projects/smartfiles/backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Frontend:

```bash
cd /Users/Kellar/Develop/Projects/smartfiles/frontend
npm install
```

## Typical Development Loop

1. **Start backend pieces (CLI or server):**

   - For indexing and search via CLI:

     ```bash
     cd /Users/Kellar/Develop/Projects/smartfiles/backend
     source .venv/bin/activate

     # Extract, index, and search
     smartfiles index <folder>
     smartfiles search "your query here"
     ```

   - For an HTTP API (future):

     ```bash
     cd /Users/Kellar/Develop/Projects/smartfiles/backend
     source .venv/bin/activate
     uvicorn smartfiles.server.api:app --reload
     ```

2. **Start frontend dev server:**

   ```bash
   cd /Users/Kellar/Develop/Projects/smartfiles/frontend
   npm run dev
   ```

3. **Iterate:**

   - Adjust backend ingestion/embeddings/search logic.
   - Refresh index as needed:

     ```bash
     # Debug parsing only
     smartfiles extract <folder>

     # Rebuild index from saved text
     smartfiles index-from-text <folder>

     # Full rebuild
     smartfiles index --recreate <folder>
     ```

4. **Run checks before commit:**

   Backend (future tests):

   ```bash
   # e.g. pytest, once tests exist
   ```

   Frontend:

   ```bash
   cd /Users/Kellar/Develop/Projects/smartfiles/frontend
   npm run lint
   npm run build
   ```

Use this file as the canonical place to append new fullstack workflows
as the project gains a web API and richer UI.
