# Know-how workspace

A local UI for capturing expert experience and asking questions with inspectable
sources. React, TypeScript, Vite, Tailwind CSS and shadcn/ui. No account is needed.

## Run

Run from `frontend/` with Node.js 22.12+ (or another current Vite-compatible version):

```sh
npm ci
cp .env.example .env.local
npm run dev
```

Open `http://127.0.0.1:5173`. This prototype uses these public frontend settings:

```dotenv
VITE_DATA_MODE=mock
VITE_API_BASE_URL=
```

Only mock mode is implemented. An unsupported mode fails explicitly rather than
silently falling back to mock answers. No backend requests or microphone access
are made. `VITE_*` values are visible in the browser; never put API keys in them.

## Use

- **Ask:** start from a sample question, inspect inline citations and the right
  Sources panel, or start a new chat. On small screens, citations open a source
  sheet. Questions outside the supported fixture topics receive a missing-evidence
  response. Mock responses are authored from the public sources, not model output.
- **Sources:** search by source title or machine and filter documents/conversations.
  Open a source to edit its title, machine scope and summary.
- **Upload document:** select one PDF/TXT/MD/DOCX up to 20 MB, review Source,
  Machine and Summary, then Save draft. PDFs use the browser's native viewer;
  TXT/MD are shown as plain text; DOCX offers metadata and download only.
  The summary starts empty and no extraction or indexing is claimed. Uploaded
  documents are not added to the mock answer corpus.
- **Start recording:** enter Record know-how to stream a sample multi-speaker
  transcript. Pause/resume, finish, review the machine and editable summary, then
  save a draft. Leaving an open session prompts you to keep, finish or discard it.
  Finishing a partial transcript leaves its summary empty rather than inventing
  facts from turns not captured. Saved sample recordings with the required turn
  can be cited by subsequent FX5 questions.

Sources, file blobs and chats are saved in IndexedDB on this browser/origin.
Changing port/hostname creates a different local workspace; this is not shared or
cloud storage. The workspace menu offers Reset demo with confirmation. Reset is
unavailable while a recording is open. Drafts are never labelled expert-approved.

## Mock provenance and service boundary

`src/fixtures/public-fixtures.json` contains three unchanged conversations and three
manual chunks copied from `benchmark/public/` only:

- FX5: `conv_068_1`, `fx5_comm_p0805_c02`.
- Servo: `conv_057_1`, `melservo_je_p0069_c01`.
- i4L: `conv_065_1`, `omron_i4l_p0108_c01`.

Speaker display names and transcript offsets are presentation-only demo metadata.
Conversation know-how is synthetic. Source IDs, text, turns and PDF page references
are retained. No generation/gold files or answer labels are imported.

`src/services/` separates storage, capture and chat transport from components.
`KnowledgeClient` streams `answer.delta`, `answer.reset`, `answer.citations` and
`run.completed`, keeping a boundary for a future backend SSE adapter. Citation
locators refer to actual source IDs plus page/turn; the frontend does not invent
upload, recording or evidence-preview API endpoints.

## Verify

```sh
npm run typecheck
npm run lint
npm run test
npm run build
npx playwright install chromium
npm run test:e2e
```

If Google Chrome is already installed, use it without downloading Chromium:

```sh
PLAYWRIGHT_CHANNEL=chrome npm run test:e2e
```

Playwright starts a localhost dev server if needed. Tests cover chat streaming,
citations, local upload persistence, recording transitions and leaving protection,
missing evidence, reset during a response, and 1440/1024/390px layouts. Screenshots
and failure traces are saved under ignored `test-results/`.

Production output is `dist/`; run `npm run preview` to inspect it. Hosting needs an
SPA fallback to `index.html` for `/ask`, `/sources` and `/record` routes. There is no
SSR, live transcription, speaker diarization, extraction, human approval workflow
or verified backend integration in this version.
