import type { RecordingDraft, Source } from "@/types";
import { recordingTemplate } from "@/fixtures/seed";

export const MAX_FILE_SIZE = 20 * 1024 * 1024;
export const ACCEPTED_FILES = ".pdf,.txt,.md,.docx";
export function validateFile(file: File): string | undefined {
  if (!/\.(pdf|txt|md|docx)$/i.test(file.name))
    return "Choose a PDF, TXT, MD or DOCX file.";
  if (file.size > MAX_FILE_SIZE)
    return "This file exceeds 20 MB. Choose a smaller file.";
  if (!file.size) return "This file is empty. Choose a file with content.";
}

export async function prepareUpload(file: File): Promise<Source> {
  const error = validateFile(file);
  if (error) throw new Error(error);
  const isText = /\.(txt|md)$/i.test(file.name);
  return {
    id: crypto.randomUUID(),
    title: file.name.replace(/\.[^.]+$/, ""),
    kind: "document",
    status: "draft",
    origin: "upload",
    createdAt: new Date().toISOString(),
    machines: [],
    summary: "",
    file,
    fileName: file.name,
    fileType: /\.pdf$/i.test(file.name) ? "application/pdf" : file.type,
    fileSize: file.size,
    text: isText ? await file.text() : undefined,
  };
}

export function initialRecording(): RecordingDraft {
  return {
    phase: "idle",
    elapsed: 0,
    title: recordingTemplate.title,
    turns: [],
    machines: recordingTemplate.machines,
    summary: "",
  };
}
export function tickRecording(draft: RecordingDraft): RecordingDraft {
  if (draft.phase !== "recording") return draft;
  const elapsed = draft.elapsed + 1;
  // Compress the original dialogue for the demo; timestamps are demo offsets.
  const count = Math.min(
    recordingTemplate.turns.length,
    Math.ceil(elapsed / 2),
  );
  return {
    ...draft,
    elapsed,
    turns: recordingTemplate.turns
      .slice(0, count)
      .map((turn, i) => ({ ...turn, seconds: i * 2 })),
  };
}
export function finishRecording(draft: RecordingDraft): RecordingDraft {
  return {
    ...draft,
    phase: "review",
    summary:
      draft.turns.length === recordingTemplate.turns.length
        ? recordingTemplate.summary
        : "",
  };
}
export function recordingSource(draft: RecordingDraft): Source {
  return {
    id: crypto.randomUUID(),
    title: draft.title,
    kind: "conversation",
    origin: "recording",
    status: "draft",
    createdAt: new Date().toISOString(),
    machines: draft.machines,
    summary: draft.summary,
    turns: draft.turns,
    scenario: "fx5",
  };
}
export function timestamp(seconds: number) {
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}
