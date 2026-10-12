export interface MachineScope {
  id: string;
  name: string;
  kind: "asset" | "family";
  family: string;
  location?: string;
}

export interface TranscriptTurn {
  id: number;
  speaker: string;
  role: string;
  text: string;
  seconds: number;
}

export interface Source {
  id: string;
  title: string;
  kind: "document" | "conversation";
  status: "reference" | "draft";
  origin: "demo" | "upload" | "recording";
  createdAt: string;
  machines: MachineScope[];
  summary: string;
  text?: string;
  turns?: TranscriptTurn[];
  pageStart?: number;
  pageEnd?: number;
  file?: Blob;
  fileName?: string;
  fileType?: string;
  fileSize?: number;
  scenario?: "fx5" | "servo" | "i4l";
  originalId?: string;
}

export interface Citation {
  id: string;
  number: number;
  sourceId: string;
  excerpt: string;
  page?: number;
  turnId?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  status: "streaming" | "complete" | "stopped" | "failed";
}

export interface ChatThread {
  id: string;
  title: string;
  updatedAt: string;
  messages: ChatMessage[];
}

// Keeps mock and future SSE transports behind the same UI boundary.
export type AnswerEvent =
  | { event: "answer.delta"; text: string }
  | { event: "answer.reset" }
  | { event: "answer.citations"; citations: Citation[] }
  | { event: "run.completed" };

export type RecordingPhase = "idle" | "recording" | "paused" | "review";

export interface RecordingDraft {
  phase: RecordingPhase;
  elapsed: number;
  title: string;
  turns: TranscriptTurn[];
  machines: MachineScope[];
  summary: string;
}
