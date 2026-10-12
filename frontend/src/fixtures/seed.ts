import fixtures from "./public-fixtures.json";
import type { MachineScope, Source, TranscriptTurn } from "@/types";

export const machines: MachineScope[] = [
  {
    id: "FX5-PLC471",
    name: "FX5-PLC471",
    kind: "asset",
    family: "FX5",
    location: "Line 8 · Inspection cell E",
  },
  {
    id: "MR-JE-AX932",
    name: "MR-JE-AX932",
    kind: "asset",
    family: "MR-JE",
    location: "Line 8",
  },
  { id: "i4L-R934", name: "i4L-R934", kind: "asset", family: "i4L" },
  { id: "FX5", name: "MELSEC FX5", kind: "family", family: "FX5" },
  { id: "MR-JE", name: "MELSERVO-JE", kind: "family", family: "MR-JE" },
  { id: "i4L", name: "OMRON i4L", kind: "family", family: "i4L" },
];

const titles = [
  "Intermittent Ethernet loss at inspection cell E",
  "Servo motor not moving after a tool change",
  "Front panel access after a cell modification",
];
const summaries = [
  "For FX5-PLC471, investigate the cable and switch before replacing the PLC. The expert identifies a damaged RJ45 clip on Line 8 and recommends comparing the earliest Event History timestamp with the HMI log. The local keep-alive setting must be checked after a module swap; the repair outcome is not reported.",
  "For MR-JE-AX932 on Line 8, check mechanical binding before parameters. The source reports a freely turning ball screw and PA11 set to zero. The expert describes a temporary diagnostic change that must be reverted; no final repair outcome is recorded.",
  "For i4L-R934, the expert discusses emergency-stop access after the front panel was moved. A swing-arm arrangement is proposed and a local reach check is described. The modification and final acceptance are not reported as complete.",
];
const scenarios = ["fx5", "servo", "i4l"] as const;
const dates = [
  "2026-10-10T02:15:00Z",
  "2026-10-09T07:30:00Z",
  "2026-10-08T03:00:00Z",
];

export function fixtureTurns(index: number): TranscriptTurn[] {
  const conversation = fixtures.conversations[index];
  return conversation.turns.map((turn, position) => ({
    id: turn.turn_id,
    speaker: turn.speaker === "expert_01" ? "Expert" : "Technician",
    role:
      conversation.speakers
        .find((speaker) => speaker.id === turn.speaker)
        ?.role.replaceAll("_", " ") ?? "Engineer",
    text: turn.text,
    seconds: position * 24,
  }));
}

export function seedSources(): Source[] {
  const conversations: Source[] = fixtures.conversations.map(
    (conversation, i) => ({
      id: conversation.conversation_id,
      originalId: conversation.conversation_id,
      title: titles[i],
      kind: "conversation",
      status: "draft",
      origin: "demo",
      createdAt: dates[i],
      machines: [machines[i]],
      summary: summaries[i],
      turns: fixtureTurns(i),
      scenario: scenarios[i],
    }),
  );
  const documents: Source[] = fixtures.documents.map((doc, i) => ({
    id: doc.chunk_id,
    originalId: doc.chunk_id,
    title: [
      "FX5 communication manual",
      "MELSERVO-JE troubleshooting manual",
      "OMRON i4L user's manual",
    ][i],
    kind: "document",
    status: "reference",
    origin: "demo",
    createdAt: dates[2],
    machines: [machines[i + 3]],
    summary: [
      "Engineering-tool diagnostics and event history.",
      "Troubleshooting a servo motor that does not operate.",
      "Front panel installation and emergency-stop access.",
    ][i],
    text: doc.text,
    pageStart: doc.page_start,
    pageEnd: doc.page_end,
    scenario: scenarios[i],
  }));
  return [...conversations, ...documents];
}

export const recordingTemplate = {
  title: titles[0],
  turns: fixtureTurns(0),
  machines: [machines[0]],
  summary: summaries[0],
};

export const suggestions = [
  {
    label: "Communication dropout",
    query: "What should I check before replacing FX5-PLC471?",
    machine: "MELSEC FX5",
  },
  {
    label: "Servo troubleshooting",
    query: "Why is MR-JE-AX932 not moving with no alarm?",
    machine: "MELSERVO-JE",
  },
  {
    label: "Robot maintenance",
    query: "What should I review when relocating the i4L front panel?",
    machine: "OMRON i4L",
  },
];
