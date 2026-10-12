/// <reference types="node" />
import { Blob as NodeBlob } from "node:buffer";
import { describe, expect, it } from "vitest";
import { WorkspaceStorage, storageError } from "./storage";
import { answerFor, MockKnowledgeClient } from "./chat";
import {
  finishRecording,
  initialRecording,
  MAX_FILE_SIZE,
  recordingSource,
  tickRecording,
  validateFile,
} from "./capture";
import { recordingTemplate, seedSources, suggestions } from "@/fixtures/seed";
import type { ChatThread } from "@/types";

describe("Evidence-grounded demo answers", () => {
  it.each(suggestions)(
    "maps every citation to an exact passage: $label",
    ({ query }) => {
      const sources = seedSources();
      const answer = answerFor(query, sources);
      expect(answer.citations).toHaveLength(2);
      for (const citation of answer.citations) {
        const source = sources.find((item) => item.id === citation.sourceId)!;
        expect(source).toBeDefined();
        if (citation.turnId)
          expect(citation.excerpt).toBe(
            source.turns?.find((t) => t.id === citation.turnId)?.text,
          );
        else expect(source.text).toContain(citation.excerpt);
        expect(answer.text).toContain(`[${citation.number}]`);
      }
    },
  );
  it("abstains on unrelated requests, including unsupported specifications for a known family", () => {
    for (const query of [
      "How much does a servo cost?",
      "What is the battery specification for i4L?",
      "How do I repair a furnace?",
      "What should I check before replacing FX5-PLC999?",
    ]) {
      expect(answerFor(query, seedSources()).citations).toEqual([]);
      expect(answerFor(query, seedSources()).text).toContain(
        "couldn't find enough information",
      );
    }
  });
  it("prefers a saved recording with the cited turn, never an incomplete recording", () => {
    let draft = { ...initialRecording(), phase: "recording" as const };
    for (let i = 0; i < 12; i++) draft = tickRecording(draft) as typeof draft;
    const saved = recordingSource(finishRecording(draft));
    const answer = answerFor(suggestions[0].query, [saved, ...seedSources()]);
    expect(answer.citations[0].sourceId).toBe(saved.id);
    saved.turns = saved.turns?.slice(0, 2);
    expect(
      answerFor(suggestions[0].query, [saved, ...seedSources()]).citations[0]
        .sourceId,
    ).toBe("conv_068_1");
  });
  it("reassembles the complete streamed answer and stops without a completion event", async () => {
    const client = new MockKnowledgeClient();
    const controller = new AbortController();
    let text = "";
    const events: string[] = [];
    for await (const event of client.streamAnswer(
      suggestions[0].query,
      seedSources(),
      controller.signal,
    )) {
      events.push(event.event);
      if (event.event === "answer.delta") text += event.text;
    }
    expect(text).toBe(answerFor(suggestions[0].query, seedSources()).text);
    expect(events.at(-1)).toBe("run.completed");
    controller.abort();
    const stopped: string[] = [];
    await expect(
      (async () => {
        for await (const event of client.streamAnswer(
          suggestions[0].query,
          seedSources(),
          controller.signal,
        ))
          stopped.push(event.event);
      })(),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(stopped).not.toContain("run.completed");
  });
});

describe("Recording and document boundaries", () => {
  it("does not advance paused sessions and never summarizes an incomplete transcript", () => {
    const first = tickRecording({ ...initialRecording(), phase: "recording" });
    const paused = { ...first, phase: "paused" as const };
    expect(tickRecording(paused)).toEqual(paused);
    expect(finishRecording(first).summary).toBe("");
    let complete = first;
    for (let i = 0; i < 40; i++) complete = tickRecording(complete);
    expect(complete.turns).toHaveLength(recordingTemplate.turns.length);
    expect(finishRecording(complete).summary).toBe(recordingTemplate.summary);
    expect(recordingSource(complete).status).toBe("draft");
  });
  it("validates allowed extensions, content and the 20 MB limit", () => {
    expect(validateFile(new File(["note"], "process.md"))).toBeUndefined();
    expect(validateFile(new File(["note"], "process.exe"))).toContain("PDF");
    expect(validateFile(new File([], "empty.txt"))).toContain("empty");
    expect(
      validateFile({ name: "huge.pdf", size: MAX_FILE_SIZE + 1 } as File),
    ).toContain("20 MB");
  });
});

describe("Browser persistence", () => {
  it("seeds once, retains edits/files and threads across reopen, and restores fixtures on reset", async () => {
    const name = `test-${crypto.randomUUID()}`;
    const first = new WorkspaceStorage(name);
    const data = await first.load();
    expect(data.sources).toHaveLength(6);
    const changed = {
      ...data.sources[0],
      title: "Reviewed source title",
      summary: "An expert's local note",
    };
    await first.saveSource(changed);
    // fake-indexeddb uses Node structuredClone, which cannot clone jsdom Blob wrappers.
    const file = new NodeBlob(["Expert note"], {
      type: "text/plain",
    }) as unknown as Blob;
    await first.saveSource({
      ...changed,
      id: "uploaded",
      origin: "upload",
      file,
    });
    const thread: ChatThread = {
      id: "chat-one",
      title: "Question",
      updatedAt: "2026-10-10",
      messages: [
        {
          id: "a",
          role: "assistant",
          status: "streaming",
          content: "Partial answer",
          citations: [],
        },
      ],
    };
    await first.saveThread(thread);
    await first.close();
    const second = new WorkspaceStorage(name);
    const restored = await second.load();
    expect(restored.sources).toHaveLength(7);
    expect(restored.sources.find((s) => s.id === changed.id)?.title).toBe(
      "Reviewed source title",
    );
    expect(restored.sources.find((s) => s.id === "uploaded")?.file?.size).toBe(
      file.size,
    );
    expect(restored.threads[0].messages[0].status).toBe("stopped");
    const reset = await second.reset();
    expect(reset.sources).toHaveLength(6);
    expect(reset.threads).toHaveLength(0);
    await second.close();
  });
  it("reports storage quota errors as unsaved changes", () => {
    expect(
      storageError(new DOMException("full", "QuotaExceededError")),
    ).toContain("not saved");
    expect(storageError(new Error("denied"))).toContain("try again");
  });
});
