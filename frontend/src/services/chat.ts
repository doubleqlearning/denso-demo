import type { AnswerEvent, Citation, Source } from "@/types";

export interface KnowledgeClient {
  streamAnswer(
    query: string,
    sources: Source[],
    signal: AbortSignal,
  ): AsyncIterable<AnswerEvent>;
}

const fallback =
  "I couldn't find enough information in the demo sources to answer this question. Try a question about FX5 communication loss, MR-JE servo troubleshooting, or i4L front panel installation. Uploaded documents are saved locally and are not indexed in this demo.";

function scenarioFor(query: string) {
  // Deliberately narrow fixture routing, not semantic search or a live model.
  if (
    /fx5|plc471|ethernet|communication|rj45/i.test(query) &&
    /replac|dropout|communication|link|ethernet|rj45|switch/i.test(query)
  )
    return "fx5";
  if (
    /mr-je|ax932|servo|pa11/i.test(query) &&
    /not mov|not rotat|won.?t|no alarm|torque|pa11|bind/i.test(query)
  )
    return "servo";
  if (
    /i4l|r934|front panel/i.test(query) &&
    /front.?panel|relocat|emergency|e.stop/i.test(query)
  )
    return "i4l";
}

export function answerFor(
  query: string,
  sources: Source[],
): { text: string; citations: Citation[] } {
  const scenario = scenarioFor(query);
  if (!scenario) return { text: fallback, citations: [] };
  const namedAssets =
    query.match(/(?:FX5-[A-Z]*\d+|MR-JE-[A-Z]*\d+|i4L-[A-Z]*\d+)/gi) ?? [];
  const supportedAsset = {
    fx5: "FX5-PLC471",
    servo: "MR-JE-AX932",
    i4l: "i4L-R934",
  }[scenario];
  if (
    namedAssets.some(
      (asset) => asset.toLowerCase() !== supportedAsset.toLowerCase(),
    )
  )
    return { text: fallback, citations: [] };
  // Use saved recordings only if the actual cited turn exists and is unchanged.
  const conversations = sources.filter(
    (s) => s.scenario === scenario && s.kind === "conversation",
  );
  const conversation =
    conversations.find(
      (s) =>
        s.origin === "recording" &&
        s.turns?.some((t) => t.id === (scenario === "fx5" ? 6 : 2)),
    ) ?? conversations.find((s) => s.origin === "demo");
  const document = sources.find(
    (s) =>
      s.scenario === scenario && s.kind === "document" && s.origin === "demo",
  );
  if (!conversation || !document) return { text: fallback, citations: [] };
  const turn = conversation.turns?.find(
    (t) => t.id === { fx5: 6, servo: 2, i4l: 8 }[scenario],
  );
  if (!turn) return { text: fallback, citations: [] };
  const documentExcerpt = {
    fx5: "Use the engineering tool to check the errors that have occurred and their history, and identify the cause(s) of the errors.",
    servo: 'Check if the torque limit value is "0".',
    i4l: "Use the dimensions below to install the Front Panel in a suitable location outside of the robot's workspace where it can be easily accessed for emergency stops.",
  }[scenario];
  if (!document.text?.includes(documentExcerpt))
    return { text: fallback, citations: [] };
  const citations: Citation[] = [
    {
      id: `${conversation.id}:turn:${turn.id}`,
      number: 1,
      sourceId: conversation.id,
      turnId: turn.id,
      excerpt: turn.text,
    },
    {
      id: `${document.id}:page:${document.pageStart}`,
      number: 2,
      sourceId: document.id,
      page: document.pageStart,
      excerpt: documentExcerpt,
    },
  ];
  const text = {
    fx5: "## Check the connection before replacing hardware\n\nFor **FX5-PLC471 on Line 8**, the expert identifies a damaged RJ45 clip at the inspection-cell switch port and recommends checking that connection before replacing the PLC. [1]\n\nThe FX5 manual describes **module diagnostics and event history** as tools for identifying errors, their causes and corrective actions. Use those records to investigate the dropout. [2]\n\nThe conversation records troubleshooting advice, not a confirmed repair outcome.",
    servo:
      "## Start with mechanical binding\n\nFor **MR-JE-AX932 on Line 8**, the expert's first check is mechanical binding. Parameters are investigated only if the ball screw moves freely. [1]\n\nThe manual lists **an incorrect torque limit, mechanical interference and an unreleased brake** among possible causes when the motor does not operate. A torque limit of zero is one of the checkpoints. [2]\n\nThese sources do not establish that the amplifier needs replacement.",
    i4l: "## Keep the front panel accessible\n\nIn the **i4L-R934** discussion, the expert corrects the assumption that the light curtain makes front panel access unimportant. The advice is to place the panel outside the workspace for emergency-stop access. [1]\n\nThe manual says to choose a suitable location **outside the robot's workspace**, accessible for emergency stops, and account for both the XSYSTEM and extension cable lengths. [2]\n\nThe conversation does not report a completed modification or final acceptance.",
  }[scenario];
  return { text, citations };
}

function delay(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Stopped", "AbortError"));
      return;
    }
    const abort = () => {
      clearTimeout(timer);
      reject(new DOMException("Stopped", "AbortError"));
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, ms);
    signal.addEventListener("abort", abort, { once: true });
  });
}

export class MockKnowledgeClient implements KnowledgeClient {
  async *streamAnswer(
    query: string,
    sources: Source[],
    signal: AbortSignal,
  ): AsyncIterable<AnswerEvent> {
    const answer = answerFor(query, sources);
    yield { event: "answer.citations", citations: answer.citations };
    await delay(240, signal);
    for (const text of answer.text.match(/.{1,24}(?:\s|$)|.{1,24}/gs) ?? []) {
      await delay(22, signal);
      yield { event: "answer.delta", text };
    }
    yield { event: "run.completed" };
  }
}

export function createKnowledgeClient(): KnowledgeClient {
  const mode = import.meta.env.VITE_DATA_MODE || "mock";
  if (mode !== "mock")
    throw new Error(
      "This frontend currently supports mock mode only. Set VITE_DATA_MODE=mock.",
    );
  return new MockKnowledgeClient();
}
