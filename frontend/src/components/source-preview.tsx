import { useEffect, useRef, useState } from "react";
import { Download, FileText, Mic } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation, Source } from "@/types";
import { timestamp } from "@/services/capture";
import { Button } from "@/components/ui/button";

export function SourceIcon({
  kind,
  size = 18,
}: {
  kind: Source["kind"];
  size?: number;
}) {
  return kind === "document" ? <FileText size={size} /> : <Mic size={size} />;
}

export function SourcePreview({
  source,
  citation,
  compact = false,
}: {
  source: Source;
  citation?: Citation;
  compact?: boolean;
}) {
  const [url, setUrl] = useState("");
  const highlighted = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!source.file) return;
    const objectUrl = URL.createObjectURL(source.file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [source.file]);
  useEffect(() => {
    if (citation?.turnId)
      highlighted.current?.scrollIntoView({
        block: "nearest",
        behavior: "instant",
      });
  }, [citation?.turnId, source.id]);

  if (source.kind === "conversation")
    return (
      <div className={`transcript ${compact ? "transcript-compact" : ""}`}>
        {source.turns?.map((turn) => (
          <div
            key={turn.id}
            ref={turn.id === citation?.turnId ? highlighted : undefined}
            className={`transcript-turn ${turn.id === citation?.turnId ? "is-cited" : ""}`}
            data-testid={`turn-${turn.id}`}
          >
            <div
              className={`speaker-avatar ${turn.speaker === "Expert" ? "expert" : ""}`}
            >
              {turn.speaker.charAt(0)}
            </div>
            <div className="turn-content">
              <div className="turn-meta">
                <strong>{turn.speaker}</strong>
                <span>{timestamp(turn.seconds)}</span>
              </div>
              <p>{turn.text}</p>
            </div>
          </div>
        ))}
        {!source.turns?.length && (
          <p className="muted">No transcript captured yet.</p>
        )}
      </div>
    );

  return (
    <div className="document-preview">
      <div className="document-meta">
        <span>
          <FileText size={14} />
          {source.fileName ?? "Manual excerpt"}
        </span>
        {source.pageStart && (
          <span>
            Page {source.pageStart}
            {source.pageEnd !== source.pageStart && `–${source.pageEnd}`}
          </span>
        )}
      </div>
      {source.fileType === "application/pdf" && url ? (
        <iframe
          title={`Preview of ${source.fileName}`}
          className="pdf-preview"
          src={`${url}${citation?.page ? `#page=${citation.page}` : ""}`}
        />
      ) : source.text ? (
        <div className={`document-text ${citation ? "is-cited" : ""}`}>
          {source.origin === "demo" ? (
            <Markdown
              remarkPlugins={[remarkGfm]}
              components={{ a: ({ children }) => <span>{children}</span> }}
            >
              {source.text}
            </Markdown>
          ) : (
            source.text
          )}
        </div>
      ) : (
        <div className="unavailable-preview">
          <FileText size={28} />
          <p>Preview is not available for this file.</p>
          <span>You can download the original document.</span>
        </div>
      )}
      {url && (
        <Button variant="outline" size="sm" asChild>
          <a href={url} download={source.fileName}>
            <Download size={14} />
            Download original
          </a>
        </Button>
      )}
    </div>
  );
}
