import { ArrowLeft, ArrowUpRight, BookOpen, Cpu } from "lucide-react";
import type { Citation, Source } from "@/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { SourceIcon, SourcePreview } from "./source-preview";

export function EvidencePanel({
  citations,
  sources,
  active,
  onSelect,
  onBack,
}: {
  citations: Citation[];
  sources: Source[];
  active?: Citation;
  onSelect(citation: Citation): void;
  onBack(): void;
}) {
  const selected =
    active && sources.find((source) => source.id === active.sourceId);
  return (
    <div className="evidence-panel">
      <div className="evidence-heading">
        <BookOpen size={17} />
        <h2>Sources</h2>
        <span className="count-pill">{citations.length}</span>
      </div>
      {!citations.length ? (
        <div className="evidence-empty">
          <span className="empty-icon">
            <BookOpen size={23} strokeWidth={1.5} />
          </span>
          <h3>See where answers come from</h3>
          <p>
            Documents and expert conversations will appear here when you ask a
            question.
          </p>
          <div className="evidence-key">
            <span>
              <SourceIcon kind="document" size={14} />
              Documents
            </span>
            <span>
              <SourceIcon kind="conversation" size={14} />
              Conversations
            </span>
          </div>
        </div>
      ) : selected && active ? (
        <div className="evidence-detail">
          <Button variant="ghost" size="sm" onClick={onBack}>
            <ArrowLeft size={14} />
            All sources
          </Button>
          <div className="source-detail-heading">
            <div className="source-kind">
              <SourceIcon kind={selected.kind} size={14} />
              {selected.kind === "document" ? "Document" : "Conversation"}
              <Badge variant="secondary">
                {selected.status === "draft" ? "Draft" : "Reference"}
              </Badge>
            </div>
            <h3>{selected.title}</h3>
            <div className="source-scope">
              <Cpu size={13} />
              {selected.machines.map((m) => m.name).join(", ") ||
                "Machine not identified"}
            </div>
          </div>
          <SourcePreview source={selected} citation={active} compact />
        </div>
      ) : (
        <div className="evidence-list">
          <p className="evidence-caption">Referenced in this answer</p>
          {citations.map((citation) => {
            const source = sources.find(
              (item) => item.id === citation.sourceId,
            );
            if (!source)
              return (
                <p key={citation.id} className="inline-error">
                  The cited source is unavailable.
                </p>
              );
            return (
              <button
                key={citation.id}
                className="evidence-card"
                onClick={() => onSelect(citation)}
                aria-label={`Open source ${citation.number}: ${source.title}`}
              >
                <div className="evidence-card-top">
                  <span className="citation-index">{citation.number}</span>
                  <span className="source-kind">
                    <SourceIcon kind={source.kind} size={13} />
                    {source.kind === "document" ? "Document" : "Conversation"}
                  </span>
                  <ArrowUpRight size={15} />
                </div>
                <h3>{source.title}</h3>
                <div className="source-scope">
                  <Cpu size={12} />
                  {source.machines.map((m) => m.name).join(", ")}
                </div>
                <p className="evidence-quote">
                  “
                  {citation.excerpt.length > 215
                    ? `${citation.excerpt.slice(0, 215).trim()}…`
                    : citation.excerpt}
                  ”
                </p>
                <div className="evidence-card-bottom">
                  <span>
                    {citation.page
                      ? `Page ${citation.page}`
                      : `Turn ${citation.turnId}`}
                  </span>
                  <span>
                    {source.status === "draft" ? "Draft" : "Reference"} ·{" "}
                    {source.origin === "recording"
                      ? "Saved recording"
                      : "Demo source"}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
