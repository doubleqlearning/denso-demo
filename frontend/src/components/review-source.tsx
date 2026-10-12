import { useState } from "react";
import { ArrowLeft, Check, Cpu, FileText, LoaderCircle } from "lucide-react";
import type { Source } from "@/types";
import { SourcePreview } from "./source-preview";
import { MachinePicker } from "./machine-picker";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

export function ReviewSource({
  source,
  onSave,
  onCancel,
  editing = false,
}: {
  source: Source;
  onSave(source: Source): Promise<void>;
  onCancel(): void;
  editing?: boolean;
}) {
  const [draft, setDraft] = useState(source);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  async function save() {
    if (!draft.title.trim()) {
      setError("Add a title before saving.");
      return;
    }
    setSaving(true);
    setError("");
    try {
      await onSave({
        ...draft,
        title: draft.title.trim(),
        summary: draft.summary.trim(),
      });
    } catch {
      setError("The draft was not saved. Please try again.");
    } finally {
      setSaving(false);
    }
  }
  return (
    <div className="review-page">
      <div className="review-heading">
        <Button variant="ghost" size="sm" onClick={onCancel}>
          <ArrowLeft size={16} />
          Back
        </Button>
        <div>
          <h1>{editing ? "Source details" : "Review your capture"}</h1>
          <p>
            {editing
              ? "Keep the source and its context together."
              : "Check the source, machine and summary before saving."}
          </p>
        </div>
        <Button onClick={() => void save()} disabled={saving}>
          {saving ? (
            <LoaderCircle className="animate-spin" size={16} />
          ) : (
            <Check size={16} />
          )}
          {editing ? "Save changes" : "Save draft"}
        </Button>
      </div>
      {error && (
        <p className="inline-error" role="alert">
          {error}
        </p>
      )}
      <div className="review-grid">
        <section className="review-source">
          <div className="section-label">
            <FileText size={17} />
            <h2>Source</h2>
            <span className="section-number">01</span>
          </div>
          <label htmlFor="source-title" className="field-label">
            Title
          </label>
          <Input
            id="source-title"
            value={draft.title}
            onChange={(e) => setDraft({ ...draft, title: e.target.value })}
          />
          <div className="review-source-body">
            <SourcePreview source={source} />
          </div>
        </section>
        <div className="review-context">
          <section>
            <div className="section-label">
              <Cpu size={17} />
              <h2>Machine</h2>
              <span className="section-number">02</span>
            </div>
            <p className="field-description">
              Which machine or family does this apply to?
            </p>
            <MachinePicker
              value={draft.machines}
              onChange={(value) => setDraft({ ...draft, machines: value })}
            />
          </section>
          <section>
            <div className="section-label">
              <FileText size={17} />
              <h2>Summary</h2>
              <span className="section-number">03</span>
            </div>
            <label htmlFor="source-summary" className="field-description">
              Key observations, actions and what happened next.
            </label>
            <Textarea
              id="source-summary"
              rows={9}
              placeholder="Add a short summary (optional)"
              value={draft.summary}
              onChange={(e) => setDraft({ ...draft, summary: e.target.value })}
            />
            {source.origin === "upload" && (
              <p className="field-note">
                Saved locally. Document extraction is not connected yet.
              </p>
            )}
            {source.kind === "conversation" && (
              <p className="field-note">
                This is a draft. Saving does not mark it as expert-approved.
              </p>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
