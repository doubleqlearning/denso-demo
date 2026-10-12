import { useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  ArrowRight,
  Check,
  ChevronRight,
  FileUp,
  LoaderCircle,
  Mic,
  Plus,
  Search,
  Upload,
  X,
} from "lucide-react";
import type { Source } from "@/types";
import { useWorkspace } from "@/hooks/workspace";
import {
  ACCEPTED_FILES,
  prepareUpload,
  validateFile,
} from "@/services/capture";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ReviewSource } from "@/components/review-source";
import { SourceIcon } from "@/components/source-preview";

export function SourcesPage() {
  const workspace = useWorkspace();
  const navigate = useNavigate();
  const { sourceId } = useParams();
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File>();
  const [uploadError, setUploadError] = useState("");
  const [preparing, setPreparing] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [uploadDraft, setUploadDraft] = useState<Source>();
  const [notice, setNotice] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const existing =
    sourceId && workspace.sources.find((source) => source.id === sourceId);
  const visible = workspace.sources
    .filter(
      (source) =>
        (filter === "all" || source.kind === filter) &&
        `${source.title} ${source.machines.map((m) => `${m.name} ${m.family}`).join(" ")}`
          .toLowerCase()
          .includes(search.toLowerCase()),
    )
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));

  function pick(files: FileList | null) {
    setUploadError("");
    if (!files?.length) return;
    if (files.length !== 1) {
      setUploadError("Choose one document at a time.");
      return;
    }
    const error = validateFile(files[0]);
    if (error) {
      setUploadError(error);
      setSelectedFile(undefined);
      return;
    }
    setSelectedFile(files[0]);
  }
  async function prepare() {
    if (!selectedFile || preparing) return;
    setPreparing(true);
    setUploadError("");
    try {
      const source = await prepareUpload(selectedFile);
      setUploadDraft(source);
      setUploadOpen(false);
      setSelectedFile(undefined);
    } catch (error) {
      setUploadError(
        error instanceof Error
          ? error.message
          : "Could not preview this document.",
      );
    } finally {
      setPreparing(false);
    }
  }
  async function save(source: Source) {
    await workspace.saveSource(source);
    setUploadDraft(undefined);
    setNotice(
      existing
        ? "Changes saved to this browser."
        : "Draft saved to this browser.",
    );
    void navigate("/sources");
  }

  if (uploadDraft || existing)
    return (
      <ReviewSource
        key={uploadDraft?.id ?? (existing && existing.id)}
        source={uploadDraft ?? (existing as Source)}
        editing={!!existing}
        onSave={save}
        onCancel={() => {
          setUploadDraft(undefined);
          void navigate("/sources");
        }}
      />
    );
  if (sourceId && !workspace.loading)
    return (
      <div className="route-empty">
        <h1>Source not found</h1>
        <p>This source is no longer available in this browser.</p>
        <Button onClick={() => void navigate("/sources")}>
          Back to sources
        </Button>
      </div>
    );

  return (
    <main className="sources-page">
      <header className="sources-header">
        <div>
          <span className="eyebrow">YOUR KNOWLEDGE LIBRARY</span>
          <h1>Sources</h1>
          <p>Keep your documents and experience in one place.</p>
        </div>
        <div className="capture-actions">
          <Button
            variant="outline"
            onClick={() => {
              setUploadOpen(true);
              setUploadError("");
              setSelectedFile(undefined);
            }}
          >
            <Upload size={16} />
            Upload document
          </Button>
          <Button
            onClick={() =>
              void navigate("/record", { state: { autoStart: true } })
            }
          >
            <Mic size={16} />
            Start recording
          </Button>
        </div>
      </header>
      {notice && (
        <div className="save-notice" role="status">
          <Check size={15} />
          {notice}
          <button
            aria-label="Dismiss save notification"
            onClick={() => setNotice("")}
          >
            <X size={14} />
          </button>
        </div>
      )}
      <div className="library-toolbar">
        <div className="search-field">
          <Search size={16} />
          <Input
            aria-label="Search sources"
            placeholder="Search sources or machines"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <Select value={filter} onValueChange={setFilter}>
          <SelectTrigger aria-label="Filter source type">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All sources</SelectItem>
            <SelectItem value="document">Documents</SelectItem>
            <SelectItem value="conversation">Conversations</SelectItem>
          </SelectContent>
        </Select>
        <span className="library-count">
          {visible.length} {visible.length === 1 ? "source" : "sources"}
        </span>
      </div>
      <div className="source-table">
        <div className="source-table-head">
          <span>Source</span>
          <span>Machine</span>
          <span>Status</span>
          <span>Added</span>
          <span />
        </div>
        {visible.map((source) => (
          <button
            key={source.id}
            className="source-row"
            onClick={() => void navigate(`/sources/${source.id}`)}
          >
            <span className="source-name">
              <span className={`source-icon-box ${source.kind}`}>
                <SourceIcon kind={source.kind} />
              </span>
              <span>
                <strong>{source.title}</strong>
                <span>
                  {source.kind === "document" ? "Document" : "Conversation"} ·{" "}
                  {source.origin === "upload"
                    ? "Saved locally · Not indexed"
                    : source.origin === "recording"
                      ? "Saved demo recording"
                      : "Demo source"}
                </span>
                <span className="source-mobile-machine">
                  {source.machines.map((machine) => machine.name).join(", ") ||
                    "Not identified"}
                </span>
              </span>
            </span>
            <span className="source-machine">
              {source.machines.length
                ? source.machines.map((m) => m.name).join(", ")
                : "Not identified"}
            </span>
            <span>
              <Badge
                variant="secondary"
                className={
                  source.status === "draft" ? "badge-draft" : "badge-reference"
                }
              >
                {source.status === "draft" ? "Draft" : "Reference"}
              </Badge>
            </span>
            <span className="source-date">
              {new Intl.DateTimeFormat("en", {
                month: "short",
                day: "numeric",
              }).format(new Date(source.createdAt))}
            </span>
            <ChevronRight size={16} />
          </button>
        ))}
        {!visible.length && (
          <div className="library-empty">
            <Search size={26} />
            <h2>
              {workspace.loading ? "Loading sources…" : "No sources found"}
            </h2>
            <p>
              {search || filter !== "all"
                ? "Try another search or source type."
                : "Upload a document or record your first session."}
            </p>
            {(search || filter !== "all") && (
              <Button
                variant="outline"
                onClick={() => {
                  setSearch("");
                  setFilter("all");
                }}
              >
                Clear filters
              </Button>
            )}
          </div>
        )}
      </div>
      <div className="library-footer">
        <span className="small-dot" />
        <span>
          Stored in this browser. Demo sources are synthetic benchmark examples.
        </span>
      </div>

      <Dialog
        open={uploadOpen}
        onOpenChange={(open) => {
          if (!preparing) setUploadOpen(open);
        }}
      >
        <DialogContent className="upload-dialog">
          <DialogHeader>
            <DialogTitle>Upload a document</DialogTitle>
            <DialogDescription>
              Add an expert note, manual or troubleshooting document.
            </DialogDescription>
          </DialogHeader>
          <input
            ref={fileInput}
            className="sr-only"
            type="file"
            accept={ACCEPTED_FILES}
            aria-label="Choose document file"
            onChange={(e) => pick(e.target.files)}
          />
          <div
            className={`upload-dropzone ${dragging ? "dragging" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              pick(e.dataTransfer.files);
            }}
          >
            <span className="upload-symbol">
              <FileUp size={28} strokeWidth={1.5} />
            </span>
            <h3>
              {selectedFile ? selectedFile.name : "Drop your document here"}
            </h3>
            <p>
              {selectedFile
                ? `${(selectedFile.size / 1024).toFixed(1)} KB · Ready to review`
                : "PDF, TXT, MD or DOCX · Up to 20 MB"}
            </p>
            <Button
              variant="outline"
              onClick={() => fileInput.current?.click()}
            >
              <Plus size={15} />
              {selectedFile ? "Choose another file" : "Choose a file"}
            </Button>
          </div>
          {uploadError && (
            <p role="alert" className="inline-error">
              {uploadError}
            </p>
          )}
          <div className="dialog-actions">
            <Button
              variant="ghost"
              disabled={preparing}
              onClick={() => setUploadOpen(false)}
            >
              Cancel
            </Button>
            <Button
              disabled={!selectedFile || preparing}
              onClick={() => void prepare()}
            >
              {preparing ? (
                <LoaderCircle size={15} className="animate-spin" />
              ) : (
                <ArrowRight size={15} />
              )}
              Review document
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </main>
  );
}
