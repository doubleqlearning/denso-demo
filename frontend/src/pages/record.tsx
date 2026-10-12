import { useEffect, useRef, useState } from "react";
import { useBlocker, useLocation, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  AudioLines,
  CircleCheck,
  Mic,
  Pause,
  Play,
  Square,
} from "lucide-react";
import { useWorkspace } from "@/hooks/workspace";
import {
  finishRecording,
  initialRecording,
  recordingSource,
  tickRecording,
  timestamp,
} from "@/services/capture";
import { recordingTemplate } from "@/fixtures/seed";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ReviewSource } from "@/components/review-source";
import { SourcePreview } from "@/components/source-preview";
import type { RecordingDraft, Source } from "@/types";

export function RecordPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const workspace = useWorkspace();
  const [draft, setDraft] = useState<RecordingDraft>(() => ({
    ...initialRecording(),
    phase: location.state?.autoStart
      ? ("recording" as const)
      : ("idle" as const),
  }));
  const [saved, setSaved] = useState(false);
  const [review, setReview] = useState<Source>();
  const scroll = useRef<HTMLDivElement>(null);
  const nearBottom = useRef(true);
  const unsaved = draft.phase !== "idle" && !saved;
  const blocker = useBlocker(unsaved);
  useEffect(() => {
    if (draft.phase !== "recording") return;
    const timer = setInterval(() => setDraft(tickRecording), 1000);
    return () => clearInterval(timer);
  }, [draft.phase]);
  useEffect(() => {
    if (!unsaved) return;
    const handler = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [unsaved]);
  useEffect(() => {
    if (nearBottom.current && scroll.current)
      scroll.current.scrollTop = scroll.current.scrollHeight;
  }, [draft.turns.length]);
  useEffect(() => {
    if (saved) void navigate("/sources");
  }, [saved, navigate]);

  function finish() {
    const finished = finishRecording(draft);
    setDraft(finished);
    setReview(recordingSource(finished));
  }
  async function save(source: Source) {
    await workspace.saveSource(source);
    setSaved(true);
  }

  return (
    <>
      {review ? (
        <ReviewSource
          source={review}
          onSave={save}
          onCancel={() => {
            setReview(undefined);
            setDraft({ ...draft, phase: "paused" });
          }}
        />
      ) : (
        <main className="record-page">
          <header className="record-header">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => void navigate("/sources")}
            >
              <ArrowLeft size={16} />
              Sources
            </Button>
            <span className="record-demo-tag">Demo recording</span>
          </header>
          <div className="record-heading">
            <span className="eyebrow">CAPTURE EXPERT EXPERIENCE</span>
            <h1>Record know-how</h1>
            <p>
              Capture a process, troubleshooting session, or expert discussion.
            </p>
          </div>
          {draft.phase === "idle" ? (
            <div className="record-idle">
              <span className="record-idle-icon">
                <Mic size={34} strokeWidth={1.5} />
              </span>
              <h2>Record an expert session</h2>
              <p>
                This demo streams a sample conversation.
                <br />
                No microphone access is needed.
              </p>
              <Button
                size="lg"
                onClick={() => setDraft({ ...draft, phase: "recording" })}
              >
                <Mic size={17} />
                Start recording
              </Button>
            </div>
          ) : (
            <>
              <div className="record-status">
                <div>
                  <span
                    className={
                      draft.phase === "recording" ? "record-dot" : "paused-dot"
                    }
                  />
                  <strong>
                    {draft.phase === "recording"
                      ? "Recording sample session"
                      : "Recording paused"}
                  </strong>
                  <span className="record-clock">
                    {timestamp(draft.elapsed)}
                  </span>
                </div>
                <span className="record-scope">
                  {draft.machines[0]?.name} · Line 8
                </span>
              </div>
              <section className="live-transcript">
                <div className="live-transcript-header">
                  <span>
                    <AudioLines size={17} />
                    Live transcript
                  </span>
                  <span>{draft.turns.length} turns</span>
                </div>
                <div
                  className="live-transcript-scroll"
                  ref={scroll}
                  onScroll={() => {
                    if (scroll.current)
                      nearBottom.current =
                        scroll.current.scrollHeight -
                          scroll.current.scrollTop -
                          scroll.current.clientHeight <
                        80;
                  }}
                >
                  <SourcePreview source={recordingSource(draft)} />
                  {!draft.turns.length && (
                    <div className="listening-placeholder">
                      <AudioLines size={20} />
                      <p>The sample conversation is about to begin.</p>
                    </div>
                  )}
                </div>
                <div className="record-controls">
                  <span>
                    {draft.turns.length === recordingTemplate.turns.length ? (
                      <>
                        <CircleCheck size={15} />
                        Sample transcript complete
                      </>
                    ) : (
                      <>
                        <AudioLines size={15} />
                        Transcript appears as the session progresses
                      </>
                    )}
                  </span>
                  <div>
                    <Button
                      variant="outline"
                      onClick={() =>
                        setDraft({
                          ...draft,
                          phase:
                            draft.phase === "paused" ? "recording" : "paused",
                        })
                      }
                    >
                      {draft.phase === "paused" ? (
                        <Play size={15} />
                      ) : (
                        <Pause size={15} />
                      )}
                      {draft.phase === "paused" ? "Resume" : "Pause"}
                    </Button>
                    <Button disabled={!draft.turns.length} onClick={finish}>
                      <Square size={13} fill="currentColor" />
                      Finish recording
                    </Button>
                  </div>
                </div>
              </section>
            </>
          )}
        </main>
      )}
      <Dialog
        open={blocker.state === "blocked"}
        onOpenChange={(open) => {
          if (!open && blocker.state === "blocked") blocker.reset();
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {review
                ? "You have an unsaved draft"
                : "Your recording is still open"}
            </DialogTitle>
            <DialogDescription>
              {review
                ? "Keep reviewing to save your source, or discard this capture."
                : "Finish recording to review and save it before leaving."}
            </DialogDescription>
          </DialogHeader>
          <div className="leave-actions">
            <Button
              variant="outline"
              onClick={() => {
                if (blocker.state === "blocked") blocker.reset();
              }}
            >
              {review ? "Keep reviewing" : "Keep recording"}
            </Button>
            {!review && (
              <Button
                disabled={!draft.turns.length}
                onClick={() => {
                  if (blocker.state === "blocked") blocker.reset();
                  finish();
                }}
              >
                Finish and review
              </Button>
            )}
            <Button
              variant="ghost"
              className="destructive-text"
              onClick={() => {
                if (blocker.state === "blocked") blocker.proceed();
              }}
            >
              Discard and leave
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
