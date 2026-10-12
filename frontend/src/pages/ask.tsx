import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  ArrowDown,
  ArrowRight,
  ArrowUp,
  BookOpen,
  Check,
  Copy,
  Cpu,
  MessageSquare,
  Square,
  Wrench,
} from "lucide-react";
import Markdown from "react-markdown";
import type { ChatMessage, ChatThread, Citation } from "@/types";
import { useWorkspace } from "@/hooks/workspace";
import { createKnowledgeClient } from "@/services/chat";
import { suggestions } from "@/fixtures/seed";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { EvidencePanel } from "@/components/evidence-panel";
import { BrandMark } from "@/components/brand";

const client = createKnowledgeClient();

function AnswerContent({
  message,
  onCitation,
}: {
  message: ChatMessage;
  onCitation(number: number): void;
}) {
  return (
    <Markdown
      components={{
        a: ({ href, children }) =>
          href?.startsWith("#citation-") ? (
            <button
              type="button"
              className="inline-citation"
              aria-label={`Open source ${href.slice(10)}`}
              onClick={() => onCitation(Number(href.slice(10)))}
            >
              {children}
            </button>
          ) : (
            <span>{children}</span>
          ),
      }}
    >
      {message.content.replace(/\[(\d+)\]/g, "[$1](#citation-$1)")}
    </Markdown>
  );
}

export function AskPage() {
  const { threadId } = useParams();
  const navigate = useNavigate();
  const workspace = useWorkspace();
  const thread = workspace.threads.find((item) => item.id === threadId);
  const [query, setQuery] = useState("");
  const [sending, setSending] = useState(false);
  const [selectedMessage, setSelectedMessage] = useState("");
  const [activeCitation, setActiveCitation] = useState<Citation>();
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [copied, setCopied] = useState("");
  const [localError, setLocalError] = useState("");
  const [awayFromBottom, setAwayFromBottom] = useState(false);
  const scroll = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);
  const controller = useRef<AbortController | null>(null);
  const activeThread = useRef<string | undefined>(undefined);
  const nearBottom = useRef(true);
  const answers =
    thread?.messages.filter((message) => message.role === "assistant") ?? [];
  const selected =
    answers.find((answer) => answer.id === selectedMessage) ?? answers.at(-1);
  const citations = selected?.citations ?? [];

  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    controller.current?.abort();
  }, [workspace.generation]);
  useEffect(() => {
    if (controller.current && activeThread.current !== threadId)
      controller.current.abort();
    setSelectedMessage("");
    setActiveCitation(undefined);
    nearBottom.current = true;
  }, [threadId]);
  useEffect(() => {
    if (nearBottom.current && scroll.current)
      scroll.current.scrollTop = scroll.current.scrollHeight;
  }, [thread?.messages]);

  function openCitation(message: ChatMessage, number: number) {
    const citation = message.citations.find((item) => item.number === number);
    if (!citation) return;
    setSelectedMessage(message.id);
    setActiveCitation(citation);
    if (window.matchMedia("(max-width: 1023px)").matches) setSourcesOpen(true);
  }

  async function send(value = query) {
    if (!value.trim() || sending) return;
    setQuery("");
    setSending(true);
    setLocalError("");
    setActiveCitation(undefined);
    const abort = new AbortController();
    controller.current = abort;
    const question = value.trim();
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
      citations: [],
      status: "complete",
    };
    let answer: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      citations: [],
      status: "streaming",
    };
    let current: ChatThread = {
      id: thread?.id ?? crypto.randomUUID(),
      title: thread?.title ?? question,
      updatedAt: new Date().toISOString(),
      messages: [...(thread?.messages ?? []), userMessage, answer],
    };
    activeThread.current = current.id;
    setSelectedMessage(answer.id);
    nearBottom.current = true;
    workspace.updateThread(current);
    if (!thread) void navigate(`/ask/${current.id}`, { replace: true });
    const update = () => {
      current = {
        ...current,
        messages: current.messages.map((item) =>
          item.id === answer.id ? answer : item,
        ),
      };
      workspace.updateThread(current);
    };
    try {
      await workspace.saveThread(current);
      for await (const event of client.streamAnswer(
        question,
        workspace.sources,
        abort.signal,
      )) {
        if (event.event === "answer.delta")
          answer = { ...answer, content: answer.content + event.text };
        if (event.event === "answer.reset") answer = { ...answer, content: "" };
        if (event.event === "answer.citations")
          answer = { ...answer, citations: event.citations };
        if (event.event === "run.completed")
          answer = { ...answer, status: "complete" };
        update();
      }
    } catch (cause) {
      const stopped = abort.signal.aborted;
      answer = {
        ...answer,
        status: stopped ? "stopped" : "failed",
        content:
          answer.content ||
          (stopped
            ? "Response stopped."
            : "The response could not be completed. Please try again."),
      };
      update();
      if (!stopped)
        setLocalError(
          cause instanceof Error
            ? cause.message
            : "Could not complete this response.",
        );
    } finally {
      update();
      try {
        await workspace.saveThread(current);
      } catch {
        /* Provider displays persistence failures. */
      }
      if (controller.current === abort) {
        controller.current = null;
        setSending(false);
      }
      composer.current?.focus();
    }
  }

  async function copy(message: ChatMessage) {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(message.id);
      setTimeout(() => setCopied(""), 1500);
    } catch {
      setLocalError(
        "Could not copy the answer. Select the text and copy it manually.",
      );
    }
  }

  const panel = (
    <EvidencePanel
      citations={citations}
      sources={workspace.sources}
      active={activeCitation}
      onSelect={setActiveCitation}
      onBack={() => setActiveCitation(undefined)}
    />
  );
  return (
    <div className="ask-layout">
      <main className="chat-area">
        <header className="page-header">
          <div>
            <h1>Ask</h1>
            <span className="header-subtitle">
              Your team's knowledge, with sources.
            </span>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="mobile-evidence-button"
            onClick={() => setSourcesOpen(true)}
          >
            <BookOpen size={15} />
            Sources
            {citations.length > 0 && (
              <span className="count-pill">{citations.length}</span>
            )}
          </Button>
        </header>
        <div
          className="chat-scroll"
          ref={scroll}
          onScroll={() => {
            if (!scroll.current) return;
            const el = scroll.current;
            nearBottom.current =
              el.scrollHeight - el.scrollTop - el.clientHeight < 100;
            setAwayFromBottom(!nearBottom.current);
          }}
        >
          {!thread?.messages.length ? (
            <div className="chat-empty">
              <div className="chat-empty-mark">
                <BrandMark />
              </div>
              <span className="eyebrow">ASK YOUR KNOWLEDGE</span>
              <h2>What are you working on?</h2>
              <p>
                Find an answer in your documents
                <br className="desktop-break" /> and your team's experience.
              </p>
              <div className="suggestion-list">
                {suggestions.map((suggestion, i) => (
                  <button
                    key={suggestion.label}
                    onClick={() => void send(suggestion.query)}
                    disabled={workspace.loading}
                  >
                    <span className="suggestion-icon">
                      {i === 0 ? (
                        <Cpu size={19} />
                      ) : i === 1 ? (
                        <Wrench size={19} />
                      ) : (
                        <MessageSquare size={19} />
                      )}
                    </span>
                    <span>
                      <strong>{suggestion.label}</strong>
                      <span>{suggestion.query}</span>
                    </span>
                    <ArrowRight size={17} />
                  </button>
                ))}
              </div>
              <div className="sample-caption">
                <span className="small-dot" />
                Try a question from the demo knowledge base
              </div>
            </div>
          ) : (
            <div className="message-list">
              {thread.messages.map((message) =>
                message.role === "user" ? (
                  <div key={message.id} className="user-message">
                    <p>{message.content}</p>
                  </div>
                ) : (
                  <article
                    key={message.id}
                    className={`assistant-message ${selected?.id === message.id ? "selected-answer" : ""}`}
                  >
                    <div className="assistant-label">
                      <BrandMark small />
                      <span>Know-how</span>
                    </div>
                    <div className="answer-prose">
                      {message.content ? (
                        <AnswerContent
                          message={message}
                          onCitation={(number) => openCitation(message, number)}
                        />
                      ) : (
                        <div className="thinking-status" role="status">
                          <span />
                          <span />
                          <span />
                          Finding relevant sources
                        </div>
                      )}
                    </div>
                    {message.status === "streaming" && message.content && (
                      <span
                        className="stream-cursor"
                        aria-label="Generating response"
                      />
                    )}
                    {message.status !== "streaming" && (
                      <div className="answer-actions">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            setSelectedMessage(message.id);
                            setActiveCitation(undefined);
                            if (
                              window.matchMedia("(max-width: 1023px)").matches
                            )
                              setSourcesOpen(true);
                          }}
                        >
                          <BookOpen size={14} />
                          {message.citations.length} sources
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          aria-label="Copy answer"
                          onClick={() => void copy(message)}
                        >
                          {copied === message.id ? (
                            <Check size={14} />
                          ) : (
                            <Copy size={14} />
                          )}
                        </Button>
                        {message.status === "stopped" && (
                          <span className="field-note">Stopped</span>
                        )}
                      </div>
                    )}
                  </article>
                ),
              )}
            </div>
          )}
        </div>
        {awayFromBottom && (
          <Button
            variant="outline"
            size="icon"
            className="scroll-to-bottom"
            aria-label="Scroll to latest answer"
            onClick={() => {
              nearBottom.current = true;
              scroll.current?.scrollTo({
                top: scroll.current.scrollHeight,
                behavior: "smooth",
              });
            }}
          >
            <ArrowDown size={16} />
          </Button>
        )}
        <div className="composer-wrap">
          {localError && (
            <p role="alert" className="inline-error">
              {localError}
            </p>
          )}
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              void send();
            }}
          >
            <textarea
              ref={composer}
              rows={2}
              aria-label="Ask a question"
              placeholder="Ask about a machine, issue or process…"
              value={query}
              maxLength={20000}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  void send();
                }
              }}
            />
            <div className="composer-bottom">
              <span>
                <BookOpen size={13} />
                Demo knowledge base
              </span>
              {sending ? (
                <Button
                  type="button"
                  size="icon"
                  aria-label="Stop response"
                  onClick={() => controller.current?.abort()}
                >
                  <Square size={15} fill="currentColor" />
                </Button>
              ) : (
                <Button
                  type="submit"
                  size="icon"
                  aria-label="Send question"
                  disabled={!query.trim() || workspace.loading}
                >
                  <ArrowUp size={19} />
                </Button>
              )}
            </div>
          </form>
          <p className="composer-note">
            Answers link back to the original source.
          </p>
        </div>
      </main>
      <aside className="desktop-evidence">{panel}</aside>
      <Sheet open={sourcesOpen} onOpenChange={setSourcesOpen}>
        <SheetContent side="right" className="evidence-sheet">
          <SheetHeader className="sr-only">
            <SheetTitle>Answer sources</SheetTitle>
            <SheetDescription>
              Inspect the documents and conversations cited by the selected
              answer.
            </SheetDescription>
          </SheetHeader>
          {panel}
        </SheetContent>
      </Sheet>
    </div>
  );
}
