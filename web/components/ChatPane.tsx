"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { AnswerBody } from "@/components/AnswerBody";
import { RetrievalProgress } from "@/components/RetrievalProgress";
import { SourceList } from "@/components/SourceList";
import { IconAlert, IconCheck, IconCopy, IconInsight, IconSend } from "@/components/icons";
import { sourcesUsedInAnswer } from "@/lib/citations";
import type { Citation, Health, Turn } from "@/lib/types";

/** Starter prompts, labelled with the behaviour each one exercises. */
const SUGGESTIONS = [
  {
    label: "Single-paper lookup",
    text: "What is the QA-CTS task and what model is proposed for it?",
  },
  {
    label: "Spans several papers",
    text: "Which datasets are used to evaluate offensive language identification?",
  },
  {
    label: "Comparison across papers",
    text: "How do these papers measure annotation quality in crowdsourcing?",
  },
];

/**
 * Per-answer diagnostics. Each term gets a `title` because "Candidates 50" is meaningless
 * without knowing it counts retrieved chunks before reranking.
 */
function MetaRow({ turn }: { turn: Turn }) {
  const { meta, done } = turn;
  if (!meta) return null;
  const retrievalMs = Object.values(meta.timings_ms).reduce((a, b) => a + b, 0);

  const items: { term: string; value: string; hint: string }[] = [
    {
      term: "Retrieval",
      value: `${Math.round(retrievalMs)}ms`,
      hint: "Time to embed the question, search, and rerank",
    },
    ...(done
      ? [
          {
            term: "Total",
            value: `${(done.latency_ms / 1000).toFixed(1)}s`,
            hint: "End to end, including generation",
          },
        ]
      : []),
    {
      term: "Context",
      value: `${meta.prompt_tokens} tok`,
      hint: "Tokens of source text sent to the model",
    },
    {
      term: "Candidates",
      value: String(meta.n_candidates),
      hint: "Chunks retrieved before reranking",
    },
    ...(meta.config.carried_forward > 0
      ? [
          {
            term: "Carried over",
            value: String(meta.config.carried_forward),
            hint: "Chunks reused from the previous answer because this looked like a follow-up",
          },
        ]
      : []),
  ];

  return (
    <dl className="flex flex-wrap items-center gap-x-3.5 gap-y-1 text-[13px] text-ink-subtle">
      {items.map((item) => (
        <div key={item.term} className="flex items-center gap-1" title={item.hint}>
          <dt>{item.term}</dt>
          <dd className="font-mono text-ink-muted">{item.value}</dd>
        </div>
      ))}
      <div className="flex items-center gap-1" title="Model that wrote this answer">
        <dt className="sr-only">Model</dt>
        <dd className="font-mono">{meta.model}</dd>
      </div>
    </dl>
  );
}

function TurnFooter({
  turn,
  onCite,
  onExplain,
}: {
  turn: Turn;
  onCite: (c: Citation) => void;
  onExplain: (t: Turn) => void;
}) {
  const cited = sourcesUsedInAnswer(turn.citations, turn.done?.cited_chunk_ids);
  if (cited.length === 0 && !turn.meta) return null;

  return (
    <div className="mt-5 rounded-xl bg-sunken px-4 py-3">
      <SourceList citations={cited} onCite={onCite} />
      {turn.meta && (
        <div
          className={`flex flex-wrap items-center justify-between gap-2 ${
            cited.length > 0 ? "mt-3 border-t border-border pt-3" : ""
          }`}
        >
          <MetaRow turn={turn} />
          <div className="flex items-center gap-1">
            {turn.done && (
              <>
                <CopyButton text={turn.answer} />
                <button
                  type="button"
                  onClick={() => onExplain(turn)}
                  className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[13px] font-medium text-accent transition-colors hover:bg-accent-soft"
                >
                  <IconInsight className="h-3.5 w-3.5" />
                  Why this answer?
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1600);
    return () => clearTimeout(timer);
  }, [copied]);

  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
        } catch {
          /* clipboard unavailable (insecure origin or denied permission) */
        }
      }}
      className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[13px] font-medium text-ink-muted transition-colors hover:bg-rail hover:text-ink"
    >
      {copied ? <IconCheck className="h-3.5 w-3.5" /> : <IconCopy className="h-3.5 w-3.5" />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

export function ChatPane({
  turns,
  health,
  busy,
  canAsk,
  onAsk,
  onCite,
  onExplain,
}: {
  turns: Turn[];
  health: Health | null;
  busy: boolean;
  canAsk: boolean;
  onAsk: (question: string) => void;
  onCite: (c: Citation) => void;
  onExplain: (turn: Turn) => void;
}) {
  const [value, setValue] = useState("");
  const conversationRef = useRef<HTMLDivElement>(null);
  const followStreamRef = useRef(true);
  const previousTurnCountRef = useRef(0);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const latestTurn = turns.at(-1);

  useLayoutEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.style.height = "auto";
    const nextHeight = Math.min(input.scrollHeight, 128);
    input.style.height = `${nextHeight}px`;
    input.style.overflowY = input.scrollHeight > 128 ? "auto" : "hidden";
  }, [value]);

  useLayoutEffect(() => {
    const conversation = conversationRef.current;
    const addedTurn = turns.length > previousTurnCountRef.current;
    previousTurnCountRef.current = turns.length;
    if (addedTurn) followStreamRef.current = true;
    if (!conversation || (!addedTurn && !followStreamRef.current)) return;

    // Move only the transcript's scrollbar. `scrollIntoView` also scrolls ancestor containers,
    // and restarting its smooth animation for every streamed token makes the workspace bounce.
    conversation.scrollTop = conversation.scrollHeight;
  }, [turns.length, latestTurn?.answer, latestTurn?.streaming]);

  function submit() {
    const question = value.trim();
    if (!question || busy || !canAsk) return;
    onAsk(question);
    setValue("");
    inputRef.current?.focus();
  }

  return (
    <div className="flex h-full flex-col bg-surface">
      {health && !health.generation_enabled && (
        <div
          role="status"
          className="flex items-start gap-2 border-b border-border bg-warning-soft px-6 py-2.5 text-[13px] text-warning"
        >
          <IconAlert className="mt-px h-4 w-4 shrink-0" />
          <p>
            <strong className="font-semibold">Retrieval-only mode.</strong> No language model is
            configured, so answers show the retrieved passages instead of written prose. Set{" "}
            <code className="font-mono">LLM_PROVIDER</code> in{" "}
            <code className="font-mono">.env</code> to enable generation.
          </p>
        </div>
      )}

      {/* role="log" announces new turns without reading every streamed token. aria-busy holds
          announcements until a turn finishes, which is what makes streaming bearable on a
          screen reader. */}
      <div
        ref={conversationRef}
        className="scroll-area flex-1 overflow-y-auto px-6"
        role="region"
        aria-label="Conversation"
        tabIndex={0}
        onScroll={(event) => {
          const region = event.currentTarget;
          const distanceFromBottom =
            region.scrollHeight - region.scrollTop - region.clientHeight;
          followStreamRef.current = distanceFromBottom < 80;
        }}
      >
        <div className="mx-auto max-w-[46rem]">
          <div role="log" aria-relevant="additions" aria-busy={busy}>
            {turns.map((turn, i) => (
              <article
                key={turn.id}
                aria-label={`Question: ${turn.question}`}
                className={`rise py-8 ${i > 0 ? "border-t border-border" : ""}`}
              >
                {/* The question leads the turn, so it carries the largest type on the page. */}
                <h3 className="text-[19px] font-semibold leading-[1.35] tracking-[-0.012em]">
                  {turn.question}
                </h3>

                <div className="mt-4">
                  {turn.error ? (
                    <p
                      role="alert"
                      className="flex items-start gap-2 rounded-lg bg-warning-soft px-3 py-2.5 text-[13px] text-danger"
                    >
                      <IconAlert className="mt-px h-4 w-4 shrink-0" />
                      {turn.error}
                    </p>
                  ) : (
                    <>
                      {turn.done?.abstained && (
                        <p className="mb-3 inline-flex items-center gap-1.5 rounded-md bg-warning-soft px-2 py-1 text-[13px] font-medium text-warning">
                          <IconAlert className="h-3.5 w-3.5" />
                          Not answerable from these documents
                        </p>
                      )}

                      {/* Until the first token lands, show retrieval progress rather than an
                          empty space. */}
                      {turn.answer === "" && turn.streaming ? (
                        <RetrievalProgress
                          meta={turn.meta}
                          documentCount={health?.documents ?? 0}
                        />
                      ) : (
                        <AnswerBody
                          text={turn.answer}
                          citations={turn.citations}
                          onCite={onCite}
                          streaming={turn.streaming}
                        />
                      )}

                      {/* Sources and diagnostics sit in a sunken block so they read as apparatus
                          attached to the answer rather than part of it. After `done`, only
                          passages the model cited — retrieval can send more context than that. */}
                      <TurnFooter turn={turn} onCite={onCite} onExplain={onExplain} />
                    </>
                  )}
                </div>
              </article>
            ))}
          </div>

          {turns.length === 0 && (
            <div className="py-10">
              <h2 className="text-[22px] font-semibold tracking-[-0.014em]">
                Ask your documents a question
              </h2>
              <p className="mt-2 max-w-[46ch] text-[14px] leading-relaxed text-ink-muted">
                Answers come only from the indexed papers, with a citation for every claim. If
                the documents don’t cover a question, the assistant says so rather than guessing.
              </p>
              <ul className="mt-6 space-y-2">
                {SUGGESTIONS.map((s) => (
                  <li key={s.text}>
                    <button
                      type="button"
                      onClick={() => onAsk(s.text)}
                      disabled={!canAsk}
                      className="group w-full rounded-xl bg-sunken px-4 py-3 text-left transition-colors hover:bg-rail disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <span className="block text-[12px] font-medium uppercase tracking-wide text-ink-subtle">
                        {s.label}
                      </span>
                      <span className="mt-1 block text-[14px] text-ink group-hover:text-accent">
                        {s.text}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      <div className="shrink-0 border-t border-border bg-surface px-6 py-3">
        <form
          className="mx-auto max-w-[46rem]"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <label htmlFor="ask" className="sr-only">
            Your question
          </label>
          {/* The input and its send button share one rounded container, so the composer reads as
              a single control rather than two adjacent boxes. */}
          <div className="flex items-end gap-2 rounded-xl bg-sunken p-2 transition-colors focus-within:bg-surface focus-within:shadow-raised">
            <textarea
              id="ask"
              ref={inputRef}
              rows={1}
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => {
                // Enter sends; Shift+Enter inserts a newline.
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
              placeholder={canAsk ? "Ask about the indexed papers…" : "Waiting for a document to finish indexing…"}
              disabled={!canAsk}
              aria-describedby="ask-hint"
              className="max-h-32 min-h-9 flex-1 resize-none overflow-y-hidden bg-transparent px-2 py-1.5 text-[14px] leading-relaxed outline-none placeholder:text-ink-subtle"
            />
            <button
              type="submit"
              disabled={busy || !canAsk || !value.trim()}
              className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-accent px-3.5 text-[13px] font-medium text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-35"
            >
              {busy ? "Thinking…" : "Ask"}
              {!busy && <IconSend className="h-4 w-4" />}
            </button>
          </div>
          <p id="ask-hint" className="mt-1.5 px-1 text-[13px] text-ink-subtle">
            Enter to send, Shift+Enter for a new line. Follow-ups reuse the previous answer’s
            sources.
          </p>
        </form>
      </div>
    </div>
  );
}
