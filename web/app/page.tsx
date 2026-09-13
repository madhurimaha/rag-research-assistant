"use client";

import { useCallback, useEffect, useState } from "react";
import { ChatPane } from "@/components/ChatPane";
import { EvidencePane } from "@/components/EvidencePane";
import { ExplainDialog } from "@/components/ExplainDialog";
import { SourcesPane } from "@/components/SourcesPane";
import { IconAlert, LogoMark } from "@/components/icons";
import { ask, getHealth } from "@/lib/api";
import type { Citation, Document, Health, Turn } from "@/lib/types";

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [citation, setCitation] = useState<Citation | null>(null);
  const [activeDoc, setActiveDoc] = useState<Document | null>(null);
  const [explaining, setExplaining] = useState<Turn | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const patch = useCallback((id: string, update: Partial<Turn>) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...update } : t)));
  }, []);

  const handleAsk = useCallback(
    async (question: string) => {
      const id = crypto.randomUUID();
      setTurns((prev) => [
        ...prev,
        { id, question, answer: "", citations: [], streaming: true },
      ]);
      setBusy(true);

      try {
        await ask(
          { question, conversation_id: conversationId },
          {
            onMeta: (meta) => {
              setConversationId(meta.conversation_id);
              patch(id, { meta });
            },
            onCitation: (c) =>
              setTurns((prev) =>
                prev.map((t) =>
                  t.id === id ? { ...t, citations: [...t.citations, c] } : t,
                ),
              ),
            onToken: (text) =>
              setTurns((prev) =>
                prev.map((t) => (t.id === id ? { ...t, answer: t.answer + text } : t)),
              ),
            onDone: (done) => patch(id, { done, streaming: false }),
            onError: (message) => patch(id, { error: message, streaming: false }),
          },
        );
      } catch (e) {
        patch(id, {
          error: e instanceof Error ? e.message : "Request failed",
          streaming: false,
        });
      } finally {
        setBusy(false);
        patch(id, { streaming: false });
      }
    },
    [conversationId, patch],
  );

  return (
    <main className="flex h-screen flex-col overflow-hidden">
      <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-border bg-surface px-4">
        <div className="flex items-center gap-2.5">
          <LogoMark />
          <div className="leading-none">
            <h1 className="text-[15px] font-semibold tracking-[-0.011em]">
              RAG Research Assistant
            </h1>
            <p className="mt-1 hidden text-[12px] text-ink-subtle sm:block">
              Grounded answers over a research-paper corpus
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {health ? (
            <>
              <span className="hidden rounded-full bg-rail px-2.5 py-1 text-[12px] text-ink-muted md:inline">
                {health.documents} documents · {health.chunks} chunks
              </span>
              <span
                className="flex items-center gap-1.5 rounded-full bg-rail px-2.5 py-1 text-[12px] text-ink-muted"
                title={
                  health.generation_enabled
                    ? `Generating with ${health.model}`
                    : "Retrieval-only: no language model configured"
                }
              >
                <span
                  aria-hidden="true"
                  className={`h-1.5 w-1.5 rounded-full ${
                    health.generation_enabled ? "bg-positive" : "bg-warning"
                  }`}
                />
                {health.generation_enabled ? health.model : "retrieval only"}
              </span>
            </>
          ) : (
            <span className="flex items-center gap-1.5 rounded-full bg-warning-soft px-2.5 py-1 text-[12px] text-danger">
              <IconAlert className="h-3.5 w-3.5" />
              API unreachable
            </span>
          )}
        </div>
      </header>

      {/* Three panes: sources, conversation, evidence. The rails carry a recessed tone and the
          conversation is white, so the boundaries read without any divider lines.
          `grid-rows-[minmax(0,1fr)]` keeps the row from growing to fit its content — without it
          the panes push each other past the viewport instead of scrolling internally. The
          evidence pane keeps its slot even when empty so opening a citation doesn't reflow the
          conversation. */}
      <div className="grid min-h-0 flex-1 grid-cols-[248px_1fr] grid-rows-[minmax(0,1fr)] lg:grid-cols-[268px_1fr_minmax(368px,34%)]">
        <SourcesPane
          activeDocumentId={citation?.document_id ?? activeDoc?.id ?? null}
          onSelectDocument={(doc) => {
            setActiveDoc(doc);
            setCitation(null);
          }}
        />

        <ChatPane
          turns={turns}
          health={health}
          busy={busy}
          onAsk={handleAsk}
          onCite={(c) => {
            setCitation(c);
            setActiveDoc(null);
          }}
          onExplain={setExplaining}
        />

        <div className="hidden min-h-0 lg:block">
          <EvidencePane
            citation={citation}
            document={activeDoc}
            onClose={() => {
              setCitation(null);
              setActiveDoc(null);
            }}
          />
        </div>
      </div>

      {explaining?.done && (
        <ExplainDialog
          messageId={explaining.done.message_id}
          config={explaining.meta?.config}
          onClose={() => setExplaining(null)}
        />
      )}
    </main>
  );
}
