"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
  type RefObject,
} from "react";
import { AuthPage } from "@/components/AuthPage";
import { ChatPane } from "@/components/ChatPane";
import { DocumentLibrary } from "@/components/DocumentLibrary";
import { EvidencePane } from "@/components/EvidencePane";
import { ExplainDialog } from "@/components/ExplainDialog";
import { SourcesPane } from "@/components/SourcesPane";
import { IconAlert, IconMenu, IconUpload, LogoMark } from "@/components/icons";
import {
  ask,
  deleteDocument,
  getConversation,
  getConversations,
  getDocuments,
  getHealth,
  getMe,
  logout as apiLogout,
  uploadDocument,
} from "@/lib/api";
import { clearSession, getToken } from "@/lib/session";
import type {
  Citation,
  Conversation,
  ConversationSummary,
  Document,
  Health,
  Turn,
  User,
} from "@/lib/types";

function turnsFromConversation(convo: Conversation): Turn[] {
  const turns: Turn[] = [];
  let pending: string | null = null;
  for (const msg of convo.messages) {
    if (msg.role === "user") {
      pending = msg.content;
      continue;
    }
    if (msg.role === "assistant" && pending) {
      turns.push({
        id: String(msg.id),
        question: pending,
        answer: msg.content,
        citations: msg.citations,
        streaming: false,
        done: {
          message_id: msg.id,
          conversation_id: convo.id,
          abstained: msg.abstained,
          retrieval_only: false,
          model: "",
          latency_ms: msg.latency_ms ?? 0,
          prompt_tokens: 0,
          cited_chunk_ids: msg.citations.map((c) => c.chunk_id),
        },
      });
      pending = null;
    }
  }
  return turns;
}

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Keep keyboard and screen-reader interaction inside an open drawer. */
function useModalFocus(
  open: boolean,
  panelRef: RefObject<HTMLDivElement | null>,
  onClose: () => void,
  returnFocusRef?: RefObject<HTMLElement | null>,
) {
  useEffect(() => {
    if (!open || !panelRef.current) return;
    const panel = panelRef.current;
    const returnTarget = returnFocusRef?.current ?? (document.activeElement as HTMLElement | null);
    const initialTarget = panel.querySelector<HTMLElement>(FOCUSABLE) ?? panel;
    queueMicrotask(() => initialTarget.focus());

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;

      const focusable = Array.from(panel.querySelectorAll<HTMLElement>(FOCUSABLE));
      if (focusable.length === 0) {
        event.preventDefault();
        panel.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      queueMicrotask(() => returnTarget?.focus());
    };
  }, [open, onClose, panelRef, returnFocusRef]);
}

const WIDE_EVIDENCE_QUERY = "(min-width: 1200px)";

function subscribeToWideEvidenceLayout(onChange: () => void) {
  const media = window.matchMedia(WIDE_EVIDENCE_QUERY);
  media.addEventListener("change", onChange);
  return () => media.removeEventListener("change", onChange);
}

function getWideEvidenceLayout() {
  return window.matchMedia(WIDE_EVIDENCE_QUERY).matches;
}

function useWideEvidenceLayout() {
  return useSyncExternalStore(subscribeToWideEvidenceLayout, getWideEvidenceLayout, () => false);
}

export default function Home() {
  const [user, setUser] = useState<User | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [documentsReady, setDocumentsReady] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [navigationOpen, setNavigationOpen] = useState(false);
  const [workspaceView, setWorkspaceView] = useState<"chat" | "library">("chat");
  const [citation, setCitation] = useState<Citation | null>(null);
  const [activeDoc, setActiveDoc] = useState<Document | null>(null);
  const [explaining, setExplaining] = useState<Turn | null>(null);
  const navigationButtonRef = useRef<HTMLButtonElement>(null);
  const navigationPanelRef = useRef<HTMLDivElement>(null);
  const evidencePanelRef = useRef<HTMLDivElement>(null);
  const evidenceReturnFocusRef = useRef<HTMLElement | null>(null);
  const wideEvidenceLayout = useWideEvidenceLayout();

  const refreshConversations = useCallback(() => {
    getConversations().then(setConversations).catch(() => setConversations([]));
  }, []);

  const closeEvidence = useCallback(() => {
    setCitation(null);
    setActiveDoc(null);
  }, []);

  const evidenceOpen = citation !== null || activeDoc !== null;
  const evidenceModalOpen = evidenceOpen && !wideEvidenceLayout;
  const workspaceBlocked = evidenceModalOpen || navigationOpen || explaining !== null;
  const closeNavigation = useCallback(() => setNavigationOpen(false), []);

  useModalFocus(evidenceModalOpen, evidencePanelRef, closeEvidence, evidenceReturnFocusRef);
  useModalFocus(navigationOpen, navigationPanelRef, closeNavigation, navigationButtonRef);

  useEffect(() => {
    const token = getToken();
    if (!token) {
      queueMicrotask(() => setAuthReady(true));
      return;
    }
    getMe()
      .then((me) => {
        setUser(me);
        refreshConversations();
      })
      .catch(() => {
        clearSession();
        setUser(null);
      })
      .finally(() => setAuthReady(true));
  }, [refreshConversations]);

  const refreshDocuments = useCallback(async () => {
    const next = await getDocuments();
    setDocuments(next);
    setDocumentsReady(true);
    return next;
  }, []);

  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll() {
      try {
        const next = await getDocuments();
        if (cancelled) return;
        setDocuments(next);
        setDocumentsReady(true);
        setUploadError(null);
        if (next.some((doc) => !["ready", "failed"].includes(doc.status))) {
          timer = setTimeout(poll, 1500);
        }
      } catch {
        if (!cancelled) {
          setDocumentsReady(true);
          setUploadError("Could not load documents. Check that the API is running.");
        }
      }
    }

    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [user]);

  useEffect(() => {
    if (!user) return;
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, [user]);

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
            onDone: (done) => {
              patch(id, { done, streaming: false });
              refreshConversations();
            },
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
    [conversationId, patch, refreshConversations],
  );

  async function openConversation(id: number) {
    const convo = await getConversation(id);
    setConversationId(convo.id);
    setTurns(turnsFromConversation(convo));
    setCitation(null);
    setExplaining(null);
    setNavigationOpen(false);
    setWorkspaceView("chat");
  }

  function newChat() {
    setConversationId(null);
    setTurns([]);
    setCitation(null);
    setExplaining(null);
    setNavigationOpen(false);
    setWorkspaceView("chat");
  }

  async function handleUpload(files: FileList | null) {
    if (!files?.length) return;
    setUploadError(null);
    setUploading(true);
    try {
      for (const file of Array.from(files)) await uploadDocument(file);
      await refreshDocuments();
      getHealth().then(setHealth).catch(() => undefined);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleDeleteDocument(doc: Document) {
    const response = await deleteDocument(doc.id);
    if (!response.ok) {
      setUploadError("Could not remove that document.");
      return;
    }
    if (activeDoc?.id === doc.id) closeEvidence();
    await refreshDocuments();
  }

  async function signOut() {
    try {
      await apiLogout();
    } catch {
      /* session may already be gone */
    }
    clearSession();
    setUser(null);
    newChat();
    setConversations([]);
  }

  if (!authReady) {
    return <div className="min-h-screen bg-rail" aria-busy="true" />;
  }

  if (!user) {
    return (
      <AuthPage
        onSignedIn={(next) => {
          setUser(next);
          refreshConversations();
        }}
      />
    );
  }

  return (
    <main className="flex h-screen flex-col overflow-hidden">
      <header
        className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-border bg-surface px-4"
        inert={workspaceBlocked ? true : undefined}
      >
        <div className="flex items-center gap-2.5">
          <button
            ref={navigationButtonRef}
            type="button"
            onClick={() => setNavigationOpen(true)}
            className="rounded-md p-1.5 text-ink-muted hover:bg-rail sm:hidden"
          >
            <IconMenu className="h-5 w-5" />
            <span className="sr-only">Open documents and chats</span>
          </button>
          <LogoMark />
          <div className="leading-none">
            <h1 className="text-[15px] font-semibold tracking-[-0.011em]">
              RAG Research Assistant
            </h1>
            <p className="mt-1 hidden text-[13px] text-ink-subtle sm:block">
              Grounded answers from your documents
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {health ? (
            <>
              <span className="hidden rounded-full bg-rail px-2.5 py-1 text-[13px] text-ink-muted lg:inline">
                {health.documents} documents · {health.chunks} chunks
              </span>
              <span
                className="flex items-center gap-1.5 rounded-full bg-rail px-2.5 py-1 text-[13px] text-ink-muted"
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
            <span className="flex items-center gap-1.5 rounded-full bg-warning-soft px-2.5 py-1 text-[13px] text-danger">
              <IconAlert className="h-3.5 w-3.5" />
              API unreachable
            </span>
          )}
          <span className="hidden text-[13px] text-ink-subtle md:inline">{user.email}</span>
          <button
            type="button"
            onClick={signOut}
            className="rounded-full bg-rail px-2.5 py-1 text-[13px] text-ink-muted hover:bg-rail-hover"
          >
            Sign out
          </button>
        </div>
      </header>

      <div
        className="grid min-h-0 flex-1 grid-cols-1 grid-rows-[minmax(0,1fr)] sm:grid-cols-[248px_minmax(0,1fr)]"
        inert={workspaceBlocked ? true : undefined}
      >
        <div className="hidden min-h-0 sm:block">
          <SourcesPane
            documents={documents}
            conversations={conversations}
            activeConversationId={conversationId}
            activeDocumentId={citation?.document_id ?? activeDoc?.id ?? null}
            uploading={uploading}
            uploadError={uploadError}
            onUpload={(files) => void handleUpload(files)}
            onDeleteDocument={(doc) => void handleDeleteDocument(doc)}
            onNewChat={newChat}
            onOpenConversation={(id) => void openConversation(id)}
            onOpenLibrary={() => setWorkspaceView("library")}
            libraryOpen={workspaceView === "library"}
            uploadInputId="desktop-pdf-upload"
            onSelectDocument={(doc) => {
              evidenceReturnFocusRef.current = document.activeElement as HTMLElement | null;
              setActiveDoc(doc);
              setCitation(null);
            }}
          />
        </div>

        <div
          className={`grid min-h-0 grid-cols-1 grid-rows-[minmax(0,1fr)] ${
            evidenceOpen
              ? "min-[1200px]:grid-cols-[minmax(0,55fr)_minmax(420px,45fr)]"
              : ""
          }`}
        >
          {!documentsReady ? (
            <div className="flex items-center justify-center bg-surface text-[13px] text-ink-subtle" aria-busy="true">
              Loading your workspace…
            </div>
          ) : documents.length === 0 ? (
            <EmptyLibrary uploading={uploading} error={uploadError} onUpload={handleUpload} />
          ) : workspaceView === "library" ? (
            <DocumentLibrary
              documents={documents}
              uploading={uploading}
              uploadError={uploadError}
              onUpload={(files) => void handleUpload(files)}
              onOpenDocument={(doc) => {
                evidenceReturnFocusRef.current = document.activeElement as HTMLElement | null;
                setActiveDoc(doc);
                setCitation(null);
              }}
              onDeleteDocument={(doc) => void handleDeleteDocument(doc)}
              onStartChat={newChat}
            />
          ) : (
            <ChatPane
              turns={turns}
              health={health}
              busy={busy}
              canAsk={documents.some((doc) => doc.status === "ready")}
              onAsk={handleAsk}
              onCite={(c) => {
                evidenceReturnFocusRef.current = document.activeElement as HTMLElement | null;
                setCitation(c);
                setActiveDoc(null);
              }}
              onExplain={setExplaining}
            />
          )}

          {evidenceOpen && (
            <div className="hidden min-h-0 border-l border-border min-[1200px]:col-start-2 min-[1200px]:block">
              <EvidencePane citation={citation} document={activeDoc} onClose={closeEvidence} />
            </div>
          )}
        </div>
      </div>

      {evidenceModalOpen && (
        <div className="fixed inset-0 z-40 sm:left-[248px] min-[1200px]:hidden">
          <button
            type="button"
            tabIndex={-1}
            className="absolute inset-0 bg-[rgb(26_24_21/0.24)] backdrop-blur-[2px]"
            aria-hidden="true"
            onClick={closeEvidence}
          />
          <div
            ref={evidencePanelRef}
            role="dialog"
            aria-modal="true"
            aria-label="Source evidence"
            tabIndex={-1}
            className="absolute inset-y-0 right-0 flex w-[94%] max-w-[44rem] flex-col shadow-overlay"
          >
            <EvidencePane citation={citation} document={activeDoc} onClose={closeEvidence} />
          </div>
        </div>
      )}

      {navigationOpen && (
        <div className="fixed inset-0 z-50 sm:hidden">
          <button
            type="button"
            tabIndex={-1}
            className="absolute inset-0 bg-[rgb(26_24_21/0.2)]"
            aria-hidden="true"
            onClick={closeNavigation}
          />
          <div
            ref={navigationPanelRef}
            role="dialog"
            aria-modal="true"
            aria-label="Documents and chats"
            tabIndex={-1}
            className="absolute inset-y-0 left-0 w-[min(88vw,20rem)] shadow-overlay"
          >
            <SourcesPane
              documents={documents}
              conversations={conversations}
              activeConversationId={conversationId}
              activeDocumentId={citation?.document_id ?? activeDoc?.id ?? null}
              uploading={uploading}
              uploadError={uploadError}
              onUpload={(files) => void handleUpload(files)}
              onDeleteDocument={(doc) => void handleDeleteDocument(doc)}
              onNewChat={newChat}
              onOpenConversation={(id) => void openConversation(id)}
              onOpenLibrary={() => {
                setWorkspaceView("library");
                setNavigationOpen(false);
              }}
              libraryOpen={workspaceView === "library"}
              onClose={closeNavigation}
              uploadInputId="mobile-pdf-upload"
              onSelectDocument={(doc) => {
                setActiveDoc(doc);
                setCitation(null);
                setNavigationOpen(false);
              }}
            />
          </div>
        </div>
      )}

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

function EmptyLibrary({
  uploading,
  error,
  onUpload,
}: {
  uploading: boolean;
  error: string | null;
  onUpload: (files: FileList | null) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <section className="flex h-full items-center justify-center bg-surface px-6 py-10">
      <div className="w-full max-w-[34rem] text-center">
        <span className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-accent-soft text-accent">
          <IconUpload className="h-6 w-6" />
        </span>
        <h2 className="mt-5 text-[24px] font-semibold tracking-[-0.018em]">Add your research papers</h2>
        <p className="mx-auto mt-2 max-w-[48ch] text-[14px] leading-relaxed text-ink-muted">
          Upload text-based PDFs to build your searchable library. Once indexing finishes, you can ask questions and verify every answer against its source.
        </p>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
          className="mt-6 inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-3 text-[14px] font-medium text-white shadow-raised transition-colors hover:bg-accent-hover disabled:opacity-50"
        >
          <IconUpload className="h-4 w-4" />
          {uploading ? "Uploading…" : "Choose PDFs"}
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          multiple
          className="sr-only"
          onChange={(event) => {
            onUpload(event.target.files);
            event.target.value = "";
          }}
        />
        <p className="mt-3 text-[13px] text-ink-subtle">PDF files up to 30 MB · Multiple files supported</p>
        {error && <p role="alert" className="mt-3 text-[13px] text-danger">{error}</p>}
      </div>
    </section>
  );
}
