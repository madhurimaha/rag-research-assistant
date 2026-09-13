import type { Citation, Document, Health, StreamDone, StreamMeta, TraceRow } from "./types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, init);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} on ${path}`);
  return res.json() as Promise<T>;
}

export const getHealth = () => json<Health>("/health");
export const getDocuments = () => json<Document[]>("/documents");
export const getExplain = (messageId: number) =>
  json<{ message_id: number; question: string; candidates: TraceRow[] }>(
    `/chat/${messageId}/explain`,
  );

export const deleteDocument = (id: number) =>
  fetch(`${API}/documents/${id}`, { method: "DELETE" });

export async function uploadDocument(file: File): Promise<Document> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${API}/documents/upload`, { method: "POST", body });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? "upload failed");
  }
  return res.json();
}

export const documentFileUrl = (id: number, page?: number) =>
  `${API}/documents/${id}/file${page ? `#page=${page}` : ""}`;

export interface AskHandlers {
  onMeta: (meta: StreamMeta) => void;
  onCitation: (citation: Citation) => void;
  onToken: (text: string) => void;
  onDone: (done: StreamDone) => void;
  onError: (message: string) => void;
}

/**
 * Stream an answer over SSE.
 *
 * Uses `fetch` + a manual parser rather than `EventSource`, because `EventSource` cannot issue
 * a POST and this request carries a JSON body. The parser splits on the blank line that
 * terminates an SSE frame and keeps any partial frame in the buffer for the next chunk.
 */
export async function ask(
  body: { question: string; conversation_id?: number | null },
  handlers: AskHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API}/chat/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok || !res.body) {
    handlers.onError(`Request failed: ${res.status} ${res.statusText}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? ""; // trailing partial frame

    for (const frame of frames) {
      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!data) continue;

      let parsed: unknown;
      try {
        parsed = JSON.parse(data);
      } catch {
        continue;
      }

      switch (event) {
        case "meta":
          handlers.onMeta(parsed as StreamMeta);
          break;
        case "citation":
          handlers.onCitation(parsed as Citation);
          break;
        case "token":
          handlers.onToken((parsed as { text: string }).text);
          break;
        case "done":
          handlers.onDone(parsed as StreamDone);
          break;
        case "error":
          handlers.onError((parsed as { message: string }).message);
          break;
      }
    }
  }
}
