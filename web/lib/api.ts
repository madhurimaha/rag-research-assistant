import { authHeaders } from "./session";
import type {
  Citation,
  Conversation,
  ConversationSummary,
  Document,
  Health,
  StreamDone,
  StreamMeta,
  TraceRow,
  User,
} from "./types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = authHeaders(init?.headers);
  const res = await fetch(`${API}${path}`, { ...init, headers });
  if (res.status === 401) {
    const err = new Error("unauthorized");
    err.name = "UnauthorizedError";
    throw err;
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof body.detail === "string" ? body.detail : res.statusText);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const signup = (email: string, password: string) =>
  json<{ token: string; user: User }>("/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

export const login = (email: string, password: string) =>
  json<{ token: string; user: User }>("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

export const getMe = () => json<User>("/auth/me");

export const logout = () => json<void>("/auth/logout", { method: "POST" });

export const getConversations = () => json<ConversationSummary[]>("/conversations");

export const getConversation = (id: number) => json<Conversation>(`/conversations/${id}`);

export const getHealth = () => json<Health>("/health");
export const getDocuments = () => json<Document[]>("/documents");
export const getExplain = (messageId: number) =>
  json<{ message_id: number; question: string; candidates: TraceRow[] }>(
    `/chat/${messageId}/explain`,
  );

export const deleteDocument = (id: number) =>
  fetch(`${API}/documents/${id}`, { method: "DELETE", headers: authHeaders() });

export async function uploadDocument(file: File): Promise<Document> {
  const body = new FormData();
  body.append("file", file);
  let res: Response;
  try {
    res = await fetch(`${API}/documents/upload`, {
      method: "POST",
      body,
      headers: authHeaders(),
    });
  } catch {
    throw new Error("Could not reach the API. Is it running on port 8000?");
  }
  if (!res.ok) {
    const bodyJson = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = bodyJson.detail;
    throw new Error(typeof detail === "string" ? detail : "Upload failed");
  }
  return res.json();
}

export const documentFileUrl = (id: number, page = 1) =>
  `${API}/documents/${id}/file#page=${page}&view=FitH&navpanes=0`;

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
    headers: authHeaders({ "Content-Type": "application/json" }),
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
