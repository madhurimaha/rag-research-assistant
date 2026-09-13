"use client";

import { useState } from "react";
import { LogoMark } from "@/components/icons";
import { login, signup } from "@/lib/api";
import { setSession } from "@/lib/session";
import type { User } from "@/lib/types";

export function AuthPage({ onSignedIn }: { onSignedIn: (user: User) => void }) {
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const result =
        mode === "signup" ? await signup(email.trim(), password) : await login(email.trim(), password);
      setSession(result.token, result.user);
      onSignedIn(result.user);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not sign in");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-rail px-4">
      <div className="w-full max-w-[400px] rounded-2xl bg-surface p-8 shadow-raised">
        <div className="flex items-center gap-2.5">
          <LogoMark />
          <div>
            <h1 className="text-[17px] font-semibold tracking-[-0.014em]">
              RAG Research Assistant
            </h1>
            <p className="mt-1 text-[13px] text-ink-muted">
              Sign in to keep your conversation history
            </p>
          </div>
        </div>

        <form onSubmit={submit} className="mt-6 space-y-3">
          <label className="block">
            <span className="text-[13px] font-medium text-ink-muted">Email</span>
            <input
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 w-full rounded-lg border border-border bg-sunken px-3 py-2 text-[14px] text-ink"
            />
          </label>
          <label className="block">
            <span className="text-[13px] font-medium text-ink-muted">Password</span>
            <input
              type="password"
              autoComplete={mode === "signup" ? "new-password" : "current-password"}
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 w-full rounded-lg border border-border bg-sunken px-3 py-2 text-[14px] text-ink"
            />
            {mode === "signup" && (
              <span className="mt-1 block text-[13px] text-ink-subtle">At least 8 characters</span>
            )}
          </label>

          {error && (
            <p role="alert" className="text-[13px] text-danger">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={busy}
            className="w-full rounded-lg bg-accent px-3 py-2.5 text-[14px] font-semibold text-white transition-colors hover:bg-accent-hover disabled:opacity-60"
          >
            {busy ? "Please wait…" : mode === "signup" ? "Create account" : "Sign in"}
          </button>
        </form>

        <p className="mt-4 text-center text-[13px] text-ink-muted">
          {mode === "signup" ? "Already have an account?" : "New here?"}{" "}
          <button
            type="button"
            className="font-semibold text-accent hover:underline"
            onClick={() => {
              setMode(mode === "signup" ? "signin" : "signup");
              setError(null);
            }}
          >
            {mode === "signup" ? "Sign in" : "Create an account"}
          </button>
        </p>
      </div>
    </main>
  );
}
