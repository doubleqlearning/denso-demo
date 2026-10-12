import { openDB, type DBSchema, type IDBPDatabase } from "idb";
import type { ChatThread, Source } from "@/types";
import { seedSources } from "@/fixtures/seed";

interface WorkspaceDB extends DBSchema {
  sources: { key: string; value: Source };
  threads: { key: string; value: ChatThread };
  settings: { key: string; value: boolean };
}

export class WorkspaceStorage {
  private database?: Promise<IDBPDatabase<WorkspaceDB>>;
  constructor(private readonly name = "know-how-workspace-v1") {}

  private open() {
    this.database ??= openDB<WorkspaceDB>(this.name, 1, {
      upgrade(db) {
        db.createObjectStore("sources", { keyPath: "id" });
        db.createObjectStore("threads", { keyPath: "id" });
        db.createObjectStore("settings");
      },
    });
    return this.database;
  }

  async load() {
    const db = await this.open();
    // Check and seed in one transaction, including across simultaneous tabs.
    const tx = db.transaction(["sources", "settings"], "readwrite");
    if (!(await tx.objectStore("settings").get("seeded"))) {
      for (const source of seedSources())
        await tx.objectStore("sources").put(source);
      await tx.objectStore("settings").put(true, "seeded");
    }
    await tx.done;
    const sources = await db.getAll("sources");
    const threads = (await db.getAll("threads")).map((thread) => ({
      ...thread,
      messages: thread.messages.map((message) =>
        message.status === "streaming"
          ? { ...message, status: "stopped" as const }
          : message,
      ),
    }));
    return { sources, threads };
  }

  async saveSource(source: Source) {
    await (await this.open()).put("sources", source);
  }
  async saveThread(thread: ChatThread) {
    await (await this.open()).put("threads", thread);
  }
  async reset() {
    const db = await this.open();
    const tx = db.transaction(["sources", "threads", "settings"], "readwrite");
    await Promise.all([
      tx.objectStore("sources").clear(),
      tx.objectStore("threads").clear(),
      tx.objectStore("settings").clear(),
    ]);
    await tx.done;
    return this.load();
  }
  async close() {
    if (this.database) (await this.database).close();
  }
}

export const storage = new WorkspaceStorage();
export function storageError(error: unknown) {
  return error instanceof DOMException && error.name === "QuotaExceededError"
    ? "Your browser storage is full. The changes were not saved."
    : "Could not save to this browser. Your changes are still on screen; please try again.";
}
