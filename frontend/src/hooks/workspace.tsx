import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { ChatThread, Source } from "@/types";
import { storage, storageError } from "@/services/storage";

interface Workspace {
  sources: Source[];
  threads: ChatThread[];
  loading: boolean;
  generation: number;
  error: string;
  clearError(): void;
  saveSource(source: Source): Promise<void>;
  updateThread(thread: ChatThread): void;
  saveThread(thread: ChatThread): Promise<void>;
  reset(): Promise<void>;
}
const Context = createContext<Workspace | null>(null);

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [sources, setSources] = useState<Source[]>([]);
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const epoch = useRef(0);
  const generation = epoch.current;
  useEffect(() => {
    let active = true;
    storage
      .load()
      .then((data) => {
        if (active) {
          setSources(data.sources);
          setThreads(data.threads);
          setLoading(false);
        }
      })
      .catch(() => {
        if (active) {
          setError(
            "Could not open browser storage. Enable site storage and reload to use this workspace.",
          );
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);
  const updateThread = (thread: ChatThread) => {
    if (generation === epoch.current)
      setThreads((current) => [
        thread,
        ...current.filter((item) => item.id !== thread.id),
      ]);
  };
  const guarded = async (action: () => Promise<void>) => {
    try {
      await action();
    } catch (cause) {
      setError(storageError(cause));
      throw cause;
    }
  };
  const value: Workspace = {
    sources,
    threads,
    loading,
    error,
    generation,
    clearError: () => setError(""),
    updateThread,
    saveSource: (source) =>
      guarded(async () => {
        await storage.saveSource(source);
        setSources((current) => [
          source,
          ...current.filter((item) => item.id !== source.id),
        ]);
      }),
    saveThread: (thread) =>
      guarded(async () => {
        if (generation !== epoch.current) return;
        await storage.saveThread(thread);
        updateThread(thread);
      }),
    reset: () =>
      guarded(async () => {
        epoch.current += 1;
        const data = await storage.reset();
        setSources(data.sources);
        setThreads(data.threads);
      }),
  };
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export function useWorkspace() {
  const context = useContext(Context);
  if (!context) throw new Error("WorkspaceProvider is required");
  return context;
}
