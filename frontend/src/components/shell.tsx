import { useState } from "react";
import {
  Link,
  NavLink,
  Outlet,
  useLocation,
  useNavigate,
} from "react-router-dom";
import {
  BookOpen,
  ChevronDown,
  CircleHelp,
  Menu,
  MessageSquare,
  Mic,
  Plus,
  RotateCcw,
  X,
} from "lucide-react";
import { BrandMark } from "./brand";
import { useWorkspace } from "@/hooks/workspace";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export function Shell() {
  const workspace = useWorkspace();
  const location = useLocation();
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);
  const [resetOpen, setResetOpen] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [resetError, setResetError] = useState("");
  const isAsk = location.pathname.startsWith("/ask");
  async function reset() {
    setResetting(true);
    setResetError("");
    try {
      await workspace.reset();
      setResetOpen(false);
      void navigate("/ask");
    } catch {
      setResetError("Could not reset browser storage. Please try again.");
    } finally {
      setResetting(false);
    }
  }
  const sidebar = (
    <div className="sidebar-content">
      <Link to="/ask" className="brand" onClick={() => setMenuOpen(false)}>
        <BrandMark />
        <span>
          Know-how<span className="brand-caption">Knowledge workspace</span>
        </span>
      </Link>
      <Button
        variant="outline"
        className="new-chat-button"
        onClick={() => {
          void navigate("/ask");
          setMenuOpen(false);
        }}
      >
        <Plus size={17} />
        New chat
      </Button>
      <nav className="primary-nav" aria-label="Main navigation">
        <NavLink
          to="/ask"
          className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
          onClick={() => setMenuOpen(false)}
        >
          <MessageSquare size={17} />
          Ask
        </NavLink>
        <NavLink
          to="/sources"
          className={({ isActive }) =>
            `nav-item ${isActive || location.pathname === "/record" ? "active" : ""}`
          }
          onClick={() => setMenuOpen(false)}
        >
          <BookOpen size={17} />
          Sources
        </NavLink>
      </nav>
      <div className="sidebar-history">
        <span className="sidebar-label">RECENT CHATS</span>
        {!workspace.threads.length ? (
          <p className="history-empty">Your questions will appear here.</p>
        ) : (
          [...workspace.threads]
            .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
            .map((thread) => (
              <Link
                key={thread.id}
                className={`history-item ${location.pathname.endsWith(thread.id) ? "active" : ""}`}
                to={`/ask/${thread.id}`}
                title={thread.title}
                onClick={() => setMenuOpen(false)}
              >
                <MessageSquare size={14} />
                <span>{thread.title}</span>
              </Link>
            ))
        )}
      </div>
      <div className="sidebar-bottom">
        <Button
          variant="outline"
          className="sidebar-record"
          onClick={() => {
            void navigate("/record", { state: { autoStart: true } });
            setMenuOpen(false);
          }}
        >
          <Mic size={16} />
          Start recording
        </Button>
        <div className="workspace-footer">
          <span className="workspace-avatar">K</span>
          <div>
            <strong>Local workspace</strong>
            <span>No account needed</span>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Workspace options"
              >
                <ChevronDown size={15} />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent side="top" align="end">
              <DropdownMenuItem
                disabled={location.pathname === "/record"}
                onSelect={() => {
                  setResetOpen(true);
                  setResetError("");
                }}
              >
                <RotateCcw size={14} />
                Reset demo
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </div>
  );
  return (
    <div className="workspace-shell">
      <aside className="desktop-sidebar">{sidebar}</aside>
      <div className="workspace-main">
        <div className="workspace-topbar">
          <div className="topbar-left">
            <Button
              variant="ghost"
              size="icon"
              className="mobile-menu"
              aria-label="Open navigation"
              onClick={() => setMenuOpen(true)}
            >
              <Menu size={19} />
            </Button>
            <span className="workspace-breadcrumb">
              Workspace<span>/</span>
              <strong>
                {isAsk
                  ? "Ask"
                  : location.pathname === "/record"
                    ? "Record know-how"
                    : "Sources"}
              </strong>
            </span>
          </div>
          <Tooltip>
            <TooltipTrigger asChild>
              <button className="demo-indicator" aria-label="About demo mode">
                <Badge variant="outline">
                  <span className="small-dot" />
                  Demo
                </Badge>
                <CircleHelp size={13} />
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="max-w-72">
              Responses and recordings are simulated. Sources and chats are
              saved in this browser only.
            </TooltipContent>
          </Tooltip>
        </div>
        {workspace.error && (
          <div className="workspace-error" role="alert">
            <span>{workspace.error}</span>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Dismiss error"
              onClick={workspace.clearError}
            >
              <X size={15} />
            </Button>
          </div>
        )}
        <div className="workspace-outlet">
          <Outlet />
        </div>
      </div>
      <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
        <SheetContent side="left" className="navigation-sheet">
          <SheetHeader className="sr-only">
            <SheetTitle>Workspace navigation</SheetTitle>
            <SheetDescription>
              Open Ask, Sources or a recent conversation.
            </SheetDescription>
          </SheetHeader>
          {sidebar}
        </SheetContent>
      </Sheet>
      <Dialog
        open={resetOpen}
        onOpenChange={(open) => {
          if (!resetting) setResetOpen(open);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reset this demo workspace?</DialogTitle>
            <DialogDescription>
              Uploaded files, saved recordings and chats in this browser will be
              removed. The original demo sources will be restored.
            </DialogDescription>
          </DialogHeader>
          {resetError && (
            <p role="alert" className="inline-error">
              {resetError}
            </p>
          )}
          <div className="dialog-actions">
            <Button
              variant="outline"
              disabled={resetting}
              onClick={() => setResetOpen(false)}
            >
              Cancel
            </Button>
            <Button disabled={resetting} onClick={() => void reset()}>
              {resetting ? "Resetting…" : "Reset demo"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
