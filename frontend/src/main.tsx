import React from "react";
import ReactDOM from "react-dom/client";
import {
  createBrowserRouter,
  Navigate,
  RouterProvider,
} from "react-router-dom";
import { WorkspaceProvider } from "@/hooks/workspace";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Shell } from "@/components/shell";
import "./index.css";

const router = createBrowserRouter([
  {
    element: <Shell />,
    children: [
      { path: "/", element: <Navigate to="/ask" replace /> },
      {
        path: "/ask/:threadId?",
        lazy: async () => ({
          Component: (await import("@/pages/ask")).AskPage,
        }),
      },
      {
        path: "/sources/:sourceId?",
        lazy: async () => ({
          Component: (await import("@/pages/sources")).SourcesPage,
        }),
      },
      {
        path: "/record",
        lazy: async () => ({
          Component: (await import("@/pages/record")).RecordPage,
        }),
      },
      { path: "*", element: <Navigate to="/ask" replace /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <WorkspaceProvider>
      <TooltipProvider delayDuration={250}>
        <RouterProvider router={router} />
      </TooltipProvider>
    </WorkspaceProvider>
  </React.StrictMode>,
);
