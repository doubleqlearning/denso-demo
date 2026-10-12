import { test, expect } from "@playwright/test";

test("chat streams, sources open the exact turn, and history survives reload", async ({
  page,
}) => {
  await page.goto("/ask");
  await page.getByRole("button", { name: /Communication dropout/ }).click();
  await expect(
    page.getByRole("button", { name: "Stop response" }),
  ).toBeVisible();
  await expect(
    page.getByText(
      "The conversation records troubleshooting advice, not a confirmed repair outcome.",
    ),
  ).toBeVisible();
  await page.screenshot({
    path: "test-results/answer-desktop.png",
    fullPage: true,
  });
  await page.locator(".inline-citation").first().click();
  await expect(
    page.locator(".desktop-evidence [data-testid=turn-6]"),
  ).toHaveClass(/is-cited/);
  await expect(
    page.locator(".desktop-evidence [data-testid=turn-6]"),
  ).toContainText("RJ45");
  await page.reload();
  await expect(
    page.getByText(
      "The conversation records troubleshooting advice, not a confirmed repair outcome.",
    ),
  ).toBeVisible();
  await expect(page.locator(".history-item")).toHaveCount(1);
});

test("upload reviews the actual text, machines and optional summary before local save", async ({
  page,
}) => {
  await page.goto("/sources");
  await expect(page.locator(".source-row")).toHaveCount(6);
  await page.screenshot({
    path: "test-results/sources-desktop.png",
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Upload document", exact: true })
    .click();
  await page.getByLabel("Choose document file").setInputFiles({
    name: "shift-note.txt",
    mimeType: "text/plain",
    buffer: Buffer.from(
      "Observed an intermittent signal. Outcome still pending.",
    ),
  });
  await page.getByRole("button", { name: "Review document" }).click();
  await expect(
    page.getByRole("heading", { name: "Machine", exact: true }),
  ).toBeVisible();
  await page.screenshot({
    path: "test-results/upload-review-desktop.png",
    fullPage: true,
  });
  await expect(
    page.getByText("Observed an intermittent signal. Outcome still pending.", {
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("textbox", { name: /Key observations/ }),
  ).toHaveValue("");
  await page.getByRole("button", { name: "Not identified" }).click();
  await page.getByRole("menuitemcheckbox", { name: /MELSEC FX5/ }).click();
  await page.keyboard.press("Escape");
  await page
    .getByRole("textbox", { name: /Key observations/ })
    .fill("Pending diagnosis. No confirmed repair yet.");
  await page.getByRole("button", { name: "Save draft" }).click();
  await expect(page.getByRole("button", { name: /shift-note/ })).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: /shift-note/ }).click();
  await expect(
    page.getByRole("textbox", { name: /Key observations/ }),
  ).toHaveValue("Pending diagnosis. No confirmed repair yet.");
  await expect(
    page.getByRole("button", { name: "MELSEC FX5", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Observed an intermittent signal. Outcome still pending.", {
      exact: true,
    }),
  ).toBeVisible();
});

test("recording pauses, blocks accidental departure, and saves edited context", async ({
  page,
}) => {
  await page.goto("/sources");
  await page
    .locator(".capture-actions")
    .getByRole("button", { name: "Start recording" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Record know-how" }),
  ).toBeVisible();
  await expect(page.locator("[data-testid=turn-6]")).toBeVisible({
    timeout: 20000,
  });
  await page.screenshot({
    path: "test-results/recording-desktop.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  const turns = await page.locator(".transcript-turn").count();
  await page.waitForTimeout(1200);
  expect(await page.locator(".transcript-turn").count()).toBe(turns);
  await page.getByRole("button", { name: "Resume", exact: true }).click();
  await page
    .locator(".record-header")
    .getByRole("button", { name: "Sources", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Your recording is still open" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Finish and review" }).click();
  await page.screenshot({
    path: "test-results/recording-review-desktop.png",
    fullPage: true,
  });
  await page
    .getByRole("textbox", { name: "Title" })
    .fill("Inspection cell handover");
  await page.getByRole("button", { name: "Remove FX5-PLC471" }).click();
  await page
    .getByRole("textbox", { name: /Key observations/ })
    .fill("Check the damaged clip before replacing hardware.");
  await page.getByRole("button", { name: "Save draft" }).click();
  await expect(
    page.getByRole("button", { name: /Inspection cell handover/ }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Ask", exact: true }).click();
  await page.getByRole("button", { name: /Communication dropout/ }).click();
  await expect(
    page.getByRole("button", {
      name: /Open source 1: Inspection cell handover/,
    }),
  ).toBeVisible();
});

test("stop, unknown questions and reset do not leave a stuck or misleading answer", async ({
  page,
}) => {
  await page.goto("/ask");
  await page.getByRole("button", { name: /Servo troubleshooting/ }).click();
  await page.getByRole("button", { name: "Stop response" }).click();
  await expect(page.getByText("Stopped", { exact: true })).toBeVisible();
  await page
    .getByRole("textbox", { name: "Ask a question" })
    .fill("What is the battery specification for i4L?");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(
    page.getByText(/I couldn't find enough information/),
  ).toBeVisible();
  await page.getByRole("button", { name: "Workspace options" }).click();
  await page.getByRole("menuitem", { name: "Reset demo" }).click();
  await page.getByRole("button", { name: "Reset demo", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "What are you working on?" }),
  ).toBeVisible();
  await expect(page.locator(".history-item")).toHaveCount(0);
});

test("responsive layouts, keyboard interaction and source sheet", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  for (const width of [1440, 1024, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/ask");
    await expect(
      page.getByRole("heading", { name: "What are you working on?" }),
    ).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({
      path: `test-results/ask-${width}.png`,
      fullPage: true,
    });
  }
  await page
    .getByRole("textbox", { name: "Ask a question" })
    .fill("What should I review when relocating the i4L front panel?");
  await page.getByRole("textbox", { name: "Ask a question" }).press("Enter");
  await expect(
    page.getByText(
      "The conversation does not report a completed modification or final acceptance.",
    ),
  ).toBeVisible();
  await page.locator(".inline-citation").last().click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(
    page.getByRole("dialog").getByText("Page 108–109"),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toBeHidden();
  await page.getByRole("button", { name: "Open navigation" }).click();
  await page
    .getByRole("dialog")
    .getByRole("link", { name: "Sources", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Sources", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/sources-390.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("source filters, upload validation and DOCX fallback remain usable with a keyboard", async ({
  page,
}) => {
  await page.goto("/sources");
  await page.getByRole("combobox", { name: "Filter source type" }).click();
  await page.getByRole("option", { name: "Documents", exact: true }).click();
  await expect(page.locator(".source-row")).toHaveCount(3);
  await page.getByRole("textbox", { name: "Search sources" }).fill("MELSEC");
  await expect(page.locator(".source-row")).toHaveCount(1);
  await page
    .getByRole("button", { name: "Upload document", exact: true })
    .click();
  await page.getByLabel("Choose document file").setInputFiles({
    name: "notes.exe",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("not a document"),
  });
  await expect(page.getByRole("alert")).toHaveText(/Choose a PDF/);
  await expect(
    page.getByRole("button", { name: "Review document" }),
  ).toBeDisabled();
  for (let i = 0; i < 8; i++) await page.keyboard.press("Tab");
  expect(
    await page
      .getByRole("dialog")
      .evaluate((dialog) => dialog.contains(document.activeElement)),
  ).toBe(true);
  await page.getByLabel("Choose document file").setInputFiles({
    name: "handover.docx",
    mimeType:
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    buffer: Buffer.from("opaque file bytes for the preview fallback test"),
  });
  await page.getByRole("button", { name: "Review document" }).click();
  await expect(
    page.getByText("Preview is not available for this file."),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Download original" }),
  ).toHaveAttribute("href", /^blob:/);
  await expect(
    page.getByRole("textbox", { name: /Key observations/ }),
  ).toHaveValue("");
});

test("mobile recording and review do not overflow or require microphone permission", async ({
  page,
}) => {
  const permissions: string[] = [];
  await page.exposeFunction("requestedMicrophone", () =>
    permissions.push("microphone"),
  );
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = () => {
      void (
        window as unknown as { requestedMicrophone(): Promise<void> }
      ).requestedMicrophone();
      return Promise.reject(new Error("Unexpected microphone request"));
    };
  });
  await page.setViewportSize({ width: 390, height: 900 });
  await page.goto("/record");
  await page
    .getByRole("button", { name: "Start recording", exact: true })
    .click();
  await expect(page.locator("[data-testid=turn-1]")).toBeVisible();
  await page.getByRole("button", { name: "Finish recording" }).click();
  await expect(
    page.getByRole("heading", { name: "Review your capture" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/recording-review-390.png",
    fullPage: true,
  });
  expect(permissions).toEqual([]);
});
