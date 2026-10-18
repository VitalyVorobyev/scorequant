import react from "@vitejs/plugin-react";
import {fileURLToPath} from "node:url";

import {defineConfig} from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: {
    // Docusaurus resolves this at build time; component tests get a plain stub.
    alias: {
      "@docusaurus/Link": fileURLToPath(new URL("./tests/stubs/Link.tsx", import.meta.url))
    }
  },
  test: {
    environment: "jsdom",
    exclude: ["tests/e2e/**", "node_modules/**"],
    globals: true,
    setupFiles: ["./tests/setup.ts"]
  }
});
