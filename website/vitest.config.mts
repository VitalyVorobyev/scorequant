import react from "@vitejs/plugin-react";
import {fileURLToPath} from "node:url";

import {defineConfig} from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: {
    // Docusaurus resolves these at build time; page tests get plain stubs.
    alias: {
      "@docusaurus/Link": fileURLToPath(new URL("./tests/stubs/Link.tsx", import.meta.url)),
      "@theme/Layout": fileURLToPath(new URL("./tests/stubs/Layout.tsx", import.meta.url)),
      "@docusaurus/router": fileURLToPath(new URL("./tests/stubs/router.ts", import.meta.url)),
      "@docusaurus/useBrokenLinks": fileURLToPath(new URL("./tests/stubs/useBrokenLinks.ts", import.meta.url))
    }
  },
  test: {
    environment: "jsdom",
    exclude: ["tests/e2e/**", "node_modules/**"],
    globals: true,
    setupFiles: ["./tests/setup.ts"]
  }
});
