import {defineConfig, devices} from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  retries: process.env.CI === undefined ? 0 : 1,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:4173/scorequant/",
    trace: "retain-on-failure"
  },
  webServer: {
    command: "pnpm exec docusaurus serve --host 127.0.0.1 --port 4173 --no-open",
    port: 4173,
    reuseExistingServer: process.env.CI === undefined,
    timeout: 120_000
  },
  projects: [
    {name: "desktop", use: {...devices["Desktop Chrome"]}},
    {name: "mobile", use: {...devices["Pixel 7"]}}
  ]
});
