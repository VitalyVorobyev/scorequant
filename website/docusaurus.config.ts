import type {Config} from "@docusaurus/types";
import type {Options as ClassicOptions} from "@docusaurus/preset-classic";
import {BannerPlugin} from "webpack";
import {themes} from "prism-react-renderer";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

// Palenight with one token colour lifted. Its numeric/constant pink `#ff5874`
// measures 4.48:1 against the code background `#292d3e`, just under the WCAG AA
// 4.5:1 threshold, so every numeric literal in a fenced snippet is a serious
// axe violation. That went unnoticed while no portal page carried substantial
// code; the four walkthroughs added in S8 are full of numbers, and the e2e
// accessibility scan failed on 93 nodes. `#ff7b93` is the same hue at 5.53:1 --
// a real margin rather than a hairline pass. Patched here rather than in CSS
// because prism-react-renderer emits these colours as inline styles, which a
// stylesheet could only override with `!important`.
// The theme writes this colour as `rgb(255, 88, 116)`, not as the hex its
// documentation shows, so both spellings are matched.
// Two of Palenight's token colours are below the WCAG AA 4.5:1 threshold against
// its own `#292d3e` code background: the numeric/constant pink at 4.48:1 and,
// much worse, the comment grey at 2.84:1. Both are replaced with the same hue
// lifted far enough to clear the threshold with margin rather than by a
// hairline. The theme writes colours in `rgb()` form, not the hex its
// documentation shows, so both spellings are matched.
const ACCESSIBLE_TOKEN_COLOURS = new Map([
  ["#ff5874", "rgb(255, 123, 147)"],
  ["rgb(255, 88, 116)", "rgb(255, 123, 147)"],
  ["#697098", "rgb(144, 153, 196)"],
  ["rgb(105, 112, 152)", "rgb(144, 153, 196)"]
]);
const accessiblePalenight = {
  ...themes.palenight,
  styles: themes.palenight.styles.map((entry) => {
    const replacement =
      entry.style.color === undefined ? undefined : ACCESSIBLE_TOKEN_COLOURS.get(entry.style.color);
    return replacement === undefined ? entry : {...entry, style: {...entry.style, color: replacement}};
  })
};

// Night Owl for dark mode's fenced code (the four walkthroughs are the only
// pages with substantial code, so this is the theme the e2e accessibility
// scan exercises in dark mode). Only one of its token colours falls short of
// WCAG AA against its own `#011627` background: the comment grey at 3.87:1.
// Replaced with the same hue lifted to 5.44:1 -- a margin, not a hairline --
// the same style as `accessiblePalenight` above. Every other Night Owl token
// colour already clears 4.5:1 (the closest being `boolean` at 6.02:1).
const NIGHT_OWL_ACCESSIBLE_TOKEN_COLOURS = new Map([["rgb(99, 119, 119)", "rgb(127, 143, 143)"]]);
const accessibleNightOwl = {
  ...themes.nightOwl,
  styles: themes.nightOwl.styles.map((entry) => {
    const replacement =
      entry.style.color === undefined ? undefined : NIGHT_OWL_ACCESSIBLE_TOKEN_COLOURS.get(entry.style.color);
    return replacement === undefined ? entry : {...entry, style: {...entry.style, color: replacement}};
  })
};

const config: Config = {
  title: "ScoreQuant",
  tagline: "Information-optimal score-space quantization",
  favicon: "img/mark.svg",
  url: "https://vitalyvorobyev.github.io",
  baseUrl: "/scorequant/",
  organizationName: "VitalyVorobyev",
  projectName: "scorequant",
  // Every route must emit its own index.html. With `false`, a docs section emits
  // both `research.html` and a `research/` directory with no index.html, and which
  // one a static host serves for `/research` is host-dependent — `docusaurus serve`
  // resolves the directory and renders the 404 shell. Directory URLs also match the
  // MkDocs reference mounted at /docs/ beneath this site (ADR 0033), so the whole
  // domain reads one way.
  trailingSlash: true,
  // Matches the separator AppShell already uses for the pages it titles itself,
  // so blog routes — whose titles come from Docusaurus metadata — read the same.
  titleDelimiter: "·",
  onBrokenLinks: "throw",
  onBrokenAnchors: "throw",
  markdown: {hooks: {onBrokenMarkdownLinks: "throw"}},
  plugins: [
    [
      "@docusaurus/plugin-content-docs",
      {
        id: "walkthroughs",
        path: "walkthroughs",
        routeBasePath: "walkthroughs",
        // AppShell (src/theme/Layout) owns navigation; the stock Docusaurus
        // sidebar is not wanted. No editUrl: these pages have no upstream
        // source to edit against.
        sidebarPath: false,
        // A lesson writes its model and score down as equations, with the
        // reference point visible (ADR 0028). Only the walkthrough instance
        // renders TeX; KaTeX's stylesheet is listed with the site CSS below.
        remarkPlugins: [remarkMath],
        rehypePlugins: [rehypeKatex]
      }
    ],
    [
      "@docusaurus/plugin-content-docs",
      {
        id: "research",
        path: "research",
        routeBasePath: "research",
        sidebarPath: false
      }
    ],
    [
      "@docusaurus/plugin-content-docs",
      {
        // `/get-started` is a docs instance rather than an MDX page route so it
        // inherits the swizzled DocItem table of contents and the single-`<main>`
        // DocRoot layout the e2e suite asserts on every route.
        id: "getstarted",
        path: "get-started",
        routeBasePath: "get-started",
        sidebarPath: false
      }
    ],
    () => ({
      name: "runtime-boundary-warnings",
      configureWebpack: (_config, isServer) => ({
        // This import intentionally resolves only in the assembled static
        // artifact; keeping it dynamic is what prevents an eager runtime load.
        ignoreWarnings: [
          {module: /SearchDialog\.tsx$/, message: /request of a dependency is an expression/}
        ],
        // Docusaurus's own ChunkAssetPlugin taps additionalTreeRuntimeRequirements
        // and appends `__webpack_require__.gca = ...` to every chunk that has a
        // runtime. A web worker built through `new Worker(new URL(...))` is its
        // own entry, and once its module needs no webpack runtime helpers webpack
        // emits it with no `__webpack_require__` binding at all -- so the appended
        // assignment is the first statement to run and it throws
        // `__webpack_require__ is not defined`, killing the worker before Pyodide
        // is fetched. The lab then reports only "the local runtime could not
        // start". Nothing in the worker ever calls `gca`; only the assignment
        // runs, so giving it an object to assign onto is enough. Guarded with
        // `||` so that a future chunk which does carry a real runtime is
        // untouched. Remove once Docusaurus scopes that plugin to chunks it owns.
        ...(isServer ? {} : {plugins: [new BannerPlugin({
          banner: "var __webpack_require__ = __webpack_require__ || {};",
          raw: true,
          test: /lab-worker\./
        })]})
      })
    })
  ],
  presets: [
    [
      "classic",
      {
        docs: false,
        blog: false,
        pages: {},
        sitemap: {changefreq: "weekly", priority: 0.6},
        // Order matters for the cascade: tokens.css declares every semantic
        // token (--surface, --border, --accent, ...) that live-fit.css's
        // rules read, so it must load first; the rest follow in the order
        // their sections appear in the pre-split global.css, and
        // responsive.css must stay after every file whose rules its media
        // queries override at equal specificity. live-fit.css stays last.
        theme: {
          customCss: [
            // KaTeX first: its rules are scoped to `.katex` and the portal's
            // own files may size or colour the rendered math after it.
            "./node_modules/katex/dist/katex.min.css",
            "./src/css/tokens.css",
            "./src/css/base.css",
            "./src/css/shell.css",
            "./src/css/components.css",
            "./src/css/home.css",
            "./src/css/walkthroughs.css",
            "./src/css/instruments.css",
            "./src/css/responsive.css",
            "./src/css/prose.css",
            "./src/css/charts.css",
            "./src/css/live-fit.css"
          ]
        }
      } satisfies ClassicOptions
    ]
  ],
  themeConfig: {
    image: "img/social-card.svg",
    prism: {theme: accessiblePalenight, darkTheme: accessibleNightOwl},
    colorMode: {defaultMode: "light", disableSwitch: false, respectPrefersColorScheme: true},
    metadata: [
      {name: "theme-color", content: "#07152f"},
      {name: "description", content: "Learn, inspect, and run information-preserving score-space quantization."}
    ]
  }
};

export default config;
