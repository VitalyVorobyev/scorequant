/**
 * Dark mode does not get to be assumed accessible just because it was
 * measured accessible once, by hand, when it was written. This test parses
 * the token declarations straight out of `src/css/tokens.css` — it never
 * hard-codes a hex value of its own — resolves every `var(--x)` alias chain
 * to a concrete color for both the light `:root` block and the dark
 * `:root[data-theme="dark"]` override, and checks a declared table of
 * foreground/background pairs against the WCAG 2 contrast formula in both
 * themes. If a future edit repoints a token and quietly drops a pair below
 * threshold, this fails and names the pair — see the file's own git history
 * for the "prove the test bites" run that broke `--ink-500` on purpose to
 * confirm that.
 *
 * Deliberately NOT in the table below: `--border` and `--border-strong`
 * against `--surface`. They measure 1.23:1 and 1.59:1 in light and 1.58:1
 * and 2.03:1 in dark — nowhere near the 3:1 WCAG floor for a UI component or
 * a meaningful graphic, which is exactly what they are not: both are
 * decorative hairlines (a card edge, a section rule), and WCAG's 3:1
 * non-text-contrast requirement does not apply to purely decorative
 * elements. The light-mode values are the ones the existing axe scan
 * (`tests/e2e/portal.spec.ts`) already accepts today. Recorded here so a
 * later reader does not "fix" these by brightening the hairlines — that
 * would be a cosmetic regression chasing a requirement that does not apply.
 */

import {readFileSync} from "node:fs";
import {dirname, resolve} from "node:path";
import {fileURLToPath} from "node:url";

import {describe, expect, it} from "vitest";

const websiteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const cssPath = resolve(websiteRoot, "src/css/tokens.css");

type Theme = "light" | "dark";

/** One token declaration block's custom properties, keyed without the `--`. */
function parseDeclarations(block: string): Map<string, string> {
  const tokens = new Map<string, string>();
  const declaration = /--([\w-]+)\s*:\s*([^;]+);/g;
  for (const match of block.matchAll(declaration)) {
    const [, name, rawValue] = match;
    if (name === undefined || rawValue === undefined) continue;
    tokens.set(name, rawValue.trim());
  }
  return tokens;
}

/** Resolve a token name to a concrete `#rrggbb`, walking `var(--x)` aliases. */
function resolveToken(name: string, tokens: Map<string, string>, seen: ReadonlySet<string> = new Set()): string {
  if (seen.has(name)) throw new Error(`alias cycle resolving --${name}`);
  const raw = tokens.get(name);
  if (raw === undefined) throw new Error(`--${name} is not declared in tokens.css`);
  const alias = /^var\(--([\w-]+)\)$/.exec(raw);
  if (alias) {
    const [, aliasName] = alias;
    if (aliasName === undefined) throw new Error(`could not parse the alias in --${name}: ${raw}`);
    return resolveToken(aliasName, tokens, new Set(seen).add(name));
  }
  if (!/^#[0-9a-fA-F]{6}$/.test(raw)) {
    throw new Error(`--${name} does not resolve to a 6-digit hex color, got: ${raw}`);
  }
  return raw;
}

function hexToRgb(hex: string): [number, number, number] {
  const int = Number.parseInt(hex.slice(1), 16);
  return [(int >> 16) & 0xff, (int >> 8) & 0xff, int & 0xff];
}

// sRGB relative luminance, per the WCAG 2 definition.
function srgbChannelToLinear(channel: number): number {
  const c = channel / 255;
  return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function relativeLuminance(hex: string): number {
  const [r, g, b] = hexToRgb(hex);
  return (
    0.2126 * srgbChannelToLinear(r) + 0.7152 * srgbChannelToLinear(g) + 0.0722 * srgbChannelToLinear(b)
  );
}

// (L1 + 0.05) / (L2 + 0.05), lighter over darker.
function contrastRatio(hexA: string, hexB: string): number {
  const lumA = relativeLuminance(hexA);
  const lumB = relativeLuminance(hexB);
  const lighter = Math.max(lumA, lumB);
  const darker = Math.min(lumA, lumB);
  return (lighter + 0.05) / (darker + 0.05);
}

const css = readFileSync(cssPath, "utf8");
const withoutComments = css.replace(/\/\*[\s\S]*?\*\//g, "");

// The bare `:root { ... }` block never matches the attribute-selector block
// below, since `:root[` is not `:root\s*{`.
const rootBlock = /:root\s*\{([\s\S]*?)\}/.exec(withoutComments);
if (!rootBlock?.[1]) throw new Error("could not find the :root token block in tokens.css");

const darkBlock = /:root\[data-theme=["']dark["']\]\s*\{([\s\S]*?)\}/.exec(withoutComments);
if (!darkBlock?.[1]) throw new Error('could not find the :root[data-theme="dark"] token block in tokens.css');

const lightTokens = parseDeclarations(rootBlock[1]);
const darkTokens = new Map(lightTokens);
for (const [name, value] of parseDeclarations(darkBlock[1])) darkTokens.set(name, value);

const tokensByTheme: Record<Theme, Map<string, string>> = {light: lightTokens, dark: darkTokens};

interface ContrastPair {
  fg: string;
  bg: string;
  threshold: number;
}

// 4.5:1 (body text) unless noted otherwise; none of these pairs are large
// text, so every threshold below is the body-text floor.
const PAIRS: ContrastPair[] = [
  {fg: "text", bg: "surface", threshold: 4.5},
  {fg: "text", bg: "surface-raised", threshold: 4.5},
  {fg: "text-muted", bg: "surface", threshold: 4.5},
  {fg: "text-muted", bg: "surface-raised", threshold: 4.5},
  {fg: "text-faint", bg: "surface", threshold: 4.5},
  {fg: "text-faint", bg: "surface-raised", threshold: 4.5},
  {fg: "accent", bg: "surface", threshold: 4.5},
  {fg: "accent", bg: "surface-raised", threshold: 4.5},
  // The home page's highlighted result row.
  {fg: "accent-strong", bg: "accent-quiet", threshold: 4.5},
  // The provenance note's link and body text.
  {fg: "accent-strong", bg: "tint-good", threshold: 4.5},
  {fg: "text-muted", bg: "tint-good", threshold: 4.5},
  // The active filter chip.
  {fg: "accent-strong", bg: "tint-accent", threshold: 4.5},
  // The primary button and its hover state.
  {fg: "on-accent", bg: "accent", threshold: 4.5},
  {fg: "on-accent", bg: "accent-strong", threshold: 4.5},
  // The error hint.
  {fg: "bad", bg: "tint-bad", threshold: 4.5}
];

const cases = (["light", "dark"] as const).flatMap((theme) => PAIRS.map((pair) => ({theme, ...pair})));

describe("dark mode contrast (parsed live from src/css/tokens.css)", () => {
  it.each(cases)("--$fg on --$bg meets $threshold:1 in $theme mode", ({theme, fg, bg, threshold}) => {
    const tokens = tokensByTheme[theme];
    const ratio = contrastRatio(resolveToken(fg, tokens), resolveToken(bg, tokens));
    expect(ratio, `--${fg} on --${bg} in ${theme} mode measured ${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(
      threshold
    );
  });
});
