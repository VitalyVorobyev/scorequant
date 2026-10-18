import Link from "@docusaurus/Link";
import {useEffect, useMemo, useRef, useState} from "react";

import {portalData} from "../data/portal";
import {REFERENCE_BASE, siteUrl} from "../lib/site";

interface PagefindData {
  excerpt: string;
  meta: {title?: string};
  url: string;
}

interface PagefindHit {
  data: () => Promise<PagefindData>;
}

interface PagefindModule {
  search: (query: string) => Promise<{results: PagefindHit[]}>;
}

interface SearchResult {
  href: string;
  summary: string;
  title: string;
  type: string;
}

interface SearchDialogProps {
  onClose: () => void;
  open: boolean;
}

const routes = [
  ["Get started", "/get-started", "Install it and follow the first fit through to what it means."],
  ["Walkthroughs", "/walkthroughs", "One applied question followed end to end, with real numbers."]
] as const;

export function SearchDialog({open, onClose}: SearchDialogProps): React.JSX.Element | null {
  const [query, setQuery] = useState("");
  const [indexedResults, setIndexedResults] = useState<SearchResult[]>([]);
  const input = useRef<HTMLInputElement>(null);
  const searchSequence = useRef(0);

  useEffect(() => {
    if (open) {
      input.current?.focus();
      setQuery("");
    }
  }, [open]);

  useEffect(() => {
    const close = (event: KeyboardEvent): void => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);

  useEffect(() => {
    const normalized = query.trim();
    if (!open || normalized.length < 2) {
      setIndexedResults([]);
      return;
    }
    const sequence = ++searchSequence.current;
    void (async (): Promise<void> => {
      try {
        const moduleURL = siteUrl("pagefind/pagefind.js");
        const pagefind = await import(/* webpackIgnore: true */ moduleURL) as PagefindModule;
        const response = await pagefind.search(normalized);
        const data = await Promise.all(response.results.slice(0, 6).map(async (hit) => hit.data()));
        if (sequence === searchSequence.current) setIndexedResults(data.map((item) => ({
          title: item.meta.title ?? "ScoreQuant",
          href: item.url,
          summary: item.excerpt.replace(/<[^>]+>/g, ""),
          type: "Page"
        })));
      } catch {
        if (sequence === searchSequence.current) setIndexedResults([]);
      }
    })();
    return () => {searchSequence.current += 1;};
  }, [open, query]);

  const localResults = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    const routeResults = routes.map(([title, href, summary]) => ({title, href, summary, type: "Route"}));
    const apiResults = portalData.api.map((symbol) => ({
      title: symbol.name,
      href: `${REFERENCE_BASE}symbols/`,
      summary: symbol.summary,
      type: symbol.kind === "class" ? "Class" : "Function"
    }));
    const all = [...routeResults, ...apiResults];
    if (!normalized) return all.slice(0, 7);
    return all
      .filter((item) => `${item.title} ${item.summary}`.toLowerCase().includes(normalized))
      .slice(0, 8);
  }, [query]);
  const results = indexedResults.length > 0 ? indexedResults : localResults;

  if (!open) return null;
  return (
    <div className="search-overlay" role="presentation" onMouseDown={onClose}>
      <section
        className="search-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Search ScoreQuant"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <label className="search-dialog__input">
          <span aria-hidden="true">⌕</span>
          <input
            ref={input}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search concepts, tasks, and symbols"
          />
          <kbd>esc</kbd>
        </label>
        <div className="search-dialog__results" aria-live="polite">
          {results.map((item) =>
            // Class/Function results point at the separately built MkDocs reference
            // tree mounted outside the Docusaurus app -- reaching it always means
            // leaving this app, so it needs a real page load, not client-side routing.
            item.type === "Class" || item.type === "Function" ? (
              <a key={`${item.type}-${item.title}`} href={item.href} onClick={onClose}>
                <span className="search-result__type">{item.type}</span>
                <span><strong>{item.title}</strong><small>{item.summary}</small></span>
                <span aria-hidden="true">↗</span>
              </a>
            ) : (
              <Link key={`${item.type}-${item.title}`} to={item.href} onClick={onClose}>
                <span className="search-result__type">{item.type}</span>
                <span><strong>{item.title}</strong><small>{item.summary}</small></span>
                <span aria-hidden="true">↗</span>
              </Link>
            )
          )}
          {results.length === 0 && <p className="empty-state">No exact match. Try “Fisher”, “ratio”, or a symbol name.</p>}
        </div>
        <footer><span>Static search index: Pagefind after production build</span><span>↑↓ navigate · enter open</span></footer>
      </section>
    </div>
  );
}
