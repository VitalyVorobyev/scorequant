import {useMemo, useState} from "react";

import {useRegisteredAnchors} from "../anchors";
import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps, Core, CoreClaim} from "../core";
import {EntityList} from "../Entity";

/** The three TeX macros the chapter section titles use, as plain characters. */
const MATH_CHARACTERS: Record<string, string> = {"\\le": "≤", "\\ge": "≥", "\\to": "→", "\\Phi": "Φ"};

/** A section title with its inline math read as text: `$K\le d$` becomes `K≤d`. */
function plainMath(title: string): string {
  return title.replace(/\$([^$]*)\$/g, (_match, inner: string) => inner.replace(/\\[A-Za-z]+/g, (macro) => MATH_CHARACTERS[macro] ?? macro.slice(1)));
}

function matches(claim: CoreClaim, query: string): boolean {
  if (query === "") return true;
  const needle = query.toLowerCase();
  return claim.title.toLowerCase().includes(needle) || claim.id.toLowerCase().includes(needle);
}

interface Group {
  key: string;
  label: string;
  sections: {ids: string[]; label: string | null; title: string}[];
  source?: string;
}

/** Chapters in registry order, then the themed open questions, then the rest. */
function groupsOf(core: Core): Group[] {
  const claims = Object.values(core.claims).filter((claim) => claim.kind !== "audit");
  const groups: Group[] = [];

  for (const chapter of core.chapters) {
    const sections = chapter.sections.map((section) => ({
      label: section.label,
      title: plainMath(section.title),
      ids: claims.filter((claim) => claim.chapter?.slug === chapter.slug && claim.chapter.section === section.label).map((claim) => claim.id)
    }));
    const loose = claims.filter((claim) => claim.chapter?.slug === chapter.slug && !chapter.sections.some((section) => section.label === claim.chapter?.section));
    if (loose.length > 0) sections.push({label: null, title: "Elsewhere in the chapter", ids: loose.map((claim) => claim.id)});
    if (sections.some((section) => section.ids.length > 0)) {
      groups.push({key: chapter.slug, label: chapter.label, source: chapter.url, sections: sections.filter((section) => section.ids.length > 0)});
    }
  }

  const open = claims.filter((claim) => claim.chapter === null && claim.kind === "question");
  const openSections = core.themes
    .map((theme) => ({
      label: null,
      title: theme.label,
      ids: open.filter((claim) => claim.theme === theme.slug).map((claim) => claim.id)
    }))
    .filter((section) => section.ids.length > 0);
  const unthemed = open.filter((claim) => !core.themes.some((theme) => theme.slug === claim.theme));
  if (unthemed.length > 0) openSections.push({label: null, title: "Not filed under a theme", ids: unthemed.map((claim) => claim.id)});
  if (openSections.length > 0) groups.push({key: "open", label: "Open questions", sections: openSections});

  const rest = claims.filter((claim) => claim.chapter === null && claim.kind !== "question");
  if (rest.length > 0) {
    groups.push({key: "unwritten", label: "Not yet written into a chapter", sections: [{label: null, title: "", ids: rest.map((claim) => claim.id)}]});
  }
  return groups;
}

/**
 * `/research/claims/`: every statement in the record, in the order of the
 * chapters that derive them, then the open questions by theme, then the
 * verification records. The filter narrows the lists in the browser; the page
 * is served whole.
 */
export default function ClaimsIndexPage({core}: AtlasPageProps<Record<string, never>>): React.JSX.Element {
  const groups = useMemo(() => groupsOf(core), [core]);
  useRegisteredAnchors([...groups.map((group) => `group-${group.key}`), "group-audits"]);
  const audits = useMemo(
    () =>
      Object.values(core.claims)
        .filter((claim) => claim.kind === "audit")
        .map((claim) => claim.id)
        .sort((a, b) => a.localeCompare(b)),
    [core]
  );
  const [query, setQuery] = useState("");

  const keep = (ids: string[]): string[] => ids.filter((id) => {
    const claim = core.claims[id];
    return claim ? matches(claim, query) : false;
  });
  const shown = groups.map((group) => ({...group, sections: group.sections.map((section) => ({...section, ids: keep(section.ids)})).filter((section) => section.ids.length > 0)}));
  const auditsShown = keep(audits);
  const total = shown.reduce((sum, group) => sum + group.sections.reduce((count, section) => count + section.ids.length, 0), 0) + auditsShown.length;

  return (
    <AtlasShell
      description="Every statement in the research record, in the order of the chapters that derive them, with the open questions and the verification records."
      title="The claims"
    >
      <h1>The claims</h1>
      <p className="atlas__lead">
        The whole record, in the order it is written up: each chapter derives its results in sections, and a claim sits in the section that
        proves it. The open questions follow, grouped by the theme they belong to, and the verification records last.
      </p>

      <div className="atlas-filter">
        <label htmlFor="claims-filter">Filter by title or identifier</label>
        <input
          autoComplete="off"
          id="claims-filter"
          onChange={(event) => setQuery(event.target.value)}
          placeholder="exchange, leverage, DS19"
          type="search"
          value={query}
        />
        {query === "" ? null : <p className="atlas__muted">{total === 0 ? "Nothing matches." : `${total} matching.`}</p>}
      </div>

      {shown.map((group) => (
        <section key={group.key} aria-labelledby={`group-${group.key}`} className="atlas__section">
          <h2 id={`group-${group.key}`}>
            {group.label}
            {group.source === undefined ? null : (
              <a className="claims-index__source" href={group.source} rel="noopener">
                source
              </a>
            )}
          </h2>
          {group.sections.map((section) => (
            <div key={`${group.key}-${section.label ?? section.title}`} className="claims-index__section">
              {section.title === "" ? null : (
                <h3>
                  {section.label === null ? null : <span className="claims-index__label">{section.label}</span>}
                  {section.title}
                </h3>
              )}
              <EntityList compact core={core} ids={section.ids} />
            </div>
          ))}
        </section>
      ))}

      {auditsShown.length === 0 ? null : (
        <section aria-labelledby="group-audits" className="atlas__section">
          <h2 id="group-audits">Verification records</h2>
          <p>
            An independent re-derivation of a result already in the record. A verification record carries no result of its own; it says that
            someone went through the argument again and what they found.
          </p>
          <EntityList compact core={core} ids={auditsShown} />
        </section>
      )}
    </AtlasShell>
  );
}
