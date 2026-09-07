import Link from "@docusaurus/Link";
import {useLocation} from "@docusaurus/router";
import Layout from "@theme/Layout";
import {useEffect, useRef, useState, type ReactNode} from "react";

const SECTIONS = [
  ["Research", "/research/"],
  ["Explore", "/research/landscape/"],
  ["Literature", "/research/literature/"],
  ["Frontier", "/research/frontier/"],
] as const;
const TOOLS = [
  ["Graph", "/research/map/"],
  ["Claims registry", "/research/claims/"],
  ["Counterexamples", "/research/counterexamples/"],
  ["Machine-checked", "/research/machine-checked/"],
  ["Library mapping", "/research/library/"],
  ["How to read", "/research/how-to-read/"],
] as const;
export interface AtlasShellProps {
  children: ReactNode;
  description: string;
  title: string;
  wide?: boolean;
}

/** Shared research navigation and fragment-aware progressive disclosure. */
export function AtlasShell({children, description, title, wide = false}: AtlasShellProps): React.JSX.Element {
  const {pathname, hash} = useLocation();
  const root = useRef<HTMLDivElement>(null);
  const [menu, setMenu] = useState(false);
  useEffect(() => {
    const reveal = (): void => {
      let id: string;
      try {
        id = decodeURIComponent(window.location.hash.slice(1));
      } catch {
        return;
      }
      if (!id) return;
      const target = document.getElementById(id);
      if (!target || !root.current?.contains(target)) return;
      if (id === "proof")
        target.querySelectorAll<HTMLDetailsElement>("details.proof").forEach((details) => {
          details.open = true;
        });
      let ancestor: HTMLElement | null = target;
      while (ancestor) {
        if (ancestor instanceof HTMLDetailsElement) ancestor.open = true;
        ancestor = ancestor.parentElement;
      }
      if (typeof target.scrollIntoView === "function") target.scrollIntoView({block: "start"});
    };
    reveal();
    window.addEventListener("hashchange", reveal);
    return () => window.removeEventListener("hashchange", reveal);
  }, [hash, pathname]);
  const active = (href: string): boolean => (href === "/research/" ? pathname.replace(/\/$/, "").endsWith("/research") : pathname.includes(href));
  return (
    <Layout title={title} description={description}>
      <div className="atlas" ref={root}>
        <nav className="atlas-nav" aria-label="Research atlas">
          <button type="button" className="atlas-nav__toggle" aria-expanded={menu} aria-controls="research-navigation" onClick={() => setMenu(!menu)}>
            Research navigation
          </button>
          <div id="research-navigation" className={`atlas-nav__links${menu ? " is-open" : ""}`}>
            {SECTIONS.map(([label, href]) => (
              <Link key={href} to={href} aria-current={active(href) ? "page" : undefined} className={active(href) ? "is-active" : ""}>
                {label}
              </Link>
            ))}
            <details className="atlas-tools">
              <summary>Research tools</summary>
              <div>
                {TOOLS.map(([label, href]) => (
                  <Link key={href} to={href} aria-current={active(href) ? "page" : undefined}>
                    {label}
                  </Link>
                ))}
              </div>
            </details>
          </div>
        </nav>
        <article className={wide ? "atlas__wide" : "atlas__measure"}>{children}</article>
      </div>
    </Layout>
  );
}
