import Link from "@docusaurus/Link";
import {useLocation} from "@docusaurus/router";
import Layout from "@theme/Layout";
import type {ReactNode} from "react";

const SECTIONS = [
  ["Map", "/research/map/"],
  ["Landscape", "/research/landscape/"],
  ["Frontier", "/research/frontier/"],
  ["Literature", "/research/literature/"],
  ["Machine-checked", "/research/machine-checked/"],
  ["In the library", "/research/library/"],
  ["Claims", "/research/claims/"],
  ["Counterexamples", "/research/counterexamples/"],
  ["How to read", "/research/how-to-read/"]
] as const;

export interface AtlasShellProps {
  children: ReactNode;
  description: string;
  title: string;
  /** Use the full page width rather than the reading measure (maps, grids). */
  wide?: boolean;
}

/**
 * The frame of every Research Atlas page: the site layout, one row of atlas
 * sub-navigation, and an article at the page width. Pages decide their own
 * inner measure; `wide` only removes the default reading cap on the article.
 */
export function AtlasShell({children, description, title, wide = false}: AtlasShellProps): React.JSX.Element {
  const {pathname} = useLocation();
  const active = (href: string): boolean => pathname.replace(/\/$/, "").endsWith(href.replace(/\/$/, "")) || pathname.includes(href);
  return (
    <Layout title={title} description={description}>
      <div className="atlas">
        <nav className="atlas-nav" aria-label="Research atlas">
          <Link to="/research/" className="atlas-nav__home">
            Research atlas
          </Link>
          {SECTIONS.map(([label, href]) => (
            <Link key={href} to={href} className={active(href) ? "is-active" : ""}>
              {label}
            </Link>
          ))}
        </nav>
        <article className={wide ? "atlas__wide" : "atlas__measure"}>{children}</article>
      </div>
    </Layout>
  );
}
