import Link from "@docusaurus/Link";
import {useLocation} from "@docusaurus/router";
import {useColorMode} from "@docusaurus/theme-common";
import clsx from "clsx";
import {useState} from "react";

import {isActiveNavEntry} from "../lib/navigation";
import {REFERENCE_BASE} from "../lib/site";
import {LiveFitProvider} from "./liveFit/LiveFitProvider";
import {Logo} from "./Logo";

const navigation = [
  ["Get started", "/get-started"],
  ["Walkthroughs", "/walkthroughs"],
  ["Ideas", "/ideas"],
  ["Research", "/research"],
  ["Reference", REFERENCE_BASE]
] as const;

/**
 * `useColorMode()` initializes `colorMode` to the configured `defaultMode`
 * on both the server render and React's first client render (see
 * `@docusaurus/theme-common`'s `colorMode` context: it reads the resolved
 * `data-theme` attribute only inside a `useEffect`, which never runs during
 * SSR or hydration), so the two renders agree and there is nothing here for
 * this component to guard against a mismatch itself -- it can read
 * `colorMode` directly.
 */
function ThemeToggle(): React.JSX.Element {
  const {colorMode, setColorMode} = useColorMode();
  const isDark = colorMode === "dark";
  return (
    <button
      className="theme-toggle"
      type="button"
      onClick={() => setColorMode(isDark ? "light" : "dark")}
      aria-pressed={isDark}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      <span aria-hidden="true">{isDark ? "☀" : "☾"}</span><span>{isDark ? "Light" : "Dark"}</span>
    </button>
  );
}

interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({children}: AppShellProps): React.JSX.Element {
  const {pathname} = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <LiveFitProvider>
      <div className="site-shell">
        <a className="skip-link" href="#main-content">Skip to content</a>
        <header className="site-header">
          <div className="site-header__inner">
            <Logo />
            <nav className={clsx("site-nav", menuOpen && "site-nav--open")} aria-label="Primary">
              {navigation.map(([label, href]) =>
                // The reference is a separately built MkDocs tree mounted outside the
                // Docusaurus app: it needs a real page load, not client-side routing,
                // and it can never be the "active" entry because reaching it always
                // means leaving this app. Every other entry stays a router Link with
                // the usual active-state highlighting.
                href === REFERENCE_BASE ? (
                  <a key={href} href={href}>
                    {label}
                  </a>
                ) : (
                  <Link key={href} to={href} className={isActiveNavEntry(pathname, href) ? "is-active" : ""}>
                    {label}
                  </Link>
                )
              )}
            </nav>
            <div className="site-actions">
              <ThemeToggle />
              <button
                className="menu-trigger"
                aria-expanded={menuOpen}
                aria-label="Toggle navigation"
                onClick={() => setMenuOpen((value) => !value)}
              >
                <span/><span/>
              </button>
            </div>
          </div>
        </header>
        <main id="main-content">{children}</main>
        <footer className="site-footer">
          <div>
            <Logo />
            <p>Hard bins, and a measurement of what the binning cost.</p>
          </div>
          <div className="site-footer__links">
            <span>Learn</span><Link to="/get-started">Get started</Link><Link to="/walkthroughs">Walkthroughs</Link>
          </div>
          <div className="site-footer__links">
            <span>Reference</span><a href={REFERENCE_BASE}>Documentation</a><a href="https://github.com/VitalyVorobyev/scorequant">GitHub</a>
          </div>
          <small>Open source · no browser data leaves your device</small>
        </footer>
      </div>
    </LiveFitProvider>
  );
}
