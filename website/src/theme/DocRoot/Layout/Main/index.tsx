import type {Props} from "@theme/DocRoot/Layout/Main";
import type {ReactNode} from "react";

/**
 * The reading frame for the two docs-plugin routes, `/walkthroughs` and `/get-started`.
 *
 * Replaces the stock `DocRoot/Layout/Main` for the same reason `BlogLayout` replaces
 * the stock blog grid: the stock version renders a second `<main>` inside the one
 * `AppShell` already renders, which is invalid HTML and a duplicate-landmark failure
 * in the accessibility scan. It also brings Infima `container/padding--*` classes,
 * which are foreign to the portal's own measure and spacing tokens.
 *
 * `hiddenSidebarContainer` is deliberately unused: both instances are configured with
 * `sidebarPath: false`, so there is no sidebar to hide and no enhanced-width variant to
 * switch between. It stays in the signature because the theme passes it.
 */
export default function DocRootLayoutMain({children}: Props): ReactNode {
  return (
    <div className="docs-frame">
      <div className="docs-frame__content">{children}</div>
    </div>
  );
}
