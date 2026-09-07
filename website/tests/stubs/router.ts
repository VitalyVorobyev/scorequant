/** Reactive browser history for URL-backed atlas controls in component tests. */
import {useSyncExternalStore} from "react";
const event = "test-router-change";
function subscribe(notify: () => void): () => void {
  window.addEventListener(event, notify);
  window.addEventListener("popstate", notify);
  return () => {
    window.removeEventListener(event, notify);
    window.removeEventListener("popstate", notify);
  };
}
export function useLocation(): {pathname: string; search: string; hash: string} {
  const href = useSyncExternalStore(
    subscribe,
    () => window.location.href,
    () => "http://localhost/scorequant/research/",
  );
  const url = new URL(href);
  return {pathname: url.pathname, search: url.search, hash: url.hash};
}
const history = {
  push: (path: string): void => {
    window.history.pushState(null, "", path);
    window.dispatchEvent(new Event(event));
  },
  replace: (path: string): void => {
    window.history.replaceState(null, "", path);
    window.dispatchEvent(new Event(event));
  },
};
export function useHistory(): typeof history {
  return history;
}
