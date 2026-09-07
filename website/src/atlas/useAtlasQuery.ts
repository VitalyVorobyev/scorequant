import {useSyncExternalStore} from "react";
import {useHistory, useLocation} from "@docusaurus/router";

const subscribe = (): (() => void) => () => undefined;

/** URL-backed controls remain shareable and follow browser Back and Forward. */
export function useAtlasQuery(): {params: URLSearchParams; update: (values: Record<string, string>, replace?: boolean) => void} {
  const {pathname, search, hash} = useLocation();
  const history = useHistory();
  const hydrated = useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );
  const params = new URLSearchParams(hydrated ? search : "");
  const update = (values: Record<string, string>, replace = false): void => {
    const next = new URLSearchParams(search);
    for (const [key, value] of Object.entries(values)) {
      if (value) next.set(key, value);
      else next.delete(key);
    }
    const query = next.toString();
    history[replace ? "replace" : "push"](`${pathname}${query ? `?${query}` : ""}${hash}`);
  };
  return {params, update};
}
