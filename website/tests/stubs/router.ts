/** `@docusaurus/router` for vitest: a fixed location and an inert history. */
export function useLocation(): {pathname: string; search: string; hash: string} {
  return {pathname: "/scorequant/portal/research/", search: "", hash: ""};
}

export function useHistory(): {push: (path: string) => void; replace: (path: string) => void} {
  return {push: () => undefined, replace: () => undefined};
}
