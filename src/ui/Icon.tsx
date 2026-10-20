// Stroke icons from the design (24×24, 1.7 stroke). SVG is fine here: small and static.
import type { CSSProperties } from "react";

const P: Record<string, string> = {
  logo: "M12 3.5a8.5 8.5 0 1 0 0 17a8.5 8.5 0 1 0 0-17ZM3.5 12h17M12 9.5a2.5 2.5 0 1 0 0 5a2.5 2.5 0 1 0 0-5Z",
  min: "M6 12h12",
  max: "M7.5 6.5h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1h-9a1 1 0 0 1-1-1v-9a1 1 0 0 1 1-1Z",
  restore: "M9 6.5h8.5v8.5M6.5 9h8.5v8.5H6.5Z",
  close: "M7 7l10 10M17 7L7 17",
  back: "M15 6l-6 6 6 6",
  moon: "M19 14.5A7.5 7.5 0 0 1 9.5 5a7.5 7.5 0 1 0 9.5 9.5z",
  sun: "M12 8a4 4 0 1 0 0 8a4 4 0 1 0 0-8ZM12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M5.6 18.4L7 17M17 7l1.4-1.4",
  system: "M4 5.5h16v10H4ZM9 19.5h6M12 15.5v4",
  help: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18ZM9.6 9.4a2.5 2.5 0 1 1 3.4 2.4c-.6.3-1 .8-1 1.5v.5M12 17v.01",
  search: "M11 4.5a6.5 6.5 0 1 0 0 13a6.5 6.5 0 1 0 0-13ZM16 16l4 4",
  clock: "M12 3.5a8.5 8.5 0 1 0 0 17a8.5 8.5 0 1 0 0-17ZM12 7.5V12l3 2",
  info: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18ZM12 11v5M12 8v.01",
  play: "M8 5.5l10 6.5-10 6.5z",
  pause: "M9 6v12M15 6v12",
  pauseTag: "M8 5v14M16 5v14",
  step: "M6 6l8 6-8 6zM18 6v12",
  reset: "M4.5 12a7.5 7.5 0 1 0 2.2-5.3M4.5 4.5v4h4",
  sliders: "M4 7h9M17 7h3M4 17h3M11 17h9M15 5a2 2 0 1 0 0 4a2 2 0 1 0 0-4ZM9 15a2 2 0 1 0 0 4a2 2 0 1 0 0-4Z",
  path: "M6 16a2 2 0 1 0 0 4a2 2 0 1 0 0-4ZM18 4a2 2 0 1 0 0 4a2 2 0 1 0 0-4ZM8 18h6a3 3 0 0 0 0-6h-4a3 3 0 0 1 0-6h6",
  check: "M5 12l5 5 9-10",
  checkCircle: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18ZM8 12l3 3 5-6",
  plus: "M12 5v14M5 12h14",
  minus: "M5 12h14",
  upload: "M12 16V4M7 9l5-5 5 5M5 20h14",
  download: "M12 4v12M7 11l5 5 5-5M5 20h14",
  target: "M12 4a8 8 0 1 0 0 16a8 8 0 1 0 0-16ZM12 9a3 3 0 1 0 0 6a3 3 0 1 0 0-6Z",
  crosshair: "M12 5a7 7 0 1 0 0 14a7 7 0 1 0 0-14ZM12 2v4M12 18v4M2 12h4M18 12h4",
  shuffle: "M4 7h3c5 0 5 10 10 10h3M4 17h3c2 0 3-1.5 4-3.5M13 9.5c1-2 2-2.5 4-2.5h3M17 4l3 3-3 3M17 14l3 3-3 3",
  undo: "M9 14l-5-5 5-5M4 9h11a5 5 0 0 1 0 10h-3",
  redo: "M15 14l5-5-5-5M20 9H9a5 5 0 0 0 0 10h3",
  warn: "M12 4l9 16H3zM12 10v4M12 17v.01",
  table: "M4 5h16v14H4zM4 10h16M10 10v9",
  expand: "M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5",
  shrink: "M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5",
  popout: "M14 4h6v6M20 4l-8 8M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
  grip: "M9 6h.01M15 6h.01M9 12h.01M15 12h.01M9 18h.01M15 18h.01",
  arrowL: "M19 12H5M11 6l-6 6 6 6",
  arrowR: "M5 12h14M13 6l6 6-6 6",
  chevUp: "M6 15l6-6 6 6",
  chevDown: "M6 9l6 6 6-6",
  trash: "M5 7h14M10 11v6M14 11v6M7 7l1 12h8l1-12M10 7V4h4v3",
  layout: "M4 5h16v14H4zM4 12h16M12 12v7",
  edit: "M4 20h4L19 9l-4-4L4 16zM13.5 6.5l4 4",
  book: "M12 6.5C10.5 5.3 8.3 4.5 4.5 4.5v13c3.8 0 6 .8 7.5 2 1.5-1.2 3.7-2 7.5-2v-13c-3.8 0-6 .8-7.5 2ZM12 6.5v13",
  copy: "M9 9h10v10H9zM15 9V5H5v10h4",
};

export function Icon({ name, size = 16, style, fill }: { name: keyof typeof P | string; size?: number; style?: CSSProperties; fill?: boolean }) {
  return (
    <svg className="i" viewBox="0 0 24 24" aria-hidden="true" style={{ width: size, height: size, ...(fill ? { fill: "currentColor" } : null), ...style }}>
      <path d={P[name] ?? ""} />
    </svg>
  );
}
