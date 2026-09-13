/**
 * Inline icon set.
 *
 * Hand-rolled rather than a dependency: eight 16px glyphs on a shared 24-unit grid weigh less
 * than an icon package and keep the stroke weight consistent. All are decorative — every icon
 * sits next to a text label or an `sr-only` name, so they are hidden from assistive tech.
 */

type IconProps = { className?: string };

function Svg({ className, children }: IconProps & { children: React.ReactNode }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      className={className ?? "h-4 w-4"}
    >
      {children}
    </svg>
  );
}

export const IconUpload = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 16V4m0 0L8 8m4-4 4 4" />
    <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
  </Svg>
);

export const IconDoc = (p: IconProps) => (
  <Svg {...p}>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
    <path d="M14 3v5h5" />
  </Svg>
);

export const IconClose = (p: IconProps) => (
  <Svg {...p}>
    <path d="M6 6l12 12M18 6 6 18" />
  </Svg>
);

export const IconExternal = (p: IconProps) => (
  <Svg {...p}>
    <path d="M14 4h6v6" />
    <path d="M20 4l-8 8" />
    <path d="M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4" />
  </Svg>
);

/** Marks the explainability affordance — the retrieval trace behind an answer. */
export const IconInsight = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="11" cy="11" r="6" />
    <path d="m20 20-4.5-4.5" />
    <path d="M11 8.5v2.5l1.75 1.25" />
  </Svg>
);

export const IconSend = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 12h15m0 0-6-6m6 6-6 6" />
  </Svg>
);

export const IconCheck = (p: IconProps) => (
  <Svg {...p}>
    <path d="m5 13 4 4L19 7" />
  </Svg>
);

export const IconCopy = (p: IconProps) => (
  <Svg {...p}>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M5 15V6a2 2 0 0 1 2-2h8" />
  </Svg>
);

export const IconAlert = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 8v4m0 3.5v.5" />
  </Svg>
);

export const IconMenu = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 7h16M4 12h16M4 17h16" />
  </Svg>
);

export const IconPlus = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 5v14M5 12h14" />
  </Svg>
);

export const IconHistory = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 12a8 8 0 1 0 2.35-5.65L4 8.7" />
    <path d="M4 4v4.7h4.7M12 8v4l2.5 1.5" />
  </Svg>
);

/** Wordmark glyph: stacked pages with a retrieval mark, drawn filled rather than stroked. */
export const LogoMark = ({ className }: IconProps) => (
  <svg
    viewBox="0 0 24 24"
    aria-hidden="true"
    focusable="false"
    className={className ?? "h-[22px] w-[22px]"}
  >
    <rect width="24" height="24" rx="6.5" fill="var(--color-accent)" />
    <path
      d="M7.5 7.25h6.25M7.5 10.5h9M7.5 13.75h5.5"
      stroke="#fff"
      strokeWidth="1.6"
      strokeLinecap="round"
      opacity="0.95"
    />
    <circle cx="15.6" cy="15.4" r="2.9" stroke="#fff" strokeWidth="1.6" fill="var(--color-accent)" />
    <path d="m17.9 17.7 1.9 1.9" stroke="#fff" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);
