// Card previews from the Main artboard: "the picture you will see inside".
import type { Preview } from "../app/catalog";

export function ModelPreview({ pv, small }: { pv: Preview; small?: boolean }) {
  const st = small ? { width: 58, height: 40 } : undefined;
  return (
    <svg className="pv" viewBox="0 0 96 64" aria-hidden="true" style={st}>
      {BODY[pv]}
    </svg>
  );
}

const BODY: Record<Preview, JSX.Element> = {
  line: (
    <>
      <path className="ln sa w2" d="M8 54L88 12" />
      {[[14, 45], [24, 47], [33, 37], [44, 38], [52, 27], [63, 29], [72, 18], [82, 20]].map(([x, y]) => <circle key={x} className="f0" cx={x} cy={y} r="3" />)}
    </>
  ),
  coef: (
    <>
      <path className="ln si w1" d="M4 34H92" />
      <rect className="f0" x="9" y="12" width="9" height="22" rx="1.5" /><rect className="fi" x="23" y="33" width="9" height="2" />
      <rect className="f1" x="37" y="34" width="9" height="14" rx="1.5" /><rect className="fi" x="51" y="33" width="9" height="2" />
      <rect className="f0" x="65" y="22" width="9" height="12" rx="1.5" /><rect className="fi" x="79" y="33" width="9" height="2" />
    </>
  ),
  linb: (
    <>
      <path className="f0 o2" d="M0 0H96L0 64Z" /><path className="f1 o2" d="M96 0V64H0Z" /><path className="ln sa w15" d="M0 64L96 0" />
      <circle className="f0" cx="14" cy="14" r="3.5" /><circle className="f0" cx="28" cy="22" r="3.5" /><circle className="f0" cx="16" cy="36" r="3.5" /><circle className="f0" cx="40" cy="10" r="3.5" />
      <rect className="f1" x="57" y="45" width="7" height="7" rx="1" /><rect className="f1" x="72" y="36" width="7" height="7" rx="1" />
      <rect className="f1" x="80" y="51" width="7" height="7" rx="1" /><rect className="f1" x="64" y="30" width="7" height="7" rx="1" />
    </>
  ),
  tree: (
    <>
      <path className="ln si w15" d="M48 13L24 33M48 13L72 33M24 33L12 53M24 33L36 53M72 33L60 53M72 33L84 53" />
      <rect className="fi" x="39" y="7" width="18" height="11" rx="2" /><rect className="fi" x="15" y="28" width="18" height="11" rx="2" /><rect className="fi" x="63" y="28" width="18" height="11" rx="2" />
      <rect className="f1" x="4" y="49" width="16" height="10" rx="2" /><rect className="f0" x="28" y="49" width="16" height="10" rx="2" />
      <rect className="f0" x="52" y="49" width="16" height="10" rx="2" /><rect className="f1" x="76" y="49" width="16" height="10" rx="2" />
    </>
  ),
  forest: (
    <>
      <path className="ln si w15" d="M18 10L10 26M18 10L26 26M48 10L40 26M48 10L56 26M78 10L70 26M78 10L86 26" />
      <circle className="fi" cx="18" cy="10" r="4" /><circle className="fi" cx="48" cy="10" r="4" /><circle className="fi" cx="78" cy="10" r="4" />
      <circle className="f1" cx="10" cy="27" r="4" /><circle className="f0" cx="26" cy="27" r="4" /><circle className="f1" cx="40" cy="27" r="4" />
      <circle className="f1" cx="56" cy="27" r="4" /><circle className="f0" cx="70" cy="27" r="4" /><circle className="f1" cx="86" cy="27" r="4" />
      <rect className="f1" x="10" y="44" width="52" height="8" rx="2" /><rect className="f0" x="64" y="44" width="22" height="8" rx="2" />
    </>
  ),
  boost: (
    <>
      <path className="ln si w1" d="M4 56H92" />
      <rect className="fa" x="8" y="12" width="10" height="44" rx="1.5" />
      {[[22, 26, 30], [36, 36, 20], [50, 43, 13], [64, 47, 9], [78, 50, 6]].map(([x, y, h]) => <rect key={x} className="fa o5" x={x} y={y} width="10" height={h} rx="1.5" />)}
    </>
  ),
  svm: (
    <>
      <path className="ln sa w2" d="M6 64C30 42 52 34 92 2" /><path className="ln si w1 dsh" d="M-4 52C22 30 44 22 84 -8" /><path className="ln si w1 dsh" d="M16 76C40 54 62 46 100 14" />
      <circle className="f0" cx="14" cy="16" r="3.5" /><circle className="f0" cx="30" cy="10" r="3.5" /><circle className="f0" cx="22" cy="30" r="3.5" />
      <circle className="ln sk w15" cx="36" cy="36" r="6" /><circle className="f0" cx="36" cy="36" r="3.5" />
      <rect className="f1" x="64" y="44" width="7" height="7" rx="1" /><rect className="f1" x="78" y="34" width="7" height="7" rx="1" />
      <circle className="ln sk w15" cx="60.5" cy="32.5" r="6" /><rect className="f1" x="57" y="29" width="7" height="7" rx="1" />
    </>
  ),
  knn: (
    <>
      <circle className="ln si w1 dsh" cx="48" cy="32" r="20" />
      <rect className="f1" x="54" y="20" width="7" height="7" rx="1" /><rect className="f1" x="34" y="36" width="7" height="7" rx="1" />
      <rect className="f1" x="52" y="40" width="7" height="7" rx="1" /><rect className="f1" x="38" y="18" width="7" height="7" rx="1" />
      <circle className="f0" cx="62" cy="34" r="3.5" /><circle className="f0" cx="10" cy="12" r="3.5" /><circle className="f0" cx="86" cy="54" r="3.5" /><circle className="f0" cx="84" cy="12" r="3.5" />
      <circle className="fa" cx="48" cy="32" r="4" />
    </>
  ),
  nb: (
    <>
      <path className="ln si w1" d="M2 58H94" />
      <path className="f0 o2" d="M4 58C20 58 22 12 34 12S48 58 64 58Z" /><path className="ln s0 w2" d="M4 58C20 58 22 12 34 12S48 58 64 58" />
      <path className="f1 o2" d="M32 58C48 58 50 22 62 22S76 58 92 58Z" /><path className="ln s1 w2" d="M32 58C48 58 50 22 62 22S76 58 92 58" />
    </>
  ),
  net: (
    <>
      <path className="ln si w1 o5" d="M14 20L48 10M14 20L48 32M14 20L48 54M14 44L48 10M14 44L48 32M14 44L48 54M48 10L82 32M48 32L82 32M48 54L82 32" />
      <circle className="fs ln sk w15" cx="14" cy="20" r="5" /><circle className="fs ln sk w15" cx="14" cy="44" r="5" />
      <circle className="fa" cx="48" cy="10" r="5" /><circle className="fs ln sk w15" cx="48" cy="32" r="5" /><circle className="fa" cx="48" cy="54" r="5" /><circle className="fa" cx="82" cy="32" r="5" />
    </>
  ),
  cluster: (
    <>
      <circle className="f0" cx="16" cy="16" r="3.5" /><circle className="f0" cx="26" cy="12" r="3.5" /><circle className="f0" cx="22" cy="26" r="3.5" /><circle className="f0" cx="12" cy="26" r="3.5" />
      <rect className="f1" x="66" y="12" width="7" height="7" rx="1" /><rect className="f1" x="78" y="16" width="7" height="7" rx="1" /><rect className="f1" x="72" y="24" width="7" height="7" rx="1" />
      <path className="f2" d="M44 40l4-7 4 7z" /><path className="f2" d="M54 50l4-7 4 7z" /><path className="f2" d="M40 54l4-7 4 7z" /><path className="f2" d="M50 60l4-7 4 7z" />
      <path className="ln sk w2" d="M16 17l6 6M22 17l-6 6M72 18l6 6M78 18l-6 6M46 46l6 6M52 46l-6 6" />
    </>
  ),
  dbscan: (
    <>
      {[[14, 22], [22, 18], [30, 22], [38, 28], [44, 36], [20, 28]].map(([x, y]) => <circle key={`${x}${y}`} className="f0" cx={x} cy={y} r="3" />)}
      {[[60, 40], [68, 44], [76, 40], [72, 34], [80, 48]].map(([x, y]) => <rect key={`${x}${y}`} className="f1" x={x} y={y} width="6" height="6" rx="1" />)}
      <circle className="ln si w15" cx="70" cy="12" r="3" /><circle className="ln si w15" cx="10" cy="52" r="3" /><circle className="ln si w15" cx="48" cy="8" r="3" />
    </>
  ),
  pca: (
    <>
      {[[20, 50], [30, 42], [36, 46], [44, 34], [52, 32], [56, 24], [64, 26], [72, 16], [40, 28], [60, 34]].map(([x, y]) => <circle key={`${x}${y}`} className="fi o5" cx={x} cy={y} r="2.6" />)}
      <path className="ln sa w2" d="M48 32L82 8M48 32L56 42" />
    </>
  ),
};
