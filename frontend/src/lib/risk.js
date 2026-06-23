import { ShieldAlert, TriangleAlert, Info, ShieldCheck } from 'lucide-react'

// Shared risk-tier classification, used by both the flagged-transactions
// table and the single-transaction prediction result so risk is color-coded
// consistently everywhere in the app. Color is reserved for actual risk
// severity here -- nothing decorative (no glow/shadow tokens).
//
// `hex` mirrors `dot`'s color one-for-one -- it's for contexts that can't
// take a Tailwind class (recharts fills, SVG stroke), so it must stay in
// sync with `dot` by hand.
//
// Tailwind class names below are written out in full (not built from
// template strings) because Tailwind's build-time scanner only generates
// CSS for class names it can find literally in source.
const TIERS = [
  {
    min: 0.9,
    label: 'Critical',
    text: 'text-red-400',
    dot: 'bg-red-500',
    hex: '#ef4444',
    tag: 'bg-red-900/70 text-red-100',
    accentBorder: 'border-l-red-600',
    icon: ShieldAlert,
  },
  {
    min: 0.7,
    label: 'High',
    text: 'text-orange-400',
    dot: 'bg-orange-500',
    hex: '#f97316',
    tag: 'bg-orange-900/70 text-orange-100',
    accentBorder: 'border-l-orange-600',
    icon: ShieldAlert,
  },
  {
    min: 0.5,
    label: 'Elevated',
    text: 'text-amber-400',
    dot: 'bg-amber-500',
    hex: '#f59e0b',
    tag: 'bg-amber-900/60 text-amber-100',
    accentBorder: 'border-l-amber-600',
    icon: TriangleAlert,
  },
  {
    min: 0.2,
    label: 'Low',
    text: 'text-neutral-400',
    dot: 'bg-neutral-500',
    hex: '#737373',
    tag: 'bg-neutral-700/70 text-neutral-200',
    accentBorder: 'border-l-neutral-600',
    icon: Info,
  },
  {
    min: 0,
    label: 'Minimal',
    text: 'text-emerald-400',
    dot: 'bg-emerald-500',
    hex: '#10b981',
    tag: 'bg-emerald-900/60 text-emerald-100',
    accentBorder: 'border-l-emerald-600',
    icon: ShieldCheck,
  },
]

export function getRiskTier(probabilityIllicit) {
  return TIERS.find((tier) => probabilityIllicit >= tier.min) ?? TIERS[TIERS.length - 1]
}
