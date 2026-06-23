import { ShieldAlert, ShieldCheck } from 'lucide-react'
import { getRiskTier } from '../lib/risk'

// A ring gauge in place of a flat progress bar -- same data (probability as
// a fraction of the circle), but reads at a glance instead of as a strip.
function ProbabilityRing({ pct, color }) {
  const size = 80
  const stroke = 7
  const r = (size - stroke) / 2
  const circumference = 2 * Math.PI * r
  const offset = circumference * (1 - pct / 100)

  return (
    <div className="relative h-20 w-20 shrink-0">
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="#262626" strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke={color}
          strokeWidth={stroke}
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="font-mono text-sm font-bold" style={{ color }}>
          {pct.toFixed(0)}%
        </span>
      </div>
    </div>
  )
}

// Color-codes the prediction (red = illicit, green = licit) and shows the
// probability as a ring gauge, a precise number, and a tier tag. The tier
// color appears on the left accent border too -- that's risk severity, a
// real signal, not decoration.
export default function PredictionResult({ result }) {
  const isIllicit = result.prediction === 'illicit'
  const pct = result.probability_illicit * 100
  const tier = getRiskTier(result.probability_illicit)
  const TierIcon = tier.icon
  const PredictionIcon = isIllicit ? ShieldAlert : ShieldCheck

  return (
    <div
      className={`panel flex h-full flex-col border-l-2 p-3 ${tier.accentBorder}`}
    >
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs text-neutral-400">tx {result.tx_id}</span>
        <span
          className={`flex items-center gap-1 text-xs font-semibold uppercase tracking-wide ${isIllicit ? 'text-red-400' : 'text-emerald-400'}`}
        >
          <PredictionIcon className="h-3.5 w-3.5" />
          {result.prediction}
        </span>
      </div>

      <div className="mt-3 flex items-center gap-4">
        <ProbabilityRing pct={pct} color={tier.hex} />
        <div className="flex-1">
          <div className="text-[11px] font-medium uppercase tracking-wide text-neutral-400">
            Illicit probability
          </div>
          <div className={`font-mono text-2xl font-bold ${tier.text}`}>{pct.toFixed(1)}%</div>
          <span
            className={`mt-1.5 inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${tier.tag}`}
          >
            <TierIcon className="h-3 w-3" />
            {tier.label} risk
          </span>
        </div>
      </div>
    </div>
  )
}
