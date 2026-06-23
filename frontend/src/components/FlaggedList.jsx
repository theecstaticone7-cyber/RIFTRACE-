import { useEffect, useState } from 'react'
import { Bar, BarChart, Cell } from 'recharts'
import { ChevronRight, Flag } from 'lucide-react'
import { getFlaggedTransactions } from '../api/riftrace'
import { getRiskTier } from '../lib/risk'
import ErrorBanner from './ErrorBanner'
import LoadingSpinner from './LoadingSpinner'

const LIMIT = 20

// A no-axes sparkline of the same probabilities the table rows already show
// -- a shape for "how risky is this batch overall", not a separate dataset.
function RiskDistribution({ flagged }) {
  const data = flagged.map((tx) => ({ value: tx.probability_illicit * 100 }))

  return (
    <BarChart width={96} height={28} data={data} barCategoryGap={1}>
      <Bar dataKey="value" isAnimationActive={false} radius={[1, 1, 0, 0]}>
        {data.map((d, i) => (
          <Cell key={i} fill={getRiskTier(d.value / 100).hex} />
        ))}
      </Bar>
    </BarChart>
  )
}

export default function FlaggedList({ onSelect, selectedTxId }) {
  const [flagged, setFlagged] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false

    getFlaggedTransactions(LIMIT)
      .then((data) => {
        if (!cancelled) setFlagged(data)
      })
      .catch((err) => {
        if (!cancelled) setError(err)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [])

  return (
    <section className="panel flex flex-col p-4">
      <div className="mb-2 flex items-start justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-semibold text-neutral-100">
            <Flag className="h-4 w-4 text-neutral-500" />
            Top flagged transactions
          </h2>
          <p className="text-xs text-neutral-500">
            Ranked by predicted illicit probability. Click a row to investigate.
          </p>
        </div>
        {flagged && flagged.flagged.length > 0 && (
          <div className="shrink-0 text-right">
            <p className="mb-0.5 text-[11px] font-medium uppercase tracking-wide text-neutral-500">
              Risk spread
            </p>
            <RiskDistribution flagged={flagged.flagged} />
          </div>
        )}
      </div>

      {loading && <LoadingSpinner label="Loading flagged transactions..." />}
      <ErrorBanner error={error} />

      {flagged && flagged.flagged.length === 0 && (
        <p className="text-sm text-neutral-500">No flagged transactions found.</p>
      )}

      {flagged && flagged.flagged.length > 0 && (
        <div className="max-h-[600px] overflow-y-auto rounded-md border border-neutral-800">
          <table className="w-full border-collapse text-sm">
            <thead className="sticky top-0 bg-neutral-900 text-left text-[11px] font-medium uppercase tracking-wide text-neutral-400">
              <tr className="border-b border-neutral-800">
                <th className="py-1.5 pl-2.5 pr-2 font-medium">Tx ID</th>
                <th className="py-1.5 pr-2 font-medium">Risk %</th>
                <th className="py-1.5 pr-2 font-medium">Tier</th>
                <th className="py-1.5 pr-2 font-medium">Actual</th>
                <th className="w-6 py-1.5" />
              </tr>
            </thead>
            <tbody>
              {flagged.flagged.map((tx) => {
                const tier = getRiskTier(tx.probability_illicit)
                const TierIcon = tier.icon
                const isSelected = selectedTxId === tx.tx_id
                // Ground truth, surfaced only when it disagrees with the
                // flag -- most rows are true positives and don't need an
                // extra marker.
                const isFalsePositive = tx.known_class !== 'illicit'

                return (
                  <tr
                    key={tx.tx_id}
                    onClick={() => onSelect(tx.tx_id)}
                    tabIndex={0}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter') onSelect(tx.tx_id)
                    }}
                    className={`group cursor-pointer border-b border-neutral-800/60 hover:bg-neutral-800/50 ${
                      isSelected ? 'bg-accent/20' : ''
                    }`}
                  >
                    <td className="py-1.5 pl-2.5 pr-2 font-mono font-medium text-neutral-100">
                      {tx.tx_id}
                    </td>
                    <td className={`py-1.5 pr-2 font-mono font-semibold ${tier.text}`}>
                      {(tx.probability_illicit * 100).toFixed(1)}
                    </td>
                    <td className="py-1.5 pr-2">
                      <span
                        className={`inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${tier.tag}`}
                      >
                        <TierIcon className="h-3 w-3" />
                        {tier.label}
                      </span>
                    </td>
                    <td className="py-1.5 pr-2 text-neutral-400">
                      {tx.known_class}
                      {isFalsePositive && <span className="ml-1 text-amber-400">(FP)</span>}
                    </td>
                    <td className="py-1.5 pr-2 text-neutral-600 group-hover:text-neutral-400">
                      <ChevronRight className="h-3.5 w-3.5" />
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {flagged && flagged.flagged.length > 0 && (
        <p className="mt-2 border-t border-neutral-800 pt-2 text-[11px] text-neutral-500">
          Showing {flagged.count} flagged transaction{flagged.count === 1 ? '' : 's'} from the
          temporal test set.
        </p>
      )}
    </section>
  )
}
