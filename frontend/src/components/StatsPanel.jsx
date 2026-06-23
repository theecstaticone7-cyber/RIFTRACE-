import { useEffect, useState } from 'react'
import { Cell, Pie, PieChart } from 'recharts'
import { Database, Gauge, ShieldAlert } from 'lucide-react'
import { CLASS_COLOR_HEX } from '../lib/graph'
import { getStats } from '../api/riftrace'
import ErrorBanner from './ErrorBanner'
import LoadingSpinner from './LoadingSpinner'

// Individual tiles (not one divided strip) so each metric reads as its own
// unit -- subtle background, label quiet and small, value large and bright.
function MetricGroup({ items }) {
  return (
    <div className="flex flex-wrap gap-2">
      {items.map(({ label, value }) => (
        <div key={label} className="tile min-w-[100px] px-3 py-2">
          <div className="text-[11px] font-medium uppercase tracking-wide text-neutral-400">
            {label}
          </div>
          <div className="font-mono text-base font-semibold text-neutral-100">{value}</div>
        </div>
      ))}
    </div>
  )
}

// Same red/emerald/neutral mapping used everywhere else known_class shows
// up (see lib/graph.js) -- unknown stays neutral, never a "third accent".
function ClassSplitChart({ dataset }) {
  const data = [
    { name: 'Illicit', value: dataset.pct_illicit, color: CLASS_COLOR_HEX.illicit },
    { name: 'Licit', value: dataset.pct_licit, color: CLASS_COLOR_HEX.licit },
    { name: 'Unknown', value: dataset.pct_unknown, color: CLASS_COLOR_HEX.unknown },
  ]

  return (
    <div className="tile flex items-center gap-4 px-4 py-3">
      <PieChart width={84} height={84}>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          innerRadius={26}
          outerRadius={40}
          paddingAngle={2}
          stroke="none"
          isAnimationActive={false}
        >
          {data.map((d) => (
            <Cell key={d.name} fill={d.color} />
          ))}
        </Pie>
      </PieChart>
      <ul className="space-y-1.5">
        {data.map((d) => (
          <li key={d.name} className="flex items-center gap-2 text-xs">
            <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: d.color }} />
            <span className="w-14 text-neutral-400">{d.name}</span>
            <span className="font-mono font-semibold text-neutral-100">{d.value.toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default function StatsPanel() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    // Guards against setting state after the component unmounts (e.g. fast
    // navigation away) -- not a real concern in this single-page layout,
    // but a cheap, standard precaution for fetch-in-useEffect.
    let cancelled = false

    getStats()
      .then((data) => {
        if (!cancelled) setStats(data)
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
    <section className="panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Gauge className="h-4 w-4 text-neutral-500" />
          <div>
            <h2 className="text-sm font-semibold text-neutral-100">Model &amp; dataset metrics</h2>
            <p className="text-xs text-neutral-500">
              Computed on the temporal test split -- never randomly resampled.
            </p>
          </div>
        </div>
        {stats && <span className="font-mono text-xs text-neutral-400">{stats.model_name}</span>}
      </div>

      {loading && <LoadingSpinner label="Loading stats..." />}
      <ErrorBanner error={error} />

      {stats && (
        <div className="space-y-3">
          <div>
            <h3 className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-neutral-500">
              <ShieldAlert className="h-3 w-3" />
              Illicit-class performance
            </h3>
            <MetricGroup
              items={[
                { label: 'Precision', value: stats.metrics.precision.toFixed(3) },
                { label: 'Recall', value: stats.metrics.recall.toFixed(3) },
                { label: 'F1', value: stats.metrics.f1.toFixed(3) },
                { label: 'ROC-AUC', value: stats.metrics.roc_auc.toFixed(3) },
              ]}
            />
          </div>

          <div>
            <h3 className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-neutral-500">
              <Database className="h-3 w-3" />
              Dataset
            </h3>
            <div className="flex flex-wrap gap-2">
              <MetricGroup
                items={[
                  { label: 'Nodes', value: stats.dataset.total_nodes.toLocaleString() },
                  { label: 'Edges', value: stats.dataset.total_edges.toLocaleString() },
                  { label: 'Time steps', value: stats.dataset.num_time_steps },
                  { label: 'Features', value: stats.dataset.feature_count },
                ]}
              />
              <ClassSplitChart dataset={stats.dataset} />
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
