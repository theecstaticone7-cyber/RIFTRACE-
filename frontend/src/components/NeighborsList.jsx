import { ArrowDownLeft, ArrowUpRight, Share2 } from 'lucide-react'
import { MAX_RENDERED_NEIGHBORS } from '../lib/graph'

const CLASS_TEXT = {
  illicit: 'text-red-400',
  licit: 'text-emerald-400',
  unknown: 'text-neutral-400',
}

const DIRECTION_ICON = {
  incoming: ArrowDownLeft,
  outgoing: ArrowUpRight,
}

function ClassLabel({ knownClass }) {
  return <span className={CLASS_TEXT[knownClass] || CLASS_TEXT.unknown}>{knownClass}</span>
}

export default function NeighborsList({ graph }) {
  const shown = graph.neighbors.slice(0, MAX_RENDERED_NEIGHBORS)
  const hiddenCount = graph.num_neighbors - shown.length

  return (
    <div className="panel flex flex-col p-3">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="flex items-center gap-1.5 text-sm font-semibold text-neutral-100">
          <Share2 className="h-3.5 w-3.5 text-neutral-500" />
          Neighbors{' '}
          <span className="font-mono text-xs font-medium text-neutral-400">
            ({graph.num_neighbors})
          </span>
        </h3>
        <ClassLabel knownClass={graph.known_class} />
      </div>

      {graph.num_neighbors === 0 ? (
        <p className="py-4 text-center text-sm text-neutral-500">
          No connected transactions in the graph.
        </p>
      ) : (
        <div className="flex flex-col">
          <div className="max-h-64 overflow-y-auto rounded-md border border-neutral-800">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-neutral-900 text-left text-[11px] font-medium uppercase tracking-wide text-neutral-400">
                <tr>
                  <th className="px-2 py-1.5 font-medium">Tx ID</th>
                  <th className="px-2 py-1.5 font-medium">Direction</th>
                  <th className="px-2 py-1.5 font-medium">Class</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((n) => {
                  const DirectionIcon = DIRECTION_ICON[n.direction]
                  return (
                    <tr
                      key={`${n.tx_id}-${n.direction}`}
                      className="border-t border-neutral-800/60 hover:bg-neutral-800/40"
                    >
                      <td className="px-2 py-1.5 font-mono font-medium text-neutral-100">{n.tx_id}</td>
                      <td className="px-2 py-1.5 text-neutral-400">
                        <span className="inline-flex items-center gap-1">
                          <DirectionIcon className="h-3 w-3" />
                          {n.direction}
                        </span>
                      </td>
                      <td className="px-2 py-1.5">
                        <ClassLabel knownClass={n.known_class} />
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {hiddenCount > 0 && (
            <p className="mt-2 text-[11px] text-neutral-500">
              Showing first {MAX_RENDERED_NEIGHBORS} of {graph.num_neighbors} neighbors ({hiddenCount}{' '}
              more not shown).
            </p>
          )}
        </div>
      )}
    </div>
  )
}
